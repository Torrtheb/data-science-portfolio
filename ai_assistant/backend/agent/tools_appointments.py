from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

from langchain_core.tools import tool, ToolException
from langchain_core.runnables import RunnableConfig
from sqlalchemy import and_, or_, func

from app.db import SessionLocal
from app.models import (
    User,
    Appointment,
    Person,
    ClientAccount,
    PrepaidBundle,
    PrepaidLedger,
)
from services.services_scheduling import generate_daily_slots, is_owner_time_bookable
from services.payments import compute_price_cents, _service_price_map

from agent.schemas import (
    ToolFindSlotsIn,
    ToolFindSlotsOut,
    ToolCreateApptIn,
    ToolCreateApptOut,
    ToolCancelAppointmentIn,
    ToolCancelAppointmentOut,
    ToolUpdateAppointmentIn,
    ToolUpdateAppointmentOut,
    ToolGetNextApptIn,
    ToolGetNextApptOut,
    ToolListAppointmentsIn,
    ToolListAppointmentsOut,
    ToolUpdateApptDetailsIn,
    ToolUpdateApptDetailsOut,
    ToolGetAppointmentDetailsIn,
    ToolGetAppointmentDetailsOut,
    ToolRescheduleApptIn,
    ToolRescheduleApptOut,
    ToolListPostApptActionsOut,
)
from agent.tool_utils import (
    _parse_owner_day,
    _parse_owner_local_dt,
    _to_utc,
    _owner_id_from_config,
    _set_if_column,
)
from agent.tool_ops import resolve_person, attach_identity_to_appointment
from routers._helpers import build_appt_email, send_email

log = logging.getLogger(__name__)


@tool("find_slots", args_schema=ToolFindSlotsIn, return_direct=False)
def find_slots_tool(
    day: str, duration_minutes: int, config: RunnableConfig
) -> ToolFindSlotsOut:
    """
    Return owner-local open slots (exact duration match) for a given day.
    """
    log.info("find_slots_tool(%s, %s)", day, duration_minutes)
    try:
        cfg = (
            (config or {}).get("configurable", {})
            if isinstance(config, dict)
            else (getattr(config, "configurable", None) or {})
        )
        owner_id = cfg.get("user_id") or cfg.get("owner_id")
        if not owner_id:
            raise ToolException("Missing owner id in tool config")

        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                return ToolFindSlotsOut(slots=[])

            try:
                day_date = _parse_owner_day(day, owner.timezone)
            except Exception:
                return ToolFindSlotsOut(slots=[])

            # Base discrete slots for the day (weekly rules, minus time-offs/appointments)
            base = generate_daily_slots(db, owner, day_date)

            # Coalesce adjacent/overlapping into continuous windows in owner-local time
            merged: List[tuple[datetime, datetime]] = []
            for s, e in sorted(base, key=lambda p: p[0]):
                if not merged:
                    merged.append((s, e))
                else:
                    ps, pe = merged[-1]
                    if s <= pe:
                        merged[-1] = (ps, max(pe, e))
                    else:
                        merged.append((s, e))

            # Provide sliding options on a reasonable grid, so hour-aligned times appear
            dur = timedelta(minutes=int(duration_minutes))
            grid_minutes = 30 if (int(duration_minutes) % 30 == 0) else 15
            grid = timedelta(minutes=grid_minutes)

            out: List[Dict] = []
            for ws, we in merged:
                cur = ws
                minutes_from_midnight = cur.hour * 60 + cur.minute
                rem = minutes_from_midnight % grid_minutes
                if rem != 0:
                    bump = grid_minutes - rem
                    cur = cur + timedelta(minutes=bump)
                while cur + dur <= we:
                    out.append(
                        {
                            "start_local": cur.replace(tzinfo=None).isoformat(
                                timespec="minutes"
                            ),
                            "end_local": (cur + dur)
                            .replace(tzinfo=None)
                            .isoformat(timespec="minutes"),
                        }
                    )
                    cur = cur + grid
            seen = set()
            uniq = []
            for row in out:
                key = (row["start_local"], row["end_local"])
                if key not in seen:
                    seen.add(key)
                    uniq.append(row)
            uniq.sort(key=lambda r: r["start_local"])

            return ToolFindSlotsOut(slots=uniq)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"find_slots failed: {e}")


@tool("create_appointment", args_schema=ToolCreateApptIn, return_direct=False)
def create_appointment_tool(
    person_id: Optional[int] = None,
    client_name: Optional[str] = None,
    client_email: Optional[str] = None,
    start_local: str = "",
    duration_minutes: int = 0,
    note: Optional[str] = None,
    service_option_id: Optional[int] = None,
    config: RunnableConfig = None,
) -> ToolCreateApptOut:
    """
    Book an appointment if the requested window is free.
    Enforces identity: person_id OR client_email OR client_name.
    """
    log.info(
        "create_appointment_tool(name=%r, email=%r, person_id=%r, start=%s, dur=%s)",
        client_name,
        client_email,
        person_id,
        start_local,
        duration_minutes,
    )
    try:
        cfg = (
            (config or {}).get("configurable", {})
            if isinstance(config, dict)
            else (getattr(config, "configurable", None) or {})
        )
        owner_id = cfg.get("user_id") or cfg.get("owner_id")
        if not owner_id:
            raise ToolException("Missing owner id in tool config")

        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            if not (person_id or client_email or client_name):
                raise ToolException(
                    "Appointment identity required: provide person_id or client_email or client_name"
                )

            start_loc = _parse_owner_local_dt(start_local, owner.timezone)
            end_loc = start_loc + timedelta(minutes=int(duration_minutes))
            start_utc, end_utc = _to_utc(start_loc), _to_utc(end_loc)

            # Allow any free time within openings (not only pre-sliced discrete slots)
            ok, reason = is_owner_time_bookable(
                db, owner, _to_utc(start_loc), _to_utc(end_loc)
            )
            if not ok:
                raise ToolException(f"Requested time is not available: {reason}")
            if person_id:
                p = (
                    db.query(Person)
                    .join(ClientAccount, ClientAccount.id == Person.account_id)
                    .filter(
                        Person.id == person_id, ClientAccount.owner_user_id == owner.id
                    )
                    .first()
                )
                if not p:
                    raise ToolException("Person not found for this owner")
                canonical_name = p.full_name
                canonical_email = p.email or (client_email or "")
                person = p
            else:
                person, canonical_name, canonical_email = resolve_person(
                    db=db,
                    owner_id=owner.id,
                    client_name=client_name,
                    client_email=client_email,
                )

            cols = {c.name for c in Appointment.__table__.columns}
            appt = Appointment()
            if "owner_id" in cols:
                appt.owner_id = owner.id
            if "start_utc" in cols:
                appt.start_utc = start_utc
            elif "start_time_utc" in cols:
                appt.start_time_utc = start_utc
            else:
                raise ToolException("Appointment model is missing a start UTC column")
            if "end_utc" in cols:
                appt.end_utc = end_utc
            elif "end_time_utc" in cols:
                appt.end_time_utc = end_utc
            else:
                raise ToolException("Appointment model is missing an end UTC column")
            if "status" in cols:
                appt.status = "booked"
            if note is not None:
                if "owner_private_note" in cols:
                    appt.owner_private_note = note
                elif "note" in cols:
                    appt.note = note
            final_email = canonical_email or (person.email if person else None)
            if not final_email and person is not None:
                from agent.tool_ops import _primary_account_email as _acct_email

                final_email = _acct_email(db, person.account_id)
            if not final_email:
                raise ToolException(
                    "NEED_CLIENT_EMAIL: This client has no email on file. Please provide client_email (or choose a person with an email) to proceed."
                )

            attach_identity_to_appointment(
                appt,
                person=person,
                canonical_name=canonical_name,
                canonical_email=final_email,
                db=db,
            )

            db.add(appt)
            db.commit()
            db.refresh(appt)

            # Try to attach client_id via the person's account for notification parity
            try:
                if person is not None and getattr(person, "account_id", None):
                    acct = (
                        db.query(ClientAccount)
                        .filter(ClientAccount.id == person.account_id)
                        .first()
                    )
                    if acct and getattr(acct, "client_user_id", None):
                        appt.client_id = acct.client_user_id
                        db.add(appt)
                        db.commit()
                        db.refresh(appt)
                        log.info(
                            "AGENT_EMAIL_LINKED_CLIENT",
                            extra={
                                "event": "agent_email_linked_client",
                                "appointment_id": str(appt.id),
                                "client_user_id": str(acct.client_user_id),
                            },
                        )
            except Exception as e:
                log.warning("AGENT_EMAIL_LINK_CLIENT_FAILED %s", e)

            # Auto-apply wallet funds (store credit) to this client's outstanding items,
            # so this new appointment is covered immediately when possible.
            try:
                client_user_id = getattr(appt, "client_id", None)
                if (
                    not client_user_id
                    and person is not None
                    and getattr(person, "account_id", None)
                ):
                    acct = (
                        db.query(ClientAccount)
                        .filter(ClientAccount.id == person.account_id)
                        .first()
                    )
                    if acct and getattr(acct, "client_user_id", None):
                        client_user_id = acct.client_user_id
                if client_user_id:
                    wallet_ids = (
                        db.query(PrepaidBundle.id)
                        .filter(
                            PrepaidBundle.owner_id == owner.id,
                            PrepaidBundle.client_id == str(client_user_id),
                            PrepaidBundle.total_credits == 0,
                        )
                        .order_by(PrepaidBundle.created_at.desc())
                        .all()
                    )
                    for (wid,) in wallet_ids:
                        auto_apply_wallet_funds = __import__(
                            "services.wallets", fromlist=["auto_apply_wallet_funds"]
                        ).auto_apply_wallet_funds
                        auto_apply_wallet_funds(
                            db,
                            owner_id=str(owner.id),
                            bundle_id=int(wid),
                            note_prefix="Auto-apply wallet funds after agent booking",
                        )
            except Exception:
                pass

            # Notify client only if a verified user email exists
            try:
                client = (
                    db.get(User, appt.client_id)
                    if getattr(appt, "client_id", None)
                    else None
                )
                if (
                    client
                    and getattr(client, "email", None)
                    and getattr(client, "emailVerified", None)
                ):
                    tz = ZoneInfo(owner.timezone)
                    start_local = appt.start_utc.astimezone(tz)
                    end_local = appt.end_utc.astimezone(tz)
                    recipient_name = (
                        getattr(client, "name", None)
                        or getattr(appt, "client_name", None)
                        or client.email
                    )
                    pkg = build_appt_email(
                        audience="client",
                        action="created",
                        owner=owner,
                        start_local=start_local,
                        end_local=end_local,
                        appointment_id=str(appt.id),
                        initiator_label=owner.name or "the owner",
                        status_label=getattr(appt, "status", "booked"),
                        recipient_name=recipient_name,
                        message=note,
                        include_ics=True,
                        organizer_email=owner.email,
                        attendee_email=client.email,
                    )
                    send_email(
                        client.email, pkg.subject, pkg.text, pkg.html, pkg.ics_text
                    )
                    log.info(
                        "AGENT_EMAIL_SENT",
                        extra={
                            "event": "agent_email_sent",
                            "action": "created",
                            "appointment_id": str(appt.id),
                            "client_user_id": str(client.id),
                            "email": client.email,
                        },
                    )
                else:
                    reason = (
                        "no_client"
                        if not client
                        else (
                            "unverified_or_missing_email"
                            if not getattr(client, "emailVerified", None)
                            or not getattr(client, "email", None)
                            else "unknown"
                        )
                    )
                    log.info(
                        "AGENT_EMAIL_SKIPPED",
                        extra={
                            "event": "agent_email_skipped",
                            "action": "created",
                            "appointment_id": str(appt.id),
                            "client_user_id": (str(client.id) if client else None),
                            "reason": reason,
                        },
                    )
            except Exception as e:
                # Do not fail booking if email fails
                log.error("AGENT_EMAIL_SEND_FAILED %s", e, exc_info=True)

            return ToolCreateApptOut(
                appointment_id=str(appt.id),
                start_utc=start_utc,
                end_utc=end_utc,
                status=getattr(appt, "status", "booked"),
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"create_appointment failed: {e}")


@tool(
    "update_appointment_details",
    args_schema=ToolUpdateApptDetailsIn,
    return_direct=False,
)
def update_appointment_details_tool(
    appointment_id: str,
    person_id: Optional[int] = None,
    client_email: Optional[str] = None,
    client_name: Optional[str] = None,
    note: Optional[str] = None,
    price_override_cents: Optional[int] = None,
    config: Optional[RunnableConfig] = None,
) -> ToolUpdateApptDetailsOut:
    """
    Attach/fix WHO an appointment is for (and optionally note/price).
    Guarantees person_id + denormalized name/email after success.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            appt: Optional[Appointment] = (
                db.query(Appointment)
                .filter(
                    Appointment.id == appointment_id, Appointment.owner_id == owner_id
                )
                .first()
            )
            if not appt:
                raise ToolException("Appointment not found")

            if not (person_id or client_email or client_name):
                raise ToolException(
                    "Appointment identity required: provide person_id or client_email or client_name"
                )

            if person_id:
                p = (
                    db.query(Person)
                    .join(ClientAccount, ClientAccount.id == Person.account_id)
                    .filter(
                        Person.id == person_id, ClientAccount.owner_user_id == owner_id
                    )
                    .first()
                )
                if not p:
                    raise ToolException("Person not found for this owner")
                canonical_name = p.full_name
                canonical_email = p.email or (client_email or "")
                person = p
            else:
                person, canonical_name, canonical_email = resolve_person(
                    db=db,
                    owner_id=owner_id,
                    client_name=client_name,
                    client_email=client_email,
                )

            attach_identity_to_appointment(
                appt,
                person=person,
                canonical_name=canonical_name,
                canonical_email=canonical_email,
                db=db,
            )

            cols = {c.name for c in Appointment.__table__.columns}
            if note is not None:
                if "owner_private_note" in cols:
                    appt.owner_private_note = note
                elif "note" in cols:
                    appt.note = note
            if price_override_cents is not None and "price_override_cents" in cols:
                appt.price_override_cents = int(price_override_cents)

            db.add(appt)
            db.commit()
            db.refresh(appt)

            return ToolUpdateApptDetailsOut(
                ok=True,
                appointment_id=str(appt.id),
                person_id=(person.id if person else appt.person_id),
                client_name=appt.client_name,
                client_email=appt.client_email,
            )

    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"update_appointment_details failed: {e}")


@tool("cancel_appointment", args_schema=ToolCancelAppointmentIn, return_direct=False)
def cancel_appointment_tool(
    appointment_id: Optional[str] = None,
    start_local: Optional[str] = None,
    duration_minutes: Optional[int] = None,
    reason: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
) -> ToolCancelAppointmentOut:
    """
    Cancel an existing appointment by id, or by (start_local + duration_minutes).
    """
    log.info(
        "cancel_appointment_tool(%s, %s, %s)",
        appointment_id,
        start_local,
        duration_minutes,
    )
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            appt: Optional[Appointment] = None

            if appointment_id:
                appt = (
                    db.query(Appointment)
                    .filter(
                        and_(
                            Appointment.id == appointment_id,
                            Appointment.owner_id == owner.id,
                        )
                    )
                    .first()
                )
            else:
                if not start_local:
                    raise ToolException(
                        "Provide appointment_id OR start_local (and duration if known)"
                    )

                s_loc = _parse_owner_local_dt(start_local, owner.timezone)
                cols = {c.name for c in Appointment.__table__.columns}
                start_col = "start_utc" if "start_utc" in cols else "start_time_utc"
                end_col = "end_utc" if "end_utc" in cols else "end_time_utc"

                s_utc = _to_utc(s_loc)
                e_utc = None
                if duration_minutes is not None:
                    e_loc = s_loc + timedelta(minutes=int(duration_minutes))
                    e_utc = _to_utc(e_loc)

                # Prefer exact match. If not found, fall back to a 1-minute tolerance on start time
                q = db.query(Appointment).filter(Appointment.owner_id == owner.id)
                exact = q.filter(getattr(Appointment, start_col) == s_utc)
                if e_utc is not None:
                    exact = exact.filter(getattr(Appointment, end_col) == e_utc)
                appt = exact.first()

                if not appt:
                    # Tolerate small parsing/seconds differences: start within [s_utc, s_utc+60s)
                    tol_q = q.filter(
                        getattr(Appointment, start_col) >= s_utc,
                        getattr(Appointment, start_col)
                        < (s_utc + timedelta(minutes=1)),
                    )
                    if e_utc is not None:
                        tol_q = tol_q.filter(
                            getattr(Appointment, end_col) >= e_utc,
                            getattr(Appointment, end_col)
                            < (e_utc + timedelta(minutes=1)),
                        )
                    appt = tol_q.order_by(getattr(Appointment, start_col).asc()).first()

            if not appt:
                raise ToolException("Appointment not found")

            # Determine if cancellation is >= 24h before start in owner's timezone
            try:
                owner_tz = (
                    ZoneInfo(owner.timezone)
                    if owner and getattr(owner, "timezone", None)
                    else ZoneInfo("UTC")
                )
            except Exception:
                owner_tz = ZoneInfo("UTC")
            now_local = datetime.now(owner_tz)
            appt_start_local = appt.start_utc.astimezone(owner_tz)
            qualifies_full_refund = (appt_start_local - now_local) >= timedelta(
                hours=24
            )
            appt.status = "canceled"
            # Apply full refund to client's wallet (cash portion) if >=24h before start
            if qualifies_full_refund:
                if getattr(appt, "bundle_id", None):
                    bid = int(appt.bundle_id)
                    net = (
                        db.query(
                            func.coalesce(func.sum(PrepaidLedger.delta_credits), 0)
                        )
                        .filter(
                            PrepaidLedger.bundle_id == bid,
                            PrepaidLedger.appointment_id == appt.id,
                            PrepaidLedger.event.in_(["consume", "restore", "revert"]),
                        )
                        .scalar()
                        or 0
                    )
                    if int(net) == -1:
                        bundle = db.get(PrepaidBundle, bid)
                        if bundle:
                            bundle.remaining_credits = (
                                int(bundle.remaining_credits or 0) + 1
                            )
                            db.add(bundle)
                        db.add(
                            PrepaidLedger(
                                bundle_id=bid,
                                event="restore",
                                delta_credits=+1,
                                amount_cents=0,
                                appointment_id=appt.id,
                                note="Auto-restore on cancel (agent)",
                            )
                        )
                    spent = (
                        db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                        .filter(
                            PrepaidLedger.bundle_id == bid,
                            PrepaidLedger.appointment_id == appt.id,
                        )
                        .scalar()
                        or 0
                    )
                    if int(spent) < 0:
                        db.add(
                            PrepaidLedger(
                                bundle_id=bid,
                                event="restore",
                                delta_credits=0,
                                amount_cents=+(-int(spent)),
                                appointment_id=appt.id,
                                note="Auto-restore on cancel (agent)",
                            )
                        )
                try:
                    cash_paid = int(getattr(appt, "amount_paid_cents", 0) or 0)
                except Exception:
                    cash_paid = 0
                if cash_paid > 0 and getattr(appt, "client_id", None):
                    # Find or create a wallet for this client under this owner
                    wallet = (
                        db.query(PrepaidBundle)
                        .filter(
                            PrepaidBundle.owner_id == owner.id,
                            PrepaidBundle.client_id == str(appt.client_id),
                            PrepaidBundle.total_credits == 0,
                        )
                        .order_by(PrepaidBundle.created_at.desc())
                        .first()
                    )
                    if not wallet:
                        wallet = PrepaidBundle(
                            owner_id=str(owner.id),
                            client_id=str(appt.client_id),
                            name="Wallet",
                            total_credits=0,
                            remaining_credits=0,
                            price_cents=0,
                            currency="USD",
                        )
                        db.add(wallet)
                        db.flush()
                    db.add(
                        PrepaidLedger(
                            bundle_id=wallet.id,
                            event="refund",
                            delta_credits=0,
                            amount_cents=int(cash_paid),
                            appointment_id=appt.id,
                            note="Full refund to wallet on >=24h cancel (agent)",
                        )
                    )
                    appt.payment_status = "refunded"
                    try:
                        appt.amount_paid_cents = 0
                    except Exception:
                        pass
                try:
                    appt.bundle_id = None
                except Exception:
                    pass
            else:
                pass
            if hasattr(appt, "cancel_reason") and reason:
                appt.cancel_reason = reason

            db.add(appt)
            db.commit()
            db.refresh(appt)

            # Notify client (only when verified email exists)
            try:
                client = (
                    db.get(User, appt.client_id)
                    if getattr(appt, "client_id", None)
                    else None
                )
                if (
                    client
                    and getattr(client, "email", None)
                    and getattr(client, "emailVerified", None)
                ):
                    tz = ZoneInfo(owner.timezone)
                    start_local = appt.start_utc.astimezone(tz)
                    end_local = appt.end_utc.astimezone(tz)
                    recipient_name = (
                        getattr(client, "name", None)
                        or getattr(appt, "client_name", None)
                        or client.email
                    )
                    pkg = build_appt_email(
                        audience="client",
                        action="canceled",
                        owner=owner,
                        start_local=start_local,
                        end_local=end_local,
                        appointment_id=str(appt.id),
                        initiator_label=owner.name or "the owner",
                        status_label=appt.status,
                        recipient_name=recipient_name,
                        message=reason,
                        include_ics=False,
                        organizer_email=owner.email,
                        attendee_email=client.email,
                    )
                    send_email(
                        client.email, pkg.subject, pkg.text, pkg.html, pkg.ics_text
                    )
                    log.info(
                        "AGENT_EMAIL_SENT",
                        extra={
                            "event": "agent_email_sent",
                            "action": "canceled",
                            "appointment_id": str(appt.id),
                            "client_user_id": str(client.id),
                            "email": client.email,
                        },
                    )
                else:
                    reason = (
                        "no_client"
                        if not client
                        else (
                            "unverified_or_missing_email"
                            if not getattr(client, "emailVerified", None)
                            or not getattr(client, "email", None)
                            else "unknown"
                        )
                    )
                    log.info(
                        "AGENT_EMAIL_SKIPPED",
                        extra={
                            "event": "agent_email_skipped",
                            "action": "canceled",
                            "appointment_id": str(appt.id),
                            "client_user_id": (str(client.id) if client else None),
                            "reason": reason,
                        },
                    )
            except Exception as e:
                log.error("AGENT_EMAIL_SEND_FAILED %s", e, exc_info=True)

            return ToolCancelAppointmentOut(
                appointment_id=str(appt.id), status="canceled"
            )

    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"cancel_appointment failed: {e}")


@tool("update_appointment", args_schema=ToolUpdateAppointmentIn, return_direct=False)
def update_appointment_tool(
    appointment_id: str,
    private_note: Optional[str] = None,
    attendance: Optional[str] = None,
    late_minutes: Optional[int] = None,
    payment_status: Optional[str] = None,
    amount_paid_cents: Optional[int] = None,
    price_override_cents: Optional[int] = None,
    bundle_id: Optional[int] = None,
    apply_wallet_now: Optional[bool] = None,
    restore_wallet_now: Optional[bool] = None,
    config: Optional[RunnableConfig] = None,
) -> ToolUpdateAppointmentOut:
    """
    Update pre/post visit metadata for a single appointment.
    Only sets fields that exist on the Appointment model (safe across schemas).
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            appt: Optional[Appointment] = (
                db.query(Appointment)
                .filter(
                    Appointment.id == appointment_id, Appointment.owner_id == owner_id
                )
                .first()
            )
            if not appt:
                raise ToolException("Appointment not found")

            cols = {c.name for c in Appointment.__table__.columns}
            changed: Dict[str, object] = {}
            prev_payment_status = (
                getattr(appt, "payment_status", None) or "unpaid"
            ).lower()
            prev_bundle_id = getattr(appt, "bundle_id", None)
            if _set_if_column(appt, cols, "private_note", private_note):
                changed["private_note"] = private_note
            if attendance:
                if attendance == "late" and late_minutes is not None:
                    _set_if_column(appt, cols, "attendance", "late")
                    changed["attendance"] = "late"
                    if _set_if_column(appt, cols, "late_minutes", late_minutes):
                        changed["late_minutes"] = late_minutes
                else:
                    if _set_if_column(appt, cols, "attendance", attendance):
                        changed["attendance"] = attendance

            if _set_if_column(appt, cols, "payment_status", payment_status):
                changed["payment_status"] = payment_status
            if _set_if_column(appt, cols, "amount_paid_cents", amount_paid_cents):
                changed["amount_paid_cents"] = amount_paid_cents
            if _set_if_column(appt, cols, "price_override_cents", price_override_cents):
                changed["price_override_cents"] = price_override_cents
            new_payment_status = (
                payment_status or getattr(appt, "payment_status", None) or ""
            ).lower()
            if new_payment_status == "refunded":
                if _set_if_column(appt, cols, "price_override_cents", 0):
                    changed["price_override_cents"] = 0
                if _set_if_column(appt, cols, "amount_paid_cents", 0):
                    changed["amount_paid_cents"] = 0
            if bundle_id is not None and "bundle_id" in cols:
                target_bundle_id = bundle_id or None
                if target_bundle_id is None:
                    appt.bundle_id = None
                    changed["bundle_id"] = None
                else:
                    b = db.get(PrepaidBundle, int(target_bundle_id))
                    if (
                        (not b)
                        or (str(b.owner_id) != str(owner_id))
                        or (
                            getattr(appt, "client_id", None)
                            and str(b.client_id) != str(appt.client_id)
                        )
                    ):
                        raise ToolException("Invalid bundle")
                    appt.bundle_id = b.id
                    changed["bundle_id"] = b.id
            new_payment_status = (
                getattr(appt, "payment_status", None) or "unpaid"
            ).lower()
            became_paid = prev_payment_status != "paid" and new_payment_status == "paid"
            became_unpaid = prev_payment_status == "paid" and new_payment_status in {
                "unpaid",
                "refunded",
                "waived",
            }
            bundle_changed = prev_bundle_id != getattr(appt, "bundle_id", None)
            if (
                new_payment_status == "paid"
                and "paid_at" in cols
                and getattr(appt, "paid_at", None) is None
            ):
                try:
                    from datetime import timezone as _tz

                    appt.paid_at = datetime.now(_tz.utc)
                except Exception:
                    pass

            # ------- Ledger helpers (consume/restore wallet funds) -------
            def _bundle_balance_cents(bid: int) -> int:
                total = (
                    db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                    .filter(PrepaidLedger.bundle_id == bid)
                    .scalar()
                )
                return int(total or 0)

            def _appt_spend_cents(bid: int) -> int:
                total = (
                    db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                    .filter(
                        PrepaidLedger.bundle_id == bid,
                        PrepaidLedger.appointment_id == appt.id,
                    )
                    .scalar()
                )
                return int(total or 0)

            def _consume_amount(bid: int, cents: int):
                if cents <= 0:
                    return
                bal = _bundle_balance_cents(bid)
                use = min(bal, cents)
                if use <= 0:
                    return
                db.add(
                    PrepaidLedger(
                        bundle_id=bid,
                        event="consume",
                        delta_credits=0,
                        amount_cents=-use,
                        appointment_id=appt.id,
                        note="Auto-consume funds on payment (agent)",
                    )
                )

            def _restore_amount(bid: int, note: str):
                spent = _appt_spend_cents(bid)
                if spent < 0:
                    db.add(
                        PrepaidLedger(
                            bundle_id=bid,
                            event="restore",
                            delta_credits=0,
                            amount_cents=+(-spent),
                            appointment_id=appt.id,
                            note=note,
                        )
                    )

            # If bundle changed and this appt had wallet spend on the old bundle, restore there
            if bundle_changed and prev_bundle_id:
                _restore_amount(
                    prev_bundle_id, "Auto-restore funds on bundle change (agent)"
                )

            # If became paid and we have a bundle, consume wallet up to owed
            if became_paid and getattr(appt, "bundle_id", None):
                try:
                    price_map = _service_price_map(db, owner_user_id=owner_id)
                    expected = compute_price_cents(db, appt, price_map)
                except Exception:
                    expected = None
                cash = int(getattr(appt, "amount_paid_cents", 0) or 0)
                owed = max(int(expected or 0) - cash, 0)
                if owed > 0:
                    _consume_amount(appt.bundle_id, owed)

            # Explicit wallet actions
            if (apply_wallet_now or False) and getattr(appt, "bundle_id", None):
                try:
                    price_map = _service_price_map(db, owner_user_id=owner_id)
                    expected = compute_price_cents(db, appt, price_map)
                except Exception:
                    expected = None
                cash = int(getattr(appt, "amount_paid_cents", 0) or 0)
                already = _appt_spend_cents(int(appt.bundle_id))
                already_applied = -int(already) if int(already) < 0 else 0
                owed = max(int(expected or 0) - cash - already_applied, 0)
                if owed > 0:
                    _consume_amount(appt.bundle_id, owed)

            if (restore_wallet_now or False) and getattr(appt, "bundle_id", None):
                _restore_amount(appt.bundle_id, "Manual restore via agent edit")

            # If became unpaid/refunded/waived and we have a bundle, ensure net is 0
            if became_unpaid and getattr(appt, "bundle_id", None):
                _restore_amount(
                    appt.bundle_id,
                    f"Auto-restore funds on status={new_payment_status} (agent)",
                )

            if changed:
                db.add(appt)
            db.commit()
            db.refresh(appt)

            return ToolUpdateAppointmentOut(
                ok=True, appointment_id=str(appt.id), updated=changed
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"update_appointment failed: {e}")


@tool(
    "get_next_appointment_for_client",
    args_schema=ToolGetNextApptIn,
    return_direct=False,
)
def get_next_appointment_for_client_tool(
    client_query: str,
    include_canceled: bool = False,
    config: Optional[RunnableConfig] = None,
) -> ToolGetNextApptOut:
    """
    Look up the earliest upcoming appointment for a given client (name/email substring).
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            now_utc = _to_utc(datetime.now(ZoneInfo(owner.timezone)))

            cols = {c.name for c in Appointment.__table__.columns}
            start_col = "start_utc" if "start_utc" in cols else "start_time_utc"
            end_col = "end_utc" if "end_utc" in cols else "end_time_utc"

            q = (
                db.query(Appointment, Person.full_name.label("person_name"))
                .outerjoin(Person, Appointment.person_id == Person.id)
                .filter(Appointment.owner_id == owner.id)
            )

            cq = (client_query or "").strip().lower()
            cq = re.sub(r"[’']s\b", "", cq)
            cq = re.sub(
                r"\b(my|the|a|an|on|at|for|with|appointment|appt|lesson|account)\b",
                " ",
                cq,
            )
            cq = re.sub(r"\b(1[0-2]|0?[1-9])(:[0-5]\d)?\s*(am|pm)\b", " ", cq)
            cq = re.sub(r"\b([01]?\d|2[0-3]):[0-5]\d\b", " ", cq)
            cq = re.sub(r"\b\d{1,2}\b", " ", cq)
            cq = re.sub(r"\s+", " ", cq).strip()
            tokens = [t for t in cq.split() if len(t) >= 2]
            if tokens:
                preds = []
                for t in tokens:
                    lk = f"%{t}%"
                    preds.append(func.lower(Person.full_name).like(lk))
                    if "client_name" in cols:
                        preds.append(
                            func.lower(getattr(Appointment, "client_name")).like(lk)
                        )
                    if "client_email" in cols:
                        preds.append(
                            func.lower(getattr(Appointment, "client_email")).like(lk)
                        )
                if preds:
                    q = q.filter(or_(*preds))

            q = q.filter(getattr(Appointment, start_col) >= now_utc)
            if not include_canceled and "status" in cols:
                q = q.filter(
                    or_(Appointment.status.is_(None), Appointment.status != "canceled")
                )

            row = q.order_by(getattr(Appointment, start_col).asc()).first()
            if not row:
                return ToolGetNextApptOut(found=False, appointment=None)

            appt, person_name = row
            tz = ZoneInfo(owner.timezone)

            def _iso_local(dt):
                return dt.astimezone(tz).isoformat(timespec="minutes")

            start_dt = getattr(appt, start_col)
            end_dt = getattr(appt, end_col)

            return ToolGetNextApptOut(
                found=True,
                appointment={
                    "id": str(appt.id),
                    "client_name": person_name or getattr(appt, "client_name", None),
                    "start_local": _iso_local(start_dt),
                    "end_local": _iso_local(end_dt),
                    "status": getattr(appt, "status", None),
                },
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"get_next_appointment_for_client failed: {e}")


@tool("list_appointments", args_schema=ToolListAppointmentsIn, return_direct=False)
def list_appointments_tool(
    day: str,
    include_canceled: bool = False,
    client_query: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
) -> ToolListAppointmentsOut:
    """
    List all appointments overlapping an owner-local calendar day.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")

        tz = ZoneInfo(owner.timezone)
        d = _parse_owner_day(day, owner.timezone)

        day_start = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
        day_end = day_start + timedelta(days=1)
        u_start, u_end = _to_utc(day_start), _to_utc(day_end)

        cols = {c.name for c in Appointment.__table__.columns}
        start_col = "start_utc" if "start_utc" in cols else "start_time_utc"
        end_col = "end_utc" if "end_utc" in cols else "end_time_utc"

        base = (
            db.query(
                Appointment,
                Person.full_name.label("person_name"),
                getattr(Appointment, start_col).label("scol"),
                getattr(Appointment, end_col).label("ecol"),
            )
            .outerjoin(Person, Appointment.person_id == Person.id)
            .filter(Appointment.owner_id == owner.id)
            .filter(getattr(Appointment, start_col) < u_end)
            .filter(getattr(Appointment, end_col) > u_start)
        )

        if not include_canceled and "status" in cols:
            base = base.filter(
                or_(Appointment.status.is_(None), Appointment.status != "canceled")
            )

        if client_query:
            cq = (client_query or "").strip().lower()
            cq = re.sub(r"[’']s\b", "", cq)
            cq = re.sub(
                r"\b(my|the|a|an|on|at|for|with|appointment|appt|lesson|account)\b",
                " ",
                cq,
            )
            cq = re.sub(r"\b(1[0-2]|0?[1-9])(:[0-5]\d)?\s*(am|pm)\b", " ", cq)
            cq = re.sub(r"\b([01]?\d|2[0-3]):[0-5]\d\b", " ", cq)
            cq = re.sub(r"\b\d{1,2}\b", " ", cq)
            cq = re.sub(r"\s+", " ", cq).strip()

            tokens = [t for t in cq.split() if len(t) >= 2]
            if tokens:
                preds = []
                for t in tokens:
                    lk = f"%{t}%"
                    preds.append(func.lower(Person.full_name).like(lk))
                    if "client_name" in cols:
                        preds.append(
                            func.lower(getattr(Appointment, "client_name")).like(lk)
                        )
                    if "client_email" in cols:
                        preds.append(
                            func.lower(getattr(Appointment, "client_email")).like(lk)
                        )
                    if "owner_private_note" in cols:
                        preds.append(
                            func.lower(getattr(Appointment, "owner_private_note")).like(
                                lk
                            )
                        )
                base = base.filter(or_(*preds))

        rows = base.order_by(getattr(Appointment, start_col).asc()).all()

        def _iso_local(dt):
            return dt.astimezone(tz).isoformat(timespec="minutes")

        out = []
        for a, person_name, sdt, edt in rows:
            out.append(
                {
                    "id": str(a.id),
                    "client_name": person_name or getattr(a, "client_name", None),
                    "client_email": getattr(a, "client_email", None),
                    "person_id": getattr(a, "person_id", None),
                    "start_local": _iso_local(sdt),
                    "end_local": _iso_local(edt),
                    "status": getattr(a, "status", None),
                }
            )
        return ToolListAppointmentsOut(appointments=out)


@tool(
    "get_appointment_details",
    args_schema=ToolGetAppointmentDetailsIn,
    return_direct=False,
)
def get_appointment_details_tool(
    appointment_id: Optional[str] = None,
    start_local: Optional[str] = None,
    duration_minutes: Optional[int] = None,
    config: Optional[RunnableConfig] = None,
) -> ToolGetAppointmentDetailsOut:
    """
    Return authoritative appointment details from the DB, including WHO it's for.
    Use appointment_id when possible; otherwise resolve by (start_local + duration_minutes).
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")

        cols = {c.name for c in Appointment.__table__.columns}
        start_col = "start_utc" if "start_utc" in cols else "start_time_utc"
        end_col = "end_utc" if "end_utc" in cols else "end_time_utc"

        appt: Optional[Appointment] = None
        if appointment_id:
            appt = (
                db.query(Appointment)
                .filter(
                    Appointment.id == appointment_id, Appointment.owner_id == owner.id
                )
                .first()
            )
        else:
            if not (start_local and duration_minutes):
                raise ToolException(
                    "Provide appointment_id OR (start_local and duration_minutes)"
                )
            s_loc = _parse_owner_local_dt(start_local, owner.timezone)
            e_loc = s_loc + timedelta(minutes=int(duration_minutes))
            s_utc, e_utc = _to_utc(s_loc), _to_utc(e_loc)
            appt = (
                db.query(Appointment)
                .filter(
                    Appointment.owner_id == owner.id,
                    getattr(Appointment, start_col) == s_utc,
                    getattr(Appointment, end_col) == e_utc,
                )
                .first()
            )

        if not appt:
            raise ToolException("Appointment not found")

        p: Optional[Person] = (
            db.query(Person).filter(Person.id == appt.person_id).first()
            if appt.person_id
            else None
        )

        tz = ZoneInfo(owner.timezone)

        def _iso_local(dt):
            return dt.astimezone(tz).isoformat(timespec="minutes")

        sdt = getattr(appt, start_col)
        edt = getattr(appt, end_col)

        details = {
            "id": str(appt.id),
            "start_local": _iso_local(sdt),
            "end_local": _iso_local(edt),
            "status": getattr(appt, "status", None),
            "person_id": appt.person_id,
            "client_name": (
                p.full_name if p and p.full_name else getattr(appt, "client_name", None)
            ),
            "client_email": (
                p.email if p and p.email else getattr(appt, "client_email", None)
            ),
            "owner_private_note": (
                getattr(appt, "owner_private_note", None)
                if "owner_private_note" in cols
                else None
            ),
            "client_previsit_note": (
                getattr(appt, "client_previsit_note", None)
                if "client_previsit_note" in cols
                else None
            ),
            "client_change_note": (
                getattr(appt, "client_change_note", None)
                if "client_change_note" in cols
                else None
            ),
            "cancel_reason": (
                getattr(appt, "cancel_reason", None)
                if "cancel_reason" in cols
                else None
            ),
        }
        return ToolGetAppointmentDetailsOut(appointment=details)


@tool("reschedule_appointment", args_schema=ToolRescheduleApptIn, return_direct=False)
def reschedule_appointment_tool(
    appointment_id: str,
    start_local: str,
    duration_minutes: int,
    allow_override: bool = False,
    message: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
) -> ToolRescheduleApptOut:
    """
    Change the start time and duration of an existing appointment.
    Prevents conflicts unless allow_override=True. On conflict, raises NO_AVAILABILITY with
    reason "conflicts with another appointment" so the agent can suggest alternatives.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")

        appt: Optional[Appointment] = (
            db.query(Appointment)
            .filter(Appointment.id == appointment_id, Appointment.owner_id == owner.id)
            .first()
        )
        if not appt:
            raise ToolException("Appointment not found")

        tz = ZoneInfo(owner.timezone)
        try:
            s_loc = _parse_owner_local_dt(start_local, owner.timezone)
        except Exception:
            raise ToolException(
                f"BAD_START_LOCAL: expected 'YYYY-MM-DDTHH:MM', got {start_local!r}"
            )
        if int(duration_minutes) <= 0:
            raise ToolException("duration_minutes must be positive")
        e_loc = s_loc + timedelta(minutes=int(duration_minutes))
        s_utc, e_utc = _to_utc(s_loc), _to_utc(e_loc)

        # Check conflicts (appointments + time off) unless overriding
        if not allow_override:
            conflicts = (
                db.query(Appointment)
                .filter(
                    Appointment.owner_id == owner.id,
                    Appointment.id != appt.id,
                    Appointment.status != "canceled",
                    Appointment.end_utc > s_utc,
                    Appointment.start_utc < e_utc,
                )
                .all()
            )
            if conflicts:
                human = f"Reschedule conflicts with another appointment ({s_loc.isoformat(timespec='minutes')}–{e_loc.isoformat(timespec='minutes')})."
                payload = {
                    "human": human,
                    "reason": "conflicts with another appointment",
                    "start_local": s_loc.isoformat(timespec="minutes"),
                    "end_local": e_loc.isoformat(timespec="minutes"),
                    "duration_min": int(duration_minutes),
                }
                raise ToolException(
                    "NO_AVAILABILITY:" + __import__("json").dumps(payload)
                )

        # Guard against duplicate exact-start per owner
        dup_same_start = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.id != appt.id,
                Appointment.start_utc == s_utc,
            )
            .first()
        )
        if dup_same_start and not allow_override:
            payload = {
                "human": "Another appointment already exists at that exact start time.",
                "reason": "conflicts with another appointment",
                "start_local": s_loc.isoformat(timespec="minutes"),
                "end_local": e_loc.isoformat(timespec="minutes"),
                "duration_min": int(duration_minutes),
            }
            raise ToolException("NO_AVAILABILITY:" + __import__("json").dumps(payload))

        # Capture old local times for email, then apply and persist
        old_start_local = appt.start_utc.astimezone(tz)
        old_end_local = appt.end_utc.astimezone(tz)

        appt.start_utc, appt.end_utc = s_utc, e_utc
        db.add(appt)
        db.commit()
        db.refresh(appt)

        # Notify client (only when verified email exists)
        try:
            client = (
                db.get(User, appt.client_id)
                if getattr(appt, "client_id", None)
                else None
            )
            if (
                client
                and getattr(client, "email", None)
                and getattr(client, "emailVerified", None)
            ):
                new_start_local = appt.start_utc.astimezone(tz)
                new_end_local = appt.end_utc.astimezone(tz)
                recipient_name = (
                    getattr(client, "name", None)
                    or getattr(appt, "client_name", None)
                    or client.email
                )
                pkg = build_appt_email(
                    audience="client",
                    action="updated",
                    owner=owner,
                    start_local=new_start_local,
                    end_local=new_end_local,
                    appointment_id=str(appt.id),
                    initiator_label=owner.name or "the owner",
                    status_label=appt.status,
                    recipient_name=recipient_name,
                    message=message,
                    old_start_local=old_start_local,
                    old_end_local=old_end_local,
                    include_ics=True,
                    organizer_email=owner.email,
                    attendee_email=client.email,
                )
                send_email(client.email, pkg.subject, pkg.text, pkg.html, pkg.ics_text)
                log.info(
                    "AGENT_EMAIL_SENT",
                    extra={
                        "event": "agent_email_sent",
                        "action": "updated",
                        "appointment_id": str(appt.id),
                        "client_user_id": str(client.id),
                        "email": client.email,
                    },
                )
            else:
                reason = (
                    "no_client"
                    if not client
                    else (
                        "unverified_or_missing_email"
                        if not getattr(client, "emailVerified", None)
                        or not getattr(client, "email", None)
                        else "unknown"
                    )
                )
                log.info(
                    "AGENT_EMAIL_SKIPPED",
                    extra={
                        "event": "agent_email_skipped",
                        "action": "updated",
                        "appointment_id": str(appt.id),
                        "client_user_id": (str(client.id) if client else None),
                        "reason": reason,
                    },
                )
        except Exception as e:
            log.error("AGENT_EMAIL_SEND_FAILED %s", e, exc_info=True)

        return ToolRescheduleApptOut(
            appointment_id=str(appt.id),
            start_utc=appt.start_utc,
            end_utc=appt.end_utc,
            status=appt.status,
        )


@tool("list_post_appointment_actions", return_direct=False)
def list_post_appointment_actions_tool(
    config: Optional[RunnableConfig] = None,
) -> ToolListPostApptActionsOut:
    """
    Return a checklist of recently completed appointments that likely need updates
    (attendance, late minutes, payment status). Useful for post-visit follow-up.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")
        tz = ZoneInfo(owner.timezone)
        now = datetime.now(tz)
        # consider appointments that ended in the past 14 days
        past_cutoff = _to_utc((now - timedelta(days=14)).astimezone(tz))

        rows = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.end_utc <= _to_utc(now),
                Appointment.end_utc >= past_cutoff,
                Appointment.status == "completed",
            )
            .order_by(Appointment.end_utc.desc())
            .all()
        )
        items = []
        for a in rows:
            s_loc = a.start_utc.astimezone(tz).isoformat(timespec="minutes")
            e_loc = a.end_utc.astimezone(tz).isoformat(timespec="minutes")
            att = getattr(a, "attendance_status", "unknown") or "unknown"
            pay = getattr(a, "payment_status", "unpaid") or "unpaid"
            items.append(
                {
                    "id": str(a.id),
                    "client_name": getattr(a, "client_name", None),
                    "start_local": s_loc,
                    "end_local": e_loc,
                    "status": a.status,
                    "needs_attendance": (att == "unknown"),
                    "needs_payment": (pay != "paid"),
                    "late_minutes": getattr(a, "late_minutes", 0),
                    "payment_status": pay,
                }
            )
        return ToolListPostApptActionsOut(items=items)


@tool("help_appointment_updates", return_direct=True)
def help_appointment_updates_tool() -> str:
    """Show how to update appointment details (identity, reschedule, notes, attendance, payment, pricing)."""
    lines = [
        "You can update any appointment using these tools:",
        "",
        "- Reschedule: reschedule_appointment(appointment_id, start_local='YYYY-MM-DDTHH:MM', duration_minutes)",
        "  • Prevents conflicts by default; returns alternatives on conflict.",
        "- Identity/person: update_appointment_details(appointment_id, person_id? / client_email? / client_name?, note?, price_override_cents?)",
        "- Post-visit details: update_appointment(appointment_id, owner_private_note?, attendance?, late_minutes?, payment_status?, bundle_id?, price_override_cents?, amount_paid_cents?)",
        "- Checklist: list_post_appointment_actions() to see completed appts needing attendance/payment updates.",
        "",
        "Examples:",
        "• Mark paid: update_appointment(appointment_id='...', payment_status='paid', amount_paid_cents=4000)",
        "• Override price: update_appointment(appointment_id='...', price_override_cents=4500)",
        "• Set attendance: update_appointment(appointment_id='...', attendance='late', late_minutes=10)",
        "• Reschedule: reschedule_appointment(appointment_id='...', start_local='2025-10-05T15:00', duration_minutes=30)",
    ]
    return "\n".join(lines)
