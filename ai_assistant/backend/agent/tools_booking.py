from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import json
import os
import uuid

from langchain_core.tools import tool, ToolException
from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from psycopg.errors import ExclusionViolation

from app.db import SessionLocal
from app.models import User, Appointment, Person, ClientAccount, TimeOff, RoleEnum
from agent.constants import ACTIVE_APPT_STATUSES, CANCELLED_APPT_STATUSES
from agent.schemas import (
    ToolBookAppointmentIn,
    ToolBookAppointmentOut,
    ToolBookRecurringAppointmentsIn,
    ToolBookRecurringAppointmentsOut,
)
from agent.tool_ops import resolve_person as _resolve_person_core
from agent.tool_ctx import owner_id_var
from services.services_scheduling import is_owner_time_bookable
from services.payments import (
    get_default_price_cents,
)
from services.wallets import auto_apply_wallet_funds
from app.models import PrepaidBundle
from routers._helpers import ics, send_email, build_appt_email
from services.emailer import render_basic_html, BRAND_GREEN


def _owner_id_from_config(config: RunnableConfig) -> str:
    """Extract the owner/user id from the runnable config or contextvars.

    Args:
        config: Runnable configuration possibly containing 'configurable' with
            'user_id' or 'owner_id'.

    Returns:
        Owner id as a string.

    Raises:
        ToolException: If no owner id can be found in config or context.
    """
    cfg = config or {}
    cfg = (
        cfg.get("configurable", {})
        if isinstance(cfg, dict)
        else getattr(cfg, "configurable", {}) or {}
    )
    owner_id = (
        cfg.get("user_id")
        or cfg.get("owner_id")
        or owner_id_var.get()
        or os.getenv("OWNER_ID_DEFAULT")
        or os.getenv("OWNER_ID")
    )
    if not owner_id:
        raise ToolException("Missing owner id in tool config")
    return str(owner_id)


def _lower(s: Optional[str]) -> Optional[str]:
    """Return a lowercase version of a string, preserving None/other types."""
    return s.lower() if isinstance(s, str) else None


def _iso(dt: datetime, tz: ZoneInfo) -> str:
    """Format a datetime as owner-local ISO 'YYYY-MM-DDTHH:MM'."""
    return dt.astimezone(tz).strftime("%Y-%m-%dT%H:%M")


def _collect_conflicts(
    db: Session, owner_id: str, start_utc: datetime, end_utc: datetime
) -> tuple[list[Appointment], list[TimeOff]]:
    """Return overlapping appointments and time off for the owner in [start,end).

    Args:
        db: Active database session.
        owner_id: Owner user id.
        start_utc: Start of candidate window (UTC).
        end_utc: End of candidate window (UTC).

    Returns:
        Tuple of (appointments, time_off) lists.
    """
    conflict_appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner_id,
            Appointment.end_utc > start_utc,
            Appointment.start_utc < end_utc,
            Appointment.status != "canceled",
        )
        .order_by(Appointment.start_utc.asc())
        .all()
    )

    conflict_offs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner_id,
            TimeOff.end_utc > start_utc,
            TimeOff.start_utc < end_utc,
        )
        .order_by(TimeOff.start_utc.asc())
        .all()
    )

    return conflict_appts, conflict_offs


def _format_conflicts(
    owner: User, conflict_appts: List[Appointment], conflict_offs: List[TimeOff]
) -> List[str]:
    """Return human-friendly conflict lines in owner local time."""
    out: List[str] = []
    owner_tz = ZoneInfo(owner.timezone or "UTC")
    if conflict_offs:
        preview = "; ".join(
            _iso(x.start_utc, owner_tz) + " → " + _iso(x.end_utc, owner_tz)
            for x in conflict_offs[:5]
        )
        if preview:
            suffix = (
                f" (+{len(conflict_offs) - 5} more)" if len(conflict_offs) > 5 else ""
            )
            out.append("Time Off: " + preview + suffix)
    if conflict_appts:
        preview = "; ".join(
            _iso(x.start_utc, owner_tz) + " → " + _iso(x.end_utc, owner_tz)
            for x in conflict_appts[:5]
        )
        if preview:
            suffix = (
                f" (+{len(conflict_appts) - 5} more)" if len(conflict_appts) > 5 else ""
            )
            out.append("Appointments: " + preview + suffix)
    return out


@tool("book_appointment", args_schema=ToolBookAppointmentIn, return_direct=False)
def book_appointment_tool(
    start_local: str,
    duration_min: int | None = None,
    person_id: int | None = None,
    client_name: str | None = None,
    client_email: str | None = None,
    client_query: str | None = None,
    notes: str | None = None,
    config: RunnableConfig = None,
) -> ToolBookAppointmentOut:
    """Book a single appointment at a specific owner-local time.

    Resolves identity using the provided person/email/name and denormalizes
    'client_name'/'client_email' onto the appointment. Validates
    availability and translates overlap/conflict conditions into structured
    'ToolException' payloads for the graph.

    Args:
        start_local: Owner‑local start time ('YYYY‑MM‑DDTHH:MM').
        duration_min: Duration in minutes.
        person_id: Existing 'Person.id' if known.
        client_name: Exact client name to resolve when no 'person_id'.
        client_email: Client email to resolve/create a person when needed.
        client_query: Free‑form selector hint used by the router.
        notes: Optional owner‑private note to include in notifications.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolBookAppointmentOut' including ids and UTC timestamps.

    Raises:
        ToolException: On missing duration, bad start format, owner not found,
        no availability, or overlap with existing appointments.
    """
    owner_id = _owner_id_from_config(config)
    if not duration_min:
        hinted = (
            (config or {}).get("configurable", {}).get("hints", {}).get("duration_min")
        )
        if hinted:
            duration_min = hinted
    if not duration_min:
        raise ToolException(
            "DURATION_REQUIRED: Please provide duration_min in minutes."
        )

    with SessionLocal() as db:
        owner: User | None = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")

        tz = ZoneInfo(owner.timezone)
        try:
            start_dt_local = datetime.fromisoformat(start_local)
        except Exception:
            raise ToolException(
                f"BAD_START_LOCAL: expected 'YYYY-MM-DDTHH:MM', got {start_local!r}"
            )
        start_utc = start_dt_local.replace(tzinfo=tz).astimezone(ZoneInfo("UTC"))
        end_utc = start_utc + timedelta(minutes=int(duration_min))

        # ---------- Resolve person/email/name ----------
        selector = client_query or client_name or client_email
        person: Person | None = None
        original_name_input = (client_name or "").strip() or None
        original_email_input = (client_email or "").strip() or None
        resolved_name = original_name_input
        resolved_email = original_email_input

        # 1) Direct person_id (authoritative)
        if person_id:
            person = (
                db.query(Person)
                .join(ClientAccount, ClientAccount.id == Person.account_id)
                .filter(
                    ClientAccount.owner_user_id == owner_id,
                    Person.id == person_id,
                )
                .first()
            )
            # person_id is authoritative: prefer the Person's name/email over any account-level name
            if person:
                if getattr(person, "full_name", None):
                    resolved_name = person.full_name
                if getattr(person, "email", None):
                    resolved_email = person.email

        # 2) Single-person account heuristic BEFORE potentially creating shadow person
        if person is None and selector:
            acct = (
                db.query(ClientAccount)
                .filter(
                    ClientAccount.owner_user_id == owner_id,
                    func.lower(ClientAccount.name) == _lower(selector),
                )
                .first()
            )
            if acct and len(acct.people) == 1:
                person = acct.people[0]
                if not resolved_name:
                    resolved_name = person.full_name
                if not resolved_email:
                    resolved_email = person.email

        # 3) Existing person by email (avoid creating shadow if we already know actual person)
        if person is None and resolved_email:
            person = (
                db.query(Person)
                .join(ClientAccount, ClientAccount.id == Person.account_id)
                .filter(
                    ClientAccount.owner_user_id == owner_id,
                    func.lower(Person.email) == _lower(resolved_email),
                )
                .first()
            )
            if person and not resolved_name:
                resolved_name = person.full_name

        # 4) Core resolver (may create shadow person if necessary)
        if person is None:
            try:
                fallback_name = resolved_name or selector
                rp, rname, remail = _resolve_person_core(
                    db=db,
                    owner_id=owner_id,
                    client_email=resolved_email,
                    client_name=fallback_name,
                )
                person = rp or None
                if not resolved_name and rname:
                    resolved_name = rname
                if not resolved_email and remail:
                    resolved_email = remail
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("Person resolution failed: %s", e)
                db.rollback()

        # 5) Final fill from person (prefer the specific person's identity over account-level)
        if person:
            if getattr(person, "full_name", None):
                resolved_name = person.full_name
            if getattr(person, "email", None):
                resolved_email = person.email

        # Determine client_user_id via the person's account (for analytics joins)
        client_user_id = None
        if person and getattr(person, "account_id", None):
            acct = (
                db.query(ClientAccount)
                .filter(ClientAccount.id == person.account_id)
                .first()
            )
            if acct and getattr(acct, "client_user_id", None):
                client_user_id = acct.client_user_id

        # ---------- Identical-slot guard (treat as conflict unless clearly same identity) ----------
        duplicate = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner_id,
                Appointment.status != "canceled",
                Appointment.start_utc == start_utc,
                Appointment.end_utc == end_utc,
            )
            .first()
        )
        if duplicate:
            same_person = bool(
                person
                and duplicate.person_id
                and int(duplicate.person_id) == int(person.id)
            )
            same_email = bool(
                resolved_email
                and duplicate.client_email
                and resolved_email.strip().lower()
                == duplicate.client_email.strip().lower()
            )
            if same_person or same_email:
                return ToolBookAppointmentOut(
                    appointment_id=str(duplicate.id),
                    start_utc=duplicate.start_utc,
                    end_utc=duplicate.end_utc,
                    status=duplicate.status,
                    person_id=str(duplicate.person_id) if duplicate.person_id else None,
                    client_name=duplicate.client_name,
                    client_email=duplicate.client_email,
                )
            else:
                start_local_iso = _iso(start_utc, tz)
                end_local_iso = _iso(end_utc, tz)
                payload = {
                    "human": f"That time is already booked ({start_local_iso}–{end_local_iso}).",
                    "start_local": start_local_iso,
                    "end_local": end_local_iso,
                    "duration_min": int(duration_min),
                    "conflict_appt_id": str(duplicate.id),
                    "conflict_status": duplicate.status,
                }
                raise ToolException("APPT_OVERLAP:" + json.dumps(payload))

        ok, reason = is_owner_time_bookable(db, owner, start_utc, end_utc)
        if not ok:
            start_local_iso = _iso(start_utc, tz)
            end_local_iso = _iso(end_utc, tz)
            reason_map = {
                "time off": "That time is blocked by time off.",
                "conflicts with another appointment": "There is already an appointment or buffer around that time.",
                "no opening covers the requested time": "That slot isn't within your published availability.",
                "crosses day boundary": "The requested slot crosses midnight; adjust the start time.",
            }
            human_msg = reason_map.get(reason, f"No availability: {reason}")
            payload = {
                "human": human_msg,
                "reason": reason,
                "start_local": start_local_iso,
                "end_local": end_local_iso,
                "duration_min": int(duration_min),
            }
            raise ToolException("NO_AVAILABILITY:" + json.dumps(payload))

        # ---------- Pre-commit overlap guard ----------
        overlap_q = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner_id,
                Appointment.status.in_(ACTIVE_APPT_STATUSES),
                Appointment.start_utc < end_utc,
                Appointment.end_utc > start_utc,
            )
            .order_by(Appointment.start_utc.asc())
        )
        overlap = overlap_q.first()
        if overlap:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Overlap conflict: appt_id=%s status=%s existing=[%s,%s] requested=[%s,%s] owner=%s",
                overlap.id,
                overlap.status,
                overlap.start_utc,
                overlap.end_utc,
                start_utc,
                end_utc,
                owner_id,
            )
            others = overlap_q.limit(5).all()
            if len(others) > 1:
                _logging.getLogger(__name__).warning(
                    "Additional overlaps(%d): %s",
                    len(others) - 1,
                    [str(o.id) for o in others[1:]],
                )
            start_local_iso = _iso(start_utc, tz)
            end_local_iso = _iso(end_utc, tz)
            payload = {
                "human": f"That time is no longer available ({start_local_iso}–{end_local_iso}).",
                "start_local": start_local_iso,
                "end_local": end_local_iso,
                "duration_min": int(duration_min),
                "conflict_appt_id": str(overlap.id),
                "conflict_status": overlap.status,
            }
            raise ToolException("APPT_OVERLAP:" + json.dumps(payload))

        # ---------- Require identity one last time ----------
        if person:
            if not resolved_name and person.full_name:
                resolved_name = person.full_name
            if not resolved_email and person.email:
                resolved_email = person.email
        if not resolved_name and not resolved_email and client_query:
            resolved_name = client_query.strip() or None
        if not resolved_name and not resolved_email:
            raise ToolException(
                f"Need client_email or client_name to create appointment. "
                f"Received: client_name='{client_name}', client_email='{client_email}', "
                f"client_query='{client_query}', selector='{selector}'"
            )

        # ---------- Revive a canceled appointment if identical slot exists ----------
        revive = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner_id,
                Appointment.status.in_(CANCELLED_APPT_STATUSES),
                Appointment.start_utc == start_utc,
                Appointment.end_utc == end_utc,
            )
            .order_by(
                Appointment.updated_at.desc()
                if hasattr(Appointment, "updated_at")
                else Appointment.start_utc.desc()
            )
            .first()
        )
        if revive:
            import logging as _logging

            _logging.getLogger(__name__).info(
                "Reviving canceled appointment %s for slot [%s,%s] owner=%s",
                revive.id,
                start_utc,
                end_utc,
                owner_id,
            )
            revive.status = "booked"
            revive.person_id = person.id if person else None
            revive.client_name = resolved_name
            revive.client_email = resolved_email
            if client_user_id and not getattr(revive, "client_id", None):
                revive.client_id = client_user_id
            if hasattr(revive, "cancel_reason"):
                revive.cancel_reason = None
            if hasattr(revive, "client_change_note"):
                revive.client_change_note = None
            if hasattr(revive, "attendance_status"):
                revive.attendance_status = "attended"
            if hasattr(revive, "payment_status"):
                revive.payment_status = "unpaid"
            if hasattr(revive, "owner_private_note"):
                revive.owner_private_note = notes if notes else None
            elif hasattr(revive, "client_previsit_note"):
                revive.client_previsit_note = notes if notes else None
            db.add(revive)
            db.commit()
            db.refresh(revive)
            return ToolBookAppointmentOut(
                appointment_id=str(revive.id),
                start_utc=revive.start_utc,
                end_utc=revive.end_utc,
                status=revive.status,
                person_id=str(revive.person_id) if revive.person_id else None,
                client_name=revive.client_name,
                client_email=revive.client_email,
            )

        # ---------- Require a contact email before creating ----------
        if not resolved_email and person and getattr(person, "account_id", None):
            acct = (
                db.query(ClientAccount)
                .filter(ClientAccount.id == person.account_id)
                .first()
            )
            if acct:
                from agent.tool_ops import _primary_account_email as _acct_email

                ae = _acct_email(db, acct.id)
                if ae:
                    resolved_email = ae

        if not resolved_email:
            raise ToolException(
                "NEED_CLIENT_EMAIL: This client has no email on file. Please provide a client_email (or pick a person with an email) to proceed."
            )

        # ---------- Create ----------
        import logging

        logging.getLogger(__name__).info(
            "Creating appointment with: person_id=%s (input pid=%s), client_name(orig=%s -> final=%s), client_email(orig=%s -> final=%s), selector=%s",
            person.id if person else None,
            person_id,
            original_name_input,
            resolved_name,
            original_email_input,
            resolved_email,
            selector,
        )

        fields: Dict[str, Any] = dict(
            owner_id=owner_id,
            start_utc=start_utc,
            end_utc=end_utc,
            status="booked",
            person_id=(person.id if person else None),
            client_name=resolved_name,
            client_email=resolved_email,
        )
        if client_user_id:
            fields["client_id"] = client_user_id
        if notes:
            if hasattr(Appointment, "owner_private_note"):
                fields["owner_private_note"] = notes
            elif hasattr(Appointment, "client_previsit_note"):
                fields["client_previsit_note"] = notes

        appt = Appointment(**fields)
        db.add(appt)
        try:
            db.commit()
        except IntegrityError as e:
            db.rollback()
            if isinstance(
                e.orig, ExclusionViolation
            ) or "appointments_no_overlap" in str(e):
                start_local_iso = _iso(start_utc, tz)
                end_local_iso = _iso(end_utc, tz)
                payload = {
                    "human": f"That time is no longer available ({start_local_iso}–{end_local_iso}).",
                    "start_local": start_local_iso,
                    "end_local": end_local_iso,
                    "duration_min": int(duration_min),
                }
                raise ToolException("APPT_OVERLAP:" + json.dumps(payload))
            raise ToolException(f"DB_COMMIT_FAILED: {e}")

        db.refresh(appt)
        # ---------- Auto-apply wallet (store credit) if available ----------
        try:
            if client_user_id:
                wallet_ids = (
                    db.query(PrepaidBundle.id)
                    .filter(
                        PrepaidBundle.owner_id == owner_id,
                        PrepaidBundle.client_id == str(client_user_id),
                        PrepaidBundle.total_credits == 0,
                    )
                    .order_by(PrepaidBundle.created_at.desc())
                    .all()
                )
                for (wid,) in wallet_ids:
                    try:
                        auto_apply_wallet_funds(
                            db,
                            owner_id=str(owner_id),
                            bundle_id=int(wid),
                            note_prefix="Auto-apply wallet funds after booking",
                        )
                        db.refresh(appt)
                    except Exception:
                        db.rollback()
                        try:
                            db.refresh(appt)
                        except Exception:
                            pass
        except Exception:
            db.rollback()
            try:
                db.refresh(appt)
            except Exception:
                pass

        # --------- send confirmation email to client ---------
        try:
            to_email: str | None = None
            recipient_name: str | None = None
            client_user = (
                db.get(User, appt.client_id)
                if getattr(appt, "client_id", None)
                else None
            )
            if client_user and getattr(client_user, "email", None):
                to_email = client_user.email
                recipient_name = (
                    getattr(client_user, "name", None)
                    or getattr(appt, "client_name", None)
                    or to_email
                )
            elif getattr(appt, "client_email", None):
                to_email = appt.client_email
                recipient_name = getattr(appt, "client_name", None) or to_email

            if to_email:
                start_local_dt = appt.start_utc.astimezone(tz)
                end_local_dt = appt.end_utc.astimezone(tz)
                pkg = build_appt_email(
                    audience="client",
                    action="created",
                    owner=owner,
                    start_local=start_local_dt,
                    end_local=end_local_dt,
                    appointment_id=str(appt.id),
                    initiator_label=owner.name or "the owner",
                    status_label=getattr(appt, "status", "booked"),
                    recipient_name=recipient_name,
                    message=notes,
                    include_ics=True,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                send_email(to_email, pkg.subject, pkg.text, pkg.html, pkg.ics_text)
        except Exception:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Failed to send single booking email", exc_info=True
            )

        return ToolBookAppointmentOut(
            appointment_id=str(appt.id),
            start_utc=appt.start_utc,
            end_utc=appt.end_utc,
            status=appt.status,
            person_id=str(appt.person_id) if appt.person_id else None,
            client_name=appt.client_name,
            client_email=appt.client_email,
        )


@tool(
    "book_recurring_appointments",
    args_schema=ToolBookRecurringAppointmentsIn,
    return_direct=False,
)
def book_recurring_appointments_tool(
    start_local: str,
    duration_min: int,
    repeat_every_weeks: int,
    occurrences: int | None = None,
    until_date: str | None = None,
    client_email: str | None = None,
    client_name: str | None = None,
    confirm_if_conflicts: bool = False,
    message: str | None = None,
    config: RunnableConfig = None,
) -> ToolBookRecurringAppointmentsOut:
    """Book a weekly/recurring series of appointments.

    If conflicts are detected and 'confirm_if_conflicts' is False, raises a
    'ToolException' with a 'CONFIRM_REQUIRED:{...}' payload summarizing
    conflicts and the pending arguments to proceed.

    Args:
        start_local: Anchor owner‑local start datetime for the first occurrence.
        duration_min: Duration in minutes for each occurrence.
        repeat_every_weeks: Interval between occurrences in weeks.
        occurrences: Optional fixed number of occurrences to create.
        until_date: Optional inclusive end date ('YYYY‑MM‑DD').
        client_email: Client email; required to tie the series to an account.
        client_name: Optional display name override.
        confirm_if_conflicts: Proceed despite conflicts if True.
        message: Optional message included in notifications.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolBookRecurringAppointmentsOut' listing created appointments.

    Raises:
        ToolException: For invalid inputs, missing client/account, conflicts
        requiring confirmation, or duplicate occurrence times.
    """
    owner_id = _owner_id_from_config(config)

    if not client_email:
        raise ToolException(
            "CLIENT_EMAIL_REQUIRED: Provide the client's email to book recurring appointments."
        )

    try:
        start_dt = datetime.fromisoformat(start_local)
    except Exception:
        raise ToolException(
            f"BAD_START_LOCAL: expected 'YYYY-MM-DDTHH:MM', got {start_local!r}"
        )

    limit_date: date | None = None
    if until_date:
        try:
            limit_date = date.fromisoformat(until_date)
        except ValueError as e:
            raise ToolException(f"BAD_UNTIL_DATE: {e}")

    step = timedelta(weeks=int(repeat_every_weeks))
    duration_delta = timedelta(minutes=int(duration_min))

    with SessionLocal() as db:
        owner: User | None = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")
        owner_tz = ZoneInfo(owner.timezone or "UTC")

        start_local_dt = (
            start_dt.replace(tzinfo=owner_tz)
            if start_dt.tzinfo is None
            else start_dt.astimezone(owner_tz)
        )

        client = (
            db.query(User)
            .filter(User.email == client_email, User.role == RoleEnum.CLIENT)
            .first()
        )
        if not client:
            raise ToolException(
                f"Client with email '{client_email}' not found. Please add the client first."
            )

        client_account = (
            db.query(ClientAccount)
            .filter(
                ClientAccount.owner_user_id == owner.id,
                ClientAccount.client_user_id == client.id,
                ClientAccount.deleted_at.is_(None),
            )
            .first()
        )
        if not client_account:
            raise ToolException(
                f"Client '{client_email}' is not associated with this owner. Please add them to your clients list."
            )

        occurrences_list: List[tuple[datetime, datetime, datetime, datetime]] = []
        current = start_local_dt
        count = 0
        while True:
            if limit_date is not None and current.date() > limit_date:
                break
            end_local_dt = current + duration_delta
            start_utc = current.astimezone(ZoneInfo("UTC"))
            end_utc = end_local_dt.astimezone(ZoneInfo("UTC"))
            occurrences_list.append((current, end_local_dt, start_utc, end_utc))
            count += 1
            if occurrences is not None and count >= occurrences:
                break
            if limit_date is None and occurrences is None and count >= 104:
                break
            if count >= 104:
                break
            current = current + step

        if not occurrences_list:
            raise ToolException(
                "No occurrences were generated for the requested recurrence."
            )

        conflicts_summary: List[Dict[str, Any]] = []
        if not confirm_if_conflicts:
            for start_loc, end_loc, start_utc, end_utc in occurrences_list:
                conflict_appts, conflict_offs = _collect_conflicts(
                    db, owner.id, start_utc, end_utc
                )
                if conflict_appts or conflict_offs:
                    conflicts_summary.append(
                        {
                            "start_local": start_loc.isoformat(timespec="minutes"),
                            "conflicts": _format_conflicts(
                                owner, conflict_appts, conflict_offs
                            ),
                        }
                    )

        if conflicts_summary and not confirm_if_conflicts:
            payload = {
                "human": "Requested series conflicts with existing items. Reply 'confirm' to proceed anyway, or adjust the times.",
                "pending": {
                    "tool": "book_recurring_appointments",
                    "args": {
                        "start_local": start_local,
                        "duration_min": int(duration_min),
                        "repeat_every_weeks": int(repeat_every_weeks),
                        "occurrences": occurrences,
                        "until_date": until_date,
                        "client_email": client_email,
                        "client_name": client_name,
                        "confirm_if_conflicts": True,
                        "message": message,
                    },
                },
                "conflicts": conflicts_summary,
            }
            raise ToolException("CONFIRM_REQUIRED:" + json.dumps(payload))

        created: List[tuple[Appointment, datetime, datetime]] = []
        seen_starts: set[datetime] = set()
        display_name = client_name or client.name or client.email

        for start_loc, end_loc, start_utc, end_utc in occurrences_list:
            if start_utc in seen_starts:
                raise ToolException(
                    "Duplicate occurrence start times detected in request."
                )
            seen_starts.add(start_utc)

            dup = (
                db.query(Appointment)
                .filter(
                    Appointment.owner_id == owner.id, Appointment.start_utc == start_utc
                )
                .first()
            )
            if dup:
                raise ToolException(
                    "Another appointment already exists at one of the requested start times."
                )

            appt = Appointment(
                id=uuid.uuid4(),
                owner_id=owner.id,
                client_id=client.id,
                client_name=display_name,
                client_email=client.email,
                start_utc=start_utc,
                end_utc=end_utc,
                status="booked",
            )
            db.add(appt)

            duration_minutes = int((end_utc - start_utc).total_seconds() // 60)
            appt.price_override_cents = get_default_price_cents(
                db, owner_user_id=owner.id, duration_minutes=duration_minutes
            )
            appt.payment_status = "unpaid"
            appt.amount_paid_cents = appt.amount_paid_cents or 0

            created.append((appt, start_loc, end_loc))

        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise ToolException(
                "Another appointment already exists at one of the requested start times."
            )

        response_rows: List[Dict[str, Any]] = []
        owner_tz_str = owner.timezone or "UTC"

        for appt, start_loc, end_loc in created:
            try:
                db.refresh(appt)
            except Exception:
                pass

            uid = str(uuid.uuid4())
            ics_text = ics(
                uid=uid,
                start=start_loc,
                end=end_loc,
                summary=f"Appointment with {owner.name}",
                organizer_email=owner.email,
                attendee_email=client.email,
            )

            lines = [
                f"Hi {client.name or client.email},",
                "Your appointment is confirmed.",
                f"When: {start_loc.strftime('%Y-%m-%d %H:%M')}–{end_loc.strftime('%H:%M')} ({owner_tz_str})",
            ]
            if message:
                lines += ["", "Message from owner:", message]

            html = render_basic_html("Appointment Confirmed", lines, BRAND_GREEN)

            try:
                send_email(
                    client.email,
                    f"New appointment — {start_loc.strftime('%b %d, %Y %H:%M')} ({owner_tz_str})",
                    "\n".join(lines),
                    html,
                    ics_text,
                )
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to send recurring appointment email", exc_info=True
                )

            response_rows.append(
                {
                    "appointment_id": str(appt.id),
                    "status": appt.status,
                    "start_local": start_loc.isoformat(timespec="minutes"),
                }
            )

        return ToolBookRecurringAppointmentsOut(
            count=len(response_rows), appointments=response_rows
        )


@tool("debug_slot_conflicts", return_direct=False)
def debug_slot_conflicts_tool(
    start_local: str,
    duration_min: int,
    config: RunnableConfig = None,
) -> dict:
    """Explain why a requested time window is considered busy.

    Args:
        start_local: Owner‑local start time ('YYYY‑MM‑DDTHH:MM').
        duration_min: Duration in minutes.
        config: Runnable configuration providing the owner id.

    Returns:
        Dict with 'requested', 'conflicts', and 'notes' fields to aid
        debugging.

    Raises:
        None directly; returns an 'error' key for invalid inputs/owner.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        owner: User | None = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            return {"error": "Owner not found"}
        tz = ZoneInfo(owner.timezone)
        try:
            start_dt_local = datetime.fromisoformat(start_local)
        except Exception:
            return {
                "error": f"BAD_START_LOCAL: expected 'YYYY-MM-DDTHH:MM', got {start_local!r}"
            }
        start_utc = start_dt_local.replace(tzinfo=tz).astimezone(ZoneInfo("UTC"))
        end_utc = start_utc + timedelta(minutes=int(duration_min))

        overlaps = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner_id,
                Appointment.status.in_(ACTIVE_APPT_STATUSES),
                Appointment.start_utc < end_utc,
                Appointment.end_utc > start_utc,
            )
            .order_by(Appointment.start_utc.asc())
            .all()
        )
        conflicts = []
        for a in overlaps:
            conflicts.append(
                {
                    "appointment_id": str(a.id),
                    "status": a.status,
                    "start_local": _iso(a.start_utc, tz),
                    "end_local": _iso(a.end_utc, tz),
                    "client_name": a.client_name,
                    "client_email": a.client_email,
                }
            )
        notes = []
        if not conflicts:
            notes.append(
                "No active appointment overlaps; if booking still fails, check availability rules or that slot length matches expected service duration."
            )
        else:
            notes.append(f"Found {len(conflicts)} overlapping busy appointment(s).")
        return {
            "requested": {
                "start_local": start_local,
                "end_local": (
                    start_dt_local + timedelta(minutes=int(duration_min))
                ).isoformat(timespec="minutes"),
                "duration_min": int(duration_min),
            },
            "conflicts": conflicts,
            "notes": notes,
        }
