from __future__ import annotations
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    BackgroundTasks,
    Response,
    Query,
    Request,
)
import logging
from pydantic import BaseModel, EmailStr, model_validator, ConfigDict, Field
from datetime import datetime, timedelta, date, timezone as _tz
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from sqlalchemy.orm import noload
from typing import Any, Optional, Literal
from sqlalchemy.exc import IntegrityError
from app.schemas import (
    SnapshotOut,
    AppointmentListOut,
    AppointmentRow,
    OwnerAppointmentRowOut,
)
from sqlalchemy import and_, or_
from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import (
    User,
    AvailabilityRule,
    TimeOff,
    Appointment,
    RoleEnum,
    SpecialOpening,
    ClientAccount,
    PrepaidBundle,
    PrepaidLedger,
)
from app.schemas import (
    AppointmentUpdateOwner,
)
from ._helpers import (
    UTC,
    uuid_str,
    resolve_tz,
    send_email,
    build_appt_email,
)
from services.services_scheduling import (
    owner_calendar_snapshot,
    list_owner_appointments,
)
from services.services_scheduling import _account_primary_email
from services.payments import (
    get_default_price_cents,
    compute_price_cents,
    _service_price_map,
)
from services.wallets import auto_apply_wallet_funds
from services.features import get_owner_flag

from datetime import timezone as _tz
from sqlalchemy import func
import sqlalchemy as sa
from datetime import time as _time
import json
import uuid

ACTIVE_STATUSES = {"booked", "confirmed", "pending"}


def _confirm_required(detail: dict | str, status_code: int = 409) -> HTTPException:
    """Build a standardized 409 'confirm required' HTTPException detail.

    Keeps existing behavior: prefix detail with 'CONFIRM_REQUIRED:' and JSON-encode
    dict payloads exactly like current inline sites.
    """
    payload = json.dumps(detail) if isinstance(detail, dict) else str(detail)
    return HTTPException(status_code=status_code, detail="CONFIRM_REQUIRED:" + payload)


def _fmt_local_range(s_utc: datetime, e_utc: datetime, tz_name: str) -> str:
    """Format a UTC interval in the given timezone for human display."""
    tz = ZoneInfo(tz_name)
    return f"{s_utc.astimezone(tz).strftime('%a %b %d, %I:%M %p').replace(' 0', ' ')} → {e_utc.astimezone(tz).strftime('%I:%M %p').lstrip('0')}"


def _owner_tzname(user: User | None) -> str:
    """Return the owner's timezone string or 'UTC' as a fallback."""
    return getattr(user, "timezone", None) or "UTC"


def _local_day_bounds(d: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Return local [start_of_day, end_of_day] converted to UTC."""
    start_local = datetime.combine(d, _time.min).replace(tzinfo=tz)
    end_local = datetime.combine(d, _time.max).replace(tzinfo=tz)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def _collect_conflicts(
    db: Session,
    owner_id: str,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[list[Appointment], list[TimeOff]]:
    """Return existing appointments and time-off rows that overlap [start, end).

    Args:
        db: Active SQLAlchemy session.
        owner_id: Owner identifier to scope the query.
        start_utc: Start of the interval (UTC, inclusive of overlaps).
        end_utc: End of the interval (UTC, inclusive of overlaps).

    Returns:
        A 2-tuple of lists: (overlapping appointments, overlapping time-off rows).
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
    owner: User, conflict_appts: list[Appointment], conflict_offs: list[TimeOff]
) -> list[str]:
    """Return short, human-readable conflict summaries for errors/warnings."""
    conflicts_human: list[str] = []
    if conflict_offs:
        conflicts_human.append(
            "Time Off: "
            + "; ".join(
                _fmt_local_range(x.start_utc, x.end_utc, owner.timezone)
                for x in conflict_offs[:5]
            )
            + (f" (+{len(conflict_offs) - 5} more)" if len(conflict_offs) > 5 else "")
        )
    if conflict_appts:
        conflicts_human.append(
            "Appointments: "
            + "; ".join(
                _fmt_local_range(x.start_utc, x.end_utc, owner.timezone)
                for x in conflict_appts[:5]
            )
            + (f" (+{len(conflict_appts) - 5} more)" if len(conflict_appts) > 5 else "")
        )
    return conflicts_human


router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])
"""Router for owner-facing scheduling and appointment endpoints."""
log = logging.getLogger(__name__)


def _log_evt(name: str, **fields) -> None:
    """Emit a single-line structured log for easy grep.

    Example: event=reschedule_attempt rid=... owner_id=... appt_id=...
    """
    try:
        parts = [f"{k}={fields[k]}" for k in sorted(fields.keys())]
        log.info("event=%s %s", name, " ".join(parts))
    except Exception:
        try:
            log.info("event=%s", name)
        except Exception:
            pass


def _mask_email(s: str | None) -> str | None:
    """Mask an email for logs, keeping domain and first letter.

    Examples:
      - "alice@example.com" → "a***@example.com"
      - "x@ex.com" → "*@ex.com"
    """
    if not s:
        return s
    try:
        local, _, domain = s.partition("@")
        if not domain:
            return s
        if len(local) <= 1:
            return f"*@{domain}"
        return f"{local[0]}***@{domain}"
    except Exception:
        return s


class UpdateAppointmentPayload(BaseModel):
    """Owner-initiated appointment updates.

    Fields are optional to allow partial updates; unset values keep current
    appointment values. When 'allow_override' is false, conflicts trigger a
    409 requiring confirmation.
    """

    client_email: Optional[EmailStr] = None
    start_local: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    status: Optional[Literal["booked", "completed", "canceled"]] = None
    allow_override: bool = False
    message: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


def _collect_unique_account_recipients_for_person_ids(
    db: Session, owner_id: str, person_ids: list[int]
) -> list[tuple[str, str | None]]:
    """Return unique (to_email, to_name) per client account owning any of person_ids.

    - Uses ClientAccount primary email when available; falls back to linked auth.User email.
    - Deduplicates by account id so a parent with multiple children receives just one email.
    """
    if not person_ids:
        return []
    try:
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
            User as AuthUser,
        )

        rows = (
            db.query(
                PersonModel.id,
                ClientAccountModel.id,
                ClientAccountModel.name,
                ClientAccountModel.client_user_id,
            )
            .join(ClientAccountModel, ClientAccountModel.id == PersonModel.account_id)
            .filter(
                ClientAccountModel.owner_user_id == owner_id,
                ClientAccountModel.deleted_at.is_(None),
                PersonModel.id.in_([int(p) for p in person_ids]),
            )
            .all()
        )
        seen_accts: set[int] = set()
        recips: list[tuple[str, str | None]] = []
        for _pid, acct_id, acct_name, client_user_id in rows:
            if acct_id in seen_accts:
                continue
            seen_accts.add(int(acct_id))
            email = _account_primary_email(db, int(acct_id))
            if not email and client_user_id:
                u = db.query(AuthUser).filter(AuthUser.id == client_user_id).first()
                email = getattr(u, "email", None)
                if not acct_name:
                    acct_name = getattr(u, "name", None)
            if email:
                recips.append((email, acct_name))
        return recips
    except Exception:
        return []


@router.put("/appointments/{appt_id}", response_model=dict)
def update_appointment(
    appt_id: str,
    payload: UpdateAppointmentPayload,
    background_tasks: BackgroundTasks,
    request: Request = None,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Owner reschedule/retarget endpoint with conflict checks and messaging.

    Supports changing client, start time, duration, and status with optional
    confirmation flow when overlaps are detected.
    """
    rid = getattr(getattr(request, "state", None), "request_id", None)
    _log_evt(
        "reschedule_attempt",
        rid=rid,
        owner_id=str(user.sub),
        appt_id=str(appt_id),
        allow_override=bool(payload.allow_override),
        status=str(payload.status),
    )
    appt = db.query(Appointment).filter_by(id=appt_id, owner_id=user.sub).first()
    if not appt:
        raise HTTPException(404, "Appointment not found")

    owner = db.query(User).filter(User.id == user.sub).first()
    owner_tz = ZoneInfo(owner.timezone)

    # 1) Reassign client if requested
    if payload.client_email:
        client = (
            db.query(User)
            .filter(User.email == payload.client_email, User.role == RoleEnum.CLIENT)
            .first()
        )
        if not client:
            raise HTTPException(
                400, f"Client with email '{payload.client_email}' not found."
            )
        appt.client_id = client.id

    # 2) Compute new times (if rescheduling)
    current_start_local = appt.start_utc.astimezone(owner_tz)
    current_duration = int((appt.end_utc - appt.start_utc).total_seconds() // 60)

    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local and payload.start_local.tzinfo is None
        else (
            payload.start_local.astimezone(owner_tz)
            if payload.start_local
            else current_start_local
        )
    )
    duration = (
        payload.duration_minutes
        if payload.duration_minutes is not None
        else current_duration
    )
    if duration <= 0:
        raise HTTPException(400, "duration_minutes must be positive")
    end_local = start_local + timedelta(minutes=duration)

    new_start_utc, new_end_utc = start_local.astimezone(UTC), end_local.astimezone(UTC)

    # 3) If not overriding, prevent conflicts (skip when canceling)
    if (payload.status != "canceled") and (not payload.allow_override):
        conflict_appts = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.id != appt.id,
                Appointment.status != "canceled",
                Appointment.end_utc > new_start_utc,
                Appointment.start_utc < new_end_utc,
            )
            .order_by(Appointment.start_utc.asc())
            .all()
        )
        conflict_offs = (
            db.query(TimeOff)
            .filter(
                TimeOff.owner_id == owner.id,
                TimeOff.end_utc > new_start_utc,
                TimeOff.start_utc < new_end_utc,
            )
            .order_by(TimeOff.start_utc.asc())
            .all()
        )
        if (
            (conflict_appts or conflict_offs)
            and not payload.allow_override
            and not (payload.message and "confirm" in payload.message.lower())
        ):
            _log_evt(
                "reschedule_conflict",
                rid=rid,
                owner_id=str(user.sub),
                appt_id=str(appt.id),
                conflicts_appts=len(conflict_appts),
                conflicts_timeoff=len(conflict_offs),
            )
            conflicts_human = []
            if conflict_offs:
                conflicts_human.append(
                    "Time Off: "
                    + "; ".join(
                        _fmt_local_range(x.start_utc, x.end_utc, owner.timezone)
                        for x in conflict_offs[:5]
                    )
                    + (
                        f" (+{len(conflict_offs) - 5} more)"
                        if len(conflict_offs) > 5
                        else ""
                    )
                )
            if conflict_appts:
                conflicts_human.append(
                    "Appointments: "
                    + "; ".join(
                        _fmt_local_range(x.start_utc, x.end_utc, owner.timezone)
                        for x in conflict_appts[:5]
                    )
                    + (
                        f" (+{len(conflict_appts) - 5} more)"
                        if len(conflict_appts) > 5
                        else ""
                    )
                )
            payload_to_replay = {
                "endpoint": f"/api/scheduling/appointments/{appt.id}",
                "method": "PUT",
                "body": {
                    "client_email": payload.client_email,
                    "start_local": (
                        start_local.isoformat(timespec="minutes")
                        if payload.start_local
                        else None
                    ),
                    "duration_minutes": (
                        int(duration) if payload.duration_minutes is not None else None
                    ),
                    "status": payload.status,
                    "allow_override": True,
                    "message": payload.message,
                },
            }
            raise _confirm_required(
                {
                    "human": "Reschedule conflicts detected. Reply 'confirm' to proceed anyway, or adjust the time.",
                    "pending_http": payload_to_replay,
                    "conflicts": conflicts_human,
                }
            )

    # Respect DB unique (owner_id,start_utc,person_id): prevent double-booking the same person only.
    if payload.status != "canceled" and getattr(appt, "person_id", None) is not None:
        dup_same_person_start = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.id != appt.id,
                Appointment.start_utc == new_start_utc,
                Appointment.person_id == appt.person_id,
                Appointment.status != "canceled",
            )
            .first()
        )
        if dup_same_person_start:
            raise HTTPException(
                409, "This person already has an appointment at that time."
            )

    # 4) Persist new time window
    if payload.status != "canceled":
        appt.start_utc, appt.end_utc = new_start_utc, new_end_utc

    # 5) Status change (optional) + wallet/bundle handling when canceling
    prev_status = appt.status or "booked"
    if payload.status:
        appt.status = payload.status
        if payload.status == "canceled" and prev_status != "canceled":
            if hasattr(appt, "cancel_reason") and payload.message:
                appt.cancel_reason = payload.message
            now_local2 = datetime.now(owner_tz)
            qualifies_full_refund2 = (
                appt.start_utc.astimezone(owner_tz) - now_local2
            ) >= timedelta(hours=24)
            if qualifies_full_refund2:
                restored_any = False
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
                                note="Auto-restore on status=canceled",
                            )
                        )
                        restored_any = True
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
                                note="Auto-restore on status=canceled",
                            )
                        )
                        restored_any = True
                # Deposit any cash paid to client's wallet
                try:
                    cash_paid2 = int(getattr(appt, "amount_paid_cents", 0) or 0)
                except Exception:
                    cash_paid2 = 0
                if cash_paid2 > 0 and getattr(appt, "client_id", None):
                    wallet2 = (
                        db.query(PrepaidBundle)
                        .filter(
                            PrepaidBundle.owner_id == owner.id,
                            PrepaidBundle.client_id == str(appt.client_id),
                            PrepaidBundle.total_credits == 0,
                        )
                        .order_by(PrepaidBundle.created_at.desc())
                        .first()
                    )
                    if not wallet2:
                        wallet2 = PrepaidBundle(
                            owner_id=str(owner.id),
                            client_id=str(appt.client_id),
                            name="Wallet",
                            total_credits=0,
                            remaining_credits=0,
                            price_cents=0,
                            currency="USD",
                        )
                        db.add(wallet2)
                        db.flush()
                    db.add(
                        PrepaidLedger(
                            bundle_id=wallet2.id,
                            event="refund",
                            delta_credits=0,
                            amount_cents=int(cash_paid2),
                            appointment_id=appt.id,
                            note="Full refund to wallet on >=24h cancel",
                        )
                    )
                    appt.payment_status = "refunded"
                    try:
                        appt.amount_paid_cents = 0
                    except Exception:
                        pass
                elif restored_any:
                    appt.payment_status = "refunded"
                try:
                    appt.bundle_id = None
                except Exception:
                    pass
            else:
                pass

    # 6) Lesson target update (person on client's account)
    try:
        if payload.lesson_person_id is not None or (
            payload.lesson_person_name and payload.lesson_person_name.strip()
        ):
            from app.models import ClientAccount, Person

            acct = None
            if getattr(appt, "client_id", None):
                acct = (
                    db.query(ClientAccount)
                    .filter(
                        ClientAccount.owner_user_id == owner.id,
                        ClientAccount.client_user_id == appt.client_id,
                        ClientAccount.deleted_at.is_(None),
                    )
                    .first()
                )
            if acct is not None:
                person = None
                if payload.lesson_person_id is not None:
                    person = (
                        db.query(Person)
                        .filter(
                            Person.id == int(payload.lesson_person_id),
                            Person.account_id == acct.id,
                        )
                        .first()
                    )
                if person is None and payload.lesson_person_name:
                    nm = payload.lesson_person_name.strip()
                    if nm:
                        person = (
                            db.query(Person)
                            .filter(
                                Person.account_id == acct.id,
                                func.lower(Person.full_name) == func.lower(nm),
                            )
                            .first()
                        )
                        if person is None:
                            person = Person(
                                account_id=acct.id, full_name=nm, email=None
                            )
                            db.add(person)
                            db.flush()
                if person is not None:
                    appt.person_id = person.id
    except Exception as e:
        try:
            log.warning(
                "owner_appointments: lesson target update failed: appt_id=%s err=%s",
                getattr(appt, "id", None),
                e,
            )
        except Exception:
            pass
    old_start_local = current_start_local
    old_end_local = current_start_local + timedelta(minutes=current_duration)

    try:
        db.commit()
        db.refresh(appt)
    except IntegrityError:
        db.rollback()
        _log_evt(
            "reschedule_integrity_error",
            rid=rid,
            owner_id=str(user.sub),
            appt_id=str(appt_id),
        )
        raise HTTPException(
            409, "Another appointment already exists at that exact start time."
        )
    else:
        try:
            _log_evt(
                "reschedule_ok",
                rid=rid,
                owner_id=str(user.sub),
                appt_id=str(appt.id),
                start_utc=appt.start_utc.isoformat(),
                end_utc=appt.end_utc.isoformat(),
                status=str(appt.status),
            )
        except Exception:
            pass

    # === Email: OWNER acted -> notify CLIENT (if exists) ===
    client = db.get(User, appt.client_id) if appt.client_id else None
    if client and client.email:
        new_start_local = appt.start_utc.astimezone(owner_tz)
        new_end_local = appt.end_utc.astimezone(owner_tz)

        if payload.status == "canceled":
            email_pkg = build_appt_email(
                audience="client",
                action="canceled",
                owner=owner,
                start_local=new_start_local,
                end_local=new_end_local,
                appointment_id=str(appt.id),
                initiator_label=owner.name or "the owner",
                status_label=appt.status,
                recipient_name=client.name or client.email,
                message=payload.message,
                include_ics=False,
                organizer_email=owner.email,
                attendee_email=client.email,
            )
        else:
            email_pkg = build_appt_email(
                audience="client",
                action="updated",
                owner=owner,
                start_local=new_start_local,
                end_local=new_end_local,
                appointment_id=str(appt.id),
                initiator_label=owner.name or "the owner",
                status_label=appt.status,
                recipient_name=client.name or client.email,
                message=payload.message,
                old_start_local=old_start_local,
                old_end_local=old_end_local,
                include_ics=True,
                organizer_email=owner.email,
                attendee_email=client.email,
            )

        background_tasks.add_task(
            send_email,
            client.email,
            email_pkg.subject,
            email_pkg.text,
            email_pkg.html,
            email_pkg.ics_text,
        )
    try:
        wallet_ids = (
            db.query(PrepaidBundle.id)
            .filter(
                PrepaidBundle.owner_id == owner.id,
                PrepaidBundle.client_id == str(client.id),
                PrepaidBundle.total_credits == 0,
            )
            .order_by(PrepaidBundle.created_at.desc())
            .all()
        )
        for (wid,) in wallet_ids:
            _log_evt(
                "wallet_auto_apply_attempt",
                rid=None,
                owner_id=str(owner.id),
                appt_id=str(appt.id),
                bundle_id=int(wid),
            )
            auto_apply_wallet_funds(
                db,
                owner_id=str(owner.id),
                bundle_id=int(wid),
                note_prefix="Auto-apply wallet funds after booking",
            )
    except Exception:
        pass
    # Fallback: if still owed, directly attach the latest wallet and consume owed
    try:
        appt_refreshed = (
            db.query(Appointment)
            .filter(Appointment.id == appt.id, Appointment.owner_id == owner.id)
            .first()
        )
        price_map = _service_price_map(db, owner_user_id=owner.id)
        expected = compute_price_cents(db, appt_refreshed, price_map) or 0
        if int(expected) > 0:
            cash_paid = int(getattr(appt_refreshed, "amount_paid_cents", 0) or 0)
            ledger_total = (
                db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                .filter(PrepaidLedger.appointment_id == appt_refreshed.id)
                .scalar()
            ) or 0
            already_applied = -int(ledger_total) if int(ledger_total) < 0 else 0
            owed = max(int(expected) - cash_paid - already_applied, 0)
            if owed > 0:
                wallet = (
                    db.query(PrepaidBundle)
                    .filter(
                        PrepaidBundle.owner_id == owner.id,
                        PrepaidBundle.client_id == str(client.id),
                        PrepaidBundle.total_credits == 0,
                    )
                    .order_by(PrepaidBundle.created_at.desc())
                    .first()
                )
                if wallet:
                    balance = int(
                        db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                        .filter(PrepaidLedger.bundle_id == wallet.id)
                        .scalar()
                        or 0
                    )
                    use = min(balance, owed)
                    if use > 0:
                        try:
                            _log_evt(
                                "wallet_direct_consume",
                                rid=None,
                                owner_id=str(owner.id),
                                appt_id=str(appt_refreshed.id),
                                bundle_id=int(wallet.id),
                                used=int(use),
                                owed=int(owed),
                            )
                        except Exception:
                            pass
                        appt_refreshed.bundle_id = wallet.id
                        db.add(
                            PrepaidLedger(
                                bundle_id=wallet.id,
                                event="consume",
                                delta_credits=0,
                                amount_cents=-int(use),
                                appointment_id=appt_refreshed.id,
                                note="Auto-apply wallet on booking (direct)",
                            )
                        )
                        if (cash_paid + already_applied + int(use)) >= int(expected):
                            appt_refreshed.payment_status = "paid"
                            if getattr(appt_refreshed, "paid_at", None) is None:
                                appt_refreshed.paid_at = datetime.now(_tz.utc)
                        db.add(appt_refreshed)
                        db.commit()
    except Exception as e:
        try:
            log.warning(
                "owner_appointments: direct wallet attach/consume failed: appt_id=%s err=%s",
                getattr(appt_refreshed, "id", None),
                e,
            )
        except Exception:
            pass

    return {"ok": True, "appointment_id": appt.id, "status": appt.status}


@router.patch("/appointments/{appt_id}/owner", response_model=dict)
def update_appointment_owner(
    appt_id: str,
    payload: AppointmentUpdateOwner,
    background_tasks: BackgroundTasks,
    request: Request = None,
    apply_wallet_now: bool = Query(
        False, description="If true, immediately apply wallet funds up to owed"
    ),
    restore_wallet_now: bool = Query(
        False,
        description="If true, restore any funds applied from the current wallet for this appointment",
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Update owner-managed fields (attendance, notes, payments, wallet).

    Also supports explicit wallet apply/restore actions without status change.
    """
    appt = (
        db.query(Appointment)
        .options(noload(Appointment.person))
        .filter_by(id=appt_id, owner_id=user.sub)
        .with_for_update(of=Appointment)
        .first()
    )
    if not appt:
        raise HTTPException(404, "Appointment not found")

    prev_payment_status = appt.payment_status or "unpaid"
    prev_bundle_id = appt.bundle_id

    # --- Basic owner fields ---
    if payload.attendance_status is not None:
        appt.attendance_status = payload.attendance_status

    if payload.late_minutes is not None:
        appt.late_minutes = int(payload.late_minutes)
        if appt.late_minutes > 0 and (appt.attendance_status or "unknown") == "unknown":
            appt.attendance_status = "late"

    if payload.owner_private_note is not None:
        appt.owner_private_note = payload.owner_private_note

    if payload.amount_paid_cents is not None:
        appt.amount_paid_cents = int(payload.amount_paid_cents)
    if payload.price_override_cents is not None:
        v = int(payload.price_override_cents)
        if v < 0:
            raise HTTPException(400, "price_override_cents must be >= 0")
        appt.price_override_cents = v

    # --- Payment status & paid_at ---
    if payload.payment_status is not None:
        appt.payment_status = payload.payment_status
        if appt.payment_status == "paid":
            if appt.paid_at is None:
                appt.paid_at = payload.paid_at or datetime.now(_tz.utc)
        elif appt.payment_status in {"unpaid", "refunded", "waived"}:
            appt.paid_at = payload.paid_at  # could be None to clear
        if appt.payment_status == "refunded":
            try:
                appt.amount_paid_cents = 0
            except Exception:
                pass
            try:
                appt.price_override_cents = 0
            except Exception:
                pass

    # --- Appointment status (support cancel with wallet refund flow) ---
    if getattr(payload, "status", None) is not None:
        new_status_in = str(payload.status)
        if new_status_in == "canceled" and (appt.status or "booked") != "canceled":
            owner = db.query(User).filter(User.id == user.sub).first()
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
            if qualifies_full_refund:
                if getattr(appt, "bundle_id", None):
                    bid = int(appt.bundle_id)
                    net_now = (
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
                    if int(net_now) == -1:
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
                                note="Auto-restore on owner status=canceled",
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
                                note="Auto-restore funds on owner status=canceled",
                            )
                        )
                try:
                    cash_paid = int(getattr(appt, "amount_paid_cents", 0) or 0)
                except Exception:
                    cash_paid = 0
                if cash_paid > 0 and getattr(appt, "client_id", None):
                    wallet = (
                        db.query(PrepaidBundle)
                        .filter(
                            PrepaidBundle.owner_id == user.sub,
                            PrepaidBundle.client_id == str(appt.client_id),
                            PrepaidBundle.total_credits == 0,
                        )
                        .order_by(PrepaidBundle.created_at.desc())
                        .first()
                    )
                    if not wallet:
                        wallet = PrepaidBundle(
                            owner_id=user.sub,
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
                            note="Full refund to wallet on owner status=canceled",
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
            appt.status = "canceled"

    # --- Bundle attach/detach (validate) ---
    if payload.bundle_id is not None:
        target_bundle_id = payload.bundle_id or None
        if target_bundle_id is None:
            appt.bundle_id = None
        else:
            bundle = db.get(PrepaidBundle, target_bundle_id)
            if (
                not bundle
                or bundle.owner_id != user.sub
                or (appt.client_id and bundle.client_id != appt.client_id)
            ):
                raise HTTPException(400, "Invalid bundle")
            appt.bundle_id = bundle.id
    if payload.amount_paid_cents is not None:
        appt.amount_paid_cents = int(payload.amount_paid_cents)

    new_payment_status = appt.payment_status or "unpaid"
    became_paid = prev_payment_status != "paid" and new_payment_status == "paid"
    became_unpaid = prev_payment_status == "paid" and new_payment_status in {
        "unpaid",
        "refunded",
        "waived",
    }
    bundle_changed = prev_bundle_id != appt.bundle_id

    # ------- Ledger helpers (no schema change) -------
    def _net_for_bundle(bid: int | None) -> int:
        if not bid:
            return 0
        net = (
            db.query(func.coalesce(func.sum(PrepaidLedger.delta_credits), 0))
            .filter(
                PrepaidLedger.bundle_id == bid,
                PrepaidLedger.appointment_id == appt.id,
                PrepaidLedger.event.in_(["consume", "restore", "revert"]),
            )
            .scalar()
        )
        return int(net or 0)

    def _consume_one(bid: int):
        bundle = db.get(PrepaidBundle, bid)
        if not bundle:
            raise HTTPException(400, "Bundle not found")
        if bundle.remaining_credits <= 0:
            raise HTTPException(400, "No remaining credits in bundle")
        bundle.remaining_credits -= 1
        db.add(bundle)
        db.add(
            PrepaidLedger(
                bundle_id=bid,
                event="consume",
                delta_credits=-1,
                amount_cents=0,
                appointment_id=appt.id,
                note="Auto-consume on payment",
            )
        )

    def _restore_one(bid: int, note: str):
        bundle = db.get(PrepaidBundle, bid)
        if not bundle:
            return
        bundle.remaining_credits += 1
        db.add(bundle)
        db.add(
            PrepaidLedger(
                bundle_id=bid,
                event="restore",
                delta_credits=+1,
                amount_cents=0,
                appointment_id=appt.id,
                note=note,
            )
        )

    # ------- Monetary (store-credit) helpers -------
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
                PrepaidLedger.bundle_id == bid, PrepaidLedger.appointment_id == appt.id
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
                note="Auto-consume funds on payment",
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

    if bundle_changed and prev_bundle_id:
        net_old = _net_for_bundle(prev_bundle_id)
        if net_old == -1:
            _restore_one(prev_bundle_id, "Auto-restore on bundle change")
        _restore_amount(prev_bundle_id, "Auto-restore funds on bundle change")

    # --- If became paid and we have a wallet, consume up to owed ---
    rid = getattr(getattr(request, "state", None), "request_id", None)

    if became_paid and appt.bundle_id:
        from services.payments import compute_price_cents, _service_price_map

        price_map = _service_price_map(db, owner_user_id=user.sub)
        expected = compute_price_cents(db, appt, price_map)
        cash = int(appt.amount_paid_cents or 0)
        already = _appt_spend_cents(int(appt.bundle_id))
        already_applied = -int(already) if int(already) < 0 else 0
        owed = max(int(expected or 0) - cash - already_applied, 0)
        if owed > 0:
            _log_evt(
                "wallet_consume_on_paid",
                rid=rid,
                owner_id=str(user.sub),
                appt_id=str(appt.id),
                bundle_id=int(appt.bundle_id),
                owed=int(owed),
            )
            _consume_amount(appt.bundle_id, owed)
    elif restore_wallet_now and appt.bundle_id:
        _log_evt(
            "wallet_restore_manual",
            rid=rid,
            owner_id=str(user.sub),
            appt_id=str(appt.id),
            bundle_id=int(appt.bundle_id),
        )
        _restore_amount(appt.bundle_id, "Manual restore via owner edit")
    elif apply_wallet_now and appt.bundle_id:
        try:
            from services.payments import compute_price_cents, _service_price_map

            price_map = _service_price_map(db, owner_user_id=user.sub)
            expected = compute_price_cents(db, appt, price_map)
        except Exception:
            expected = None
        cash = int(appt.amount_paid_cents or 0)
        already = _appt_spend_cents(int(appt.bundle_id))
        already_applied = -int(already) if int(already) < 0 else 0
        owed = max(int(expected or 0) - cash - already_applied, 0)
        if owed > 0:
            _log_evt(
                "wallet_consume_manual",
                rid=rid,
                owner_id=str(user.sub),
                appt_id=str(appt.id),
                bundle_id=int(appt.bundle_id),
                owed=int(owed),
            )
            _consume_amount(appt.bundle_id, owed)

    # --- If became unpaid/refunded/waived and we have a bundle, ensure net is 0 (restore if needed) ---
    if became_unpaid and appt.bundle_id:
        net_new = _net_for_bundle(appt.bundle_id)
        if net_new == -1:
            _restore_one(appt.bundle_id, f"Auto-restore on status={new_payment_status}")
        _log_evt(
            "wallet_restore_on_unpaid",
            rid=rid,
            owner_id=str(user.sub),
            appt_id=str(appt.id),
            bundle_id=int(appt.bundle_id),
            status=str(new_payment_status),
        )
        _restore_amount(
            appt.bundle_id, f"Auto-restore funds on status={new_payment_status}"
        )
    canceled_now = getattr(payload, "status", None) == "canceled"

    db.add(appt)
    db.commit()
    db.refresh(appt)
    try:
        if canceled_now and getattr(appt, "client_id", None):
            owner = db.query(User).filter(User.id == user.sub).first()
            client = db.get(User, appt.client_id)
            if owner and client and getattr(client, "email", None):
                tz = (
                    ZoneInfo(owner.timezone)
                    if getattr(owner, "timezone", None)
                    else ZoneInfo("UTC")
                )
                start_local = appt.start_utc.astimezone(tz)
                end_local = appt.end_utc.astimezone(tz)
                pkg = build_appt_email(
                    audience="client",
                    action="canceled",
                    owner=owner,
                    start_local=start_local,
                    end_local=end_local,
                    appointment_id=str(appt.id),
                    initiator_label=owner.name or "the owner",
                    status_label=appt.status,
                    recipient_name=(client.name or client.email),
                    message=None,
                    include_ics=False,
                    organizer_email=owner.email,
                    attendee_email=client.email,
                )
                background_tasks.add_task(
                    send_email,
                    client.email,
                    pkg.subject,
                    pkg.text,
                    pkg.html,
                    pkg.ics_text,
                )
    except Exception:
        pass

    return {"ok": True, "notice": "Appointment updated"}


@router.get("/appointments", response_model=list[OwnerAppointmentRowOut])
def list_appointments(
    response: Response,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
    start: Optional[date] = Query(
        None, description="Owner-local inclusive start date (YYYY-MM-DD)"
    ),
    end: Optional[date] = Query(
        None, description="Owner-local inclusive end date (YYYY-MM-DD)"
    ),
    client_query: Optional[str] = Query(
        None, description="Filter by client name or email (icontains)"
    ),
    tz: Optional[str] = Query(None, description="IANA timezone for local timestamps"),
    limit: int = Query(
        200, ge=1, le=500, description="Max rows to return (1..500), default 200"
    ),
    offset: int = Query(0, ge=0, description="Row offset for pagination"),
) -> list[dict[str, Any]]:
    """List appointments for the owner with optional date and client filters.

    Dates are interpreted in the owner's timezone and mapped to UTC for
    storage filtering. When 'client_query' is provided, it matches case-
    insensitively on the client's name or email.
    """
    # FastAPI's Query defaults are instances of fastapi.params.Query when
    # the function is invoked directly (e.g., in unit tests). Normalize
    # those to None/primitive types so comparisons and truthiness checks
    # below behave as intended.
    if not isinstance(start, date):
        start = None
    if not isinstance(end, date):
        end = None
    if not isinstance(client_query, str):
        client_query = None
    if not isinstance(tz, str):
        tz = None

    if end and not start:
        raise HTTPException(
            status_code=422, detail="Provide 'start' before selecting 'end'."
        )
    if start and end and start > end:
        raise HTTPException(
            status_code=422, detail="'start' must be on or before 'end'."
        )

    owner_row = db.query(User).filter(User.id == user.sub).first()
    owner_tz = ZoneInfo(_owner_tzname(owner_row))
    owner_tz_name = getattr(owner_tz, "key", str(owner_tz))
    display_tz = resolve_tz(tz, getattr(owner_row, "timezone", None) or owner_tz_name)
    display_tz_name = getattr(display_tz, "key", str(display_tz))

    q = db.query(Appointment).filter(Appointment.owner_id == user.sub)
    if client_query:
        like = f"%{client_query}%"
        from app.models import User as AuthUser

        q = q.outerjoin(AuthUser, Appointment.client_id == AuthUser.id)
        q = q.filter(or_(AuthUser.name.ilike(like), AuthUser.email.ilike(like)))

    if start and not end:
        s_utc, e_utc = _local_day_bounds(start, owner_tz)  # single day
        q = q.filter(
            and_(Appointment.start_utc >= s_utc, Appointment.start_utc <= e_utc)
        )
    elif start and end:
        s_utc, _ = _local_day_bounds(start, owner_tz)
        _, e_utc = _local_day_bounds(end, owner_tz)
        q = q.filter(
            and_(Appointment.start_utc >= s_utc, Appointment.start_utc <= e_utc)
        )

    total = q.with_entities(func.count(sa.distinct(Appointment.id))).scalar() or 0
    appts = q.order_by(Appointment.start_utc.asc()).limit(limit).offset(offset).all()

    out = []
    for a in appts:
        client = db.get(User, a.client_id) if a.client_id else None
        client_account_id = None
        if a.client_id:
            try:
                acct = (
                    db.query(ClientAccount)
                    .filter(
                        ClientAccount.owner_user_id == user.sub,
                        ClientAccount.client_user_id == a.client_id,
                        ClientAccount.deleted_at.is_(None),
                    )
                    .first()
                )
                if acct:
                    client_account_id = int(acct.id)
            except Exception:
                client_account_id = None
        if client:
            client_obj = {"id": client.id, "name": client.name, "email": client.email}
        else:
            c_name = getattr(a, "client_name", None)
            c_email = getattr(a, "client_email", None)
            client_obj = (
                {"id": None, "name": c_name, "email": c_email}
                if (c_name or c_email)
                else None
            )
        duration_minutes = max(15, int((a.end_utc - a.start_utc).total_seconds() // 60))
        raw_override = getattr(a, "price_override_cents", None)
        effective_price = raw_override
        if effective_price is None:
            try:
                effective_price = get_default_price_cents(
                    db, owner_user_id=user.sub, duration_minutes=duration_minutes
                )
            except Exception:
                effective_price = None

        person_obj = None
        if getattr(a, "person_id", None):
            try:
                from app.models import Person

                p = db.query(Person).filter(Person.id == a.person_id).first()
                if p:
                    person_obj = {"id": p.id, "name": p.full_name, "email": p.email}
            except Exception:
                person_obj = None

        out.append(
            {
                "id": str(a.id),
                "start_utc": a.start_utc,
                "end_utc": a.end_utc,
                "group_id": str(getattr(a, "group_id", "") or "") or None,
                "start_local": a.start_utc.astimezone(display_tz),
                "end_local": a.end_utc.astimezone(display_tz),
                "timezone": display_tz_name,
                "status": a.status,
                "client": client_obj,
                "client_account_id": client_account_id,
                "person": person_obj,
                "cancel_reason": getattr(a, "cancel_reason", None),
                "owner_note": getattr(a, "owner_private_note", None),
                "client_note": getattr(a, "client_previsit_note", None),
                "paid": getattr(a, "payment_status", None) == "paid",
                "late": getattr(a, "attendance_status", None) == "late",
                "no_show": getattr(a, "attendance_status", None) == "no_show",
                "amount_paid_cents": getattr(a, "amount_paid_cents", None),
                "labels": getattr(a, "labels", None),
                "attendance_status": getattr(a, "attendance_status", "attended"),
                "late_minutes": getattr(a, "late_minutes", 0),
                "payment_status": getattr(a, "payment_status", "unpaid"),
                "paid_at": getattr(a, "paid_at", None),
                "bundle_id": getattr(a, "bundle_id", None),
                "price_override_cents": raw_override,
                "effective_price_cents": effective_price,
            }
        )
    try:
        response.headers["X-Total-Count"] = str(int(total))
        # Use the requested pagination window to compute the next offset,
        # rather than relying on the length of the realized result set
        # (which may not be limited in lightweight test fakes).
        computed_next = offset + int(limit)
        if computed_next < int(total):
            response.headers["X-Next-Offset"] = str(computed_next)
    except Exception:
        pass
    return out


@router.post("/appointments/{appt_id}/cancel", response_model=dict)
def cancel_appointment(
    appt_id: str,
    background_tasks: BackgroundTasks,
    request: Request = None,
    message: Optional[str] = Query(
        None, description="Optional message to include in the cancel email"
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Cancel an appointment, applying refund policy and sending notices.

    Behavior:
    - Marks the appointment canceled.
    - If canceled ≥24h before start (owner-local), restores credit and refunds
      cash to wallet; otherwise leaves financials unchanged.
    - Sends owner and client email notifications when addresses are available.
    """
    rid = getattr(getattr(request, "state", None), "request_id", None)
    _log_evt("cancel_attempt", rid=rid, owner_id=str(user.sub), appt_id=str(appt_id))
    appt = db.query(Appointment).filter_by(id=appt_id, owner_id=user.sub).first()
    if not appt:
        raise HTTPException(404, "Appointment not found")
    if appt.status != "canceled":
        owner = db.query(User).filter(User.id == user.sub).first()
        owner_tz = ZoneInfo(owner.timezone)
        start_local = appt.start_utc.astimezone(owner_tz)
        end_local = appt.end_utc.astimezone(owner_tz)
        now_local = datetime.now(owner_tz)
        qualifies_full_refund = (start_local - now_local) >= timedelta(hours=24)
        _log_evt(
            "cancel_policy",
            rid=rid,
            owner_id=str(user.sub),
            appt_id=str(appt.id),
            branch=("full_refund" if qualifies_full_refund else "no_financial_change"),
        )
        appt.status = "canceled"
        if qualifies_full_refund:
            restored_any = False
            if getattr(appt, "bundle_id", None):
                bid = int(appt.bundle_id)
                from app.models import PrepaidLedger, PrepaidBundle

                net = (
                    db.query(func.coalesce(func.sum(PrepaidLedger.delta_credits), 0))
                    .filter(
                        PrepaidLedger.bundle_id == bid,
                        PrepaidLedger.appointment_id == appt.id,
                        PrepaidLedger.event.in_(["consume", "restore", "revert"]),
                    )
                    .scalar()
                    or 0
                )
                if int(net) == -1:
                    b = db.get(PrepaidBundle, bid)
                    if b:
                        b.remaining_credits = int(b.remaining_credits or 0) + 1
                        db.add(b)
                    db.add(
                        PrepaidLedger(
                            bundle_id=bid,
                            event="restore",
                            delta_credits=+1,
                            amount_cents=0,
                            appointment_id=appt.id,
                            note="Auto-restore on cancel",
                        )
                    )
                    restored_any = True
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
                            note="Auto-restore on cancel",
                        )
                    )
                    restored_any = True
            try:
                cash_paid = int(getattr(appt, "amount_paid_cents", 0) or 0)
            except Exception:
                cash_paid = 0
            if cash_paid > 0 and getattr(appt, "client_id", None):
                from app.models import PrepaidBundle, PrepaidLedger

                wallet = (
                    db.query(PrepaidBundle)
                    .filter(
                        PrepaidBundle.owner_id == user.sub,
                        PrepaidBundle.client_id == str(appt.client_id),
                        PrepaidBundle.total_credits == 0,
                    )
                    .order_by(PrepaidBundle.created_at.desc())
                    .first()
                )
                if not wallet:
                    wallet = PrepaidBundle(
                        owner_id=user.sub,
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
                        note="Full refund to wallet on >=24h cancel",
                    )
                )
                # Normalize appointment payment to refunded and clear cash
                appt.payment_status = "refunded"
                try:
                    appt.amount_paid_cents = 0
                except Exception:
                    pass
            elif restored_any:
                appt.payment_status = "refunded"
            try:
                appt.bundle_id = None
            except Exception:
                pass
        else:
            pass
        db.commit()
        try:
            _log_evt(
                "cancel_ok",
                rid=rid,
                owner_id=str(user.sub),
                appt_id=str(appt.id),
                payment_status=str(getattr(appt, "payment_status", None)),
            )
        except Exception:
            pass
        # Email client about cancellation
        client = db.get(User, appt.client_id) if appt.client_id else None
        if client and client.email:
            if message:
                appt.cancel_reason = message

            email_pkg = build_appt_email(
                audience="client",
                action="canceled",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=str(appt.id),
                initiator_label=owner.name or "the owner",
                status_label=appt.status,
                recipient_name=client.name or client.email,
                message=message,
                include_ics=False,
                organizer_email=owner.email,
                attendee_email=client.email,
            )

            background_tasks.add_task(
                send_email,
                client.email,
                email_pkg.subject,
                email_pkg.text,
                email_pkg.html,
                email_pkg.ics_text,
            )
            try:
                _log_evt(
                    "cancel_email_queued",
                    rid=rid,
                    owner_id=str(user.sub),
                    appt_id=str(appt.id),
                    to=_mask_email(client.email),
                )
            except Exception:
                pass
    return {"ok": True, "appointment_id": appt.id, "status": appt.status}


@router.get("/appointments/{appt_id}/owner", response_model=dict)
def get_appointment_owner_details(
    appt_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Return owner-facing appointment details used by the analytics drawer."""
    appt = db.query(Appointment).filter_by(id=appt_id, owner_id=user.sub).first()
    if not appt:
        raise HTTPException(404, "Appointment not found")
    try:
        now = datetime.now(_tz.utc)
        if (
            appt.end_utc
            and appt.status not in {"completed", "canceled"}
            and appt.end_utc < now
        ):
            appt.status = "completed"
            db.commit()
            db.refresh(appt)
    except Exception:
        pass
    return {
        "id": str(appt.id),
        "owner_private_note": getattr(appt, "owner_private_note", None),
        "attendance_status": getattr(appt, "attendance_status", "attended"),
        "late_minutes": getattr(appt, "late_minutes", 0),
        "payment_status": getattr(appt, "payment_status", None),
        "price_override_cents": getattr(appt, "price_override_cents", None),
        "start_utc": appt.start_utc,
        "end_utc": appt.end_utc,
        "cancel_reason": getattr(appt, "cancel_reason", None),
        "group_id": str(getattr(appt, "group_id", "") or "") or None,
    }


@router.get("/appointments/group/{group_id}", response_model=dict)
def get_group_details(
    group_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Fetch group appointment details including attendee payment summaries."""
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    rows: list[Appointment] = (
        db.query(Appointment)
        .filter(Appointment.owner_id == user.sub, Appointment.group_id == gid)
        .order_by(Appointment.start_utc.asc(), Appointment.id.asc())
        .all()
    )
    if not rows:
        raise HTTPException(404, "Group not found")
    from services.payments import _service_price_map, compute_financials

    price_map = _service_price_map(db, owner_user_id=user.sub)

    def _person_name(a: Appointment) -> str:
        if a.person_id:
            from app.models import Person

            p = db.query(Person).filter(Person.id == a.person_id).first()
            if p and getattr(p, "full_name", None):
                return p.full_name
        return a.client_name or "Client"

    attendees = []
    for a in rows:
        fin = compute_financials(db, a, price_map)
        attendees.append(
            {
                "appointment_id": str(a.id),
                "person_id": a.person_id,
                "name": _person_name(a),
                "status": a.status,
                "payment_status": fin.get("payment_status"),
                "price_cents": fin.get("price_cents"),
                "paid_cash_cents": fin.get("paid_cash_cents"),
                "bundle_applied_cents": fin.get("bundle_applied_cents"),
                "owed_cents": fin.get("owed_cents"),
            }
        )

    return {
        "group_id": group_id,
        "start_utc": rows[0].start_utc,
        "end_utc": rows[0].end_utc,
        "attendees": attendees,
    }


@router.delete("/appointments/{appt_id}", response_model=dict)
def delete_appointment(
    appt_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Hard-delete an appointment owned by the current user.

    Safely unwinds any wallet funds and a single consumed bundle credit
    associated with the appointment before deletion.
    """
    appt = db.query(Appointment).filter_by(id=appt_id, owner_id=user.sub).first()
    if not appt:
        raise HTTPException(404, "Appointment not found")
    from app.models import PrepaidLedger, PrepaidBundle

    def _net_for_bundle(bid: int | None) -> int:
        if not bid:
            return 0
        net = (
            db.query(func.coalesce(func.sum(PrepaidLedger.delta_credits), 0))
            .filter(
                PrepaidLedger.bundle_id == bid,
                PrepaidLedger.appointment_id == appt.id,
                PrepaidLedger.event.in_(["consume", "restore", "revert"]),
            )
            .scalar()
        )
        return int(net or 0)

    def _restore_one(bid: int, note: str):
        bundle = db.get(PrepaidBundle, bid)
        if not bundle:
            return
        bundle.remaining_credits = int(bundle.remaining_credits or 0) + 1
        db.add(bundle)
        db.add(
            PrepaidLedger(
                bundle_id=bid,
                event="restore",
                delta_credits=+1,
                amount_cents=0,
                appointment_id=appt.id,
                note=note,
            )
        )

    def _restore_amount(bid: int, note: str):
        spent = (
            db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
            .filter(
                PrepaidLedger.bundle_id == bid, PrepaidLedger.appointment_id == appt.id
            )
            .scalar()
        )
        spent = int(spent or 0)
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

    if getattr(appt, "bundle_id", None):
        bid = int(appt.bundle_id)
        if _net_for_bundle(bid) == -1:
            _restore_one(bid, "Auto-restore on appointment delete")
        _restore_amount(bid, "Auto-restore funds on appointment delete")
    db.delete(appt)
    db.commit()
    return {"ok": True}


@router.get("/owner/appointments", response_model=AppointmentListOut)
def owner_appointments(
    filter: str,
    tz: str | None = None,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """List the owner's appointments using common pre-defined filters."""
    tz_str = tz or getattr(user, "timezone", None) or "America/Toronto"
    rows = list_owner_appointments(db, owner_id=user.sub, flt=filter, tz_str=tz_str)
    return AppointmentListOut(rows=[AppointmentRow(**r) for r in rows])


@router.get("/owner/snapshot", response_model=SnapshotOut)
def owner_snapshot(
    scope: str = "week",
    anchor: date | None = None,
    tz: str | None = None,
    pad_edges: bool = Query(
        False,
        description="If true, pad appointment edges by owner setting for visual safety",
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Return a snapshot of the owner's calendar window for the UI/LLM.

    Mirrors owner_calendar_snapshot and enriches with availability context
    to support downstream rendering and summaries.
    """
    tz_str = tz or getattr(user, "timezone", None) or "America/Toronto"
    base = owner_calendar_snapshot(
        db, owner_id=user.sub, scope=scope, anchor=anchor, tz_str=tz_str
    )

    from services.services_scheduling import (
        _apply_precedence,
        _collect_time_off,
        _expand_weekly_rules,
        _as_aware_utc,
        UTC,
    )
    from sqlalchemy import select, and_, or_

    window_start = base["start"]
    window_end = base["end"]
    rules = (
        db.execute(
            select(AvailabilityRule).where(AvailabilityRule.owner_id == user.sub)
        )
        .scalars()
        .all()
    )
    offs = (
        db.execute(
            select(TimeOff).where(
                TimeOff.owner_id == user.sub,
                or_(
                    and_(
                        TimeOff.start_utc >= window_start.astimezone(UTC),
                        TimeOff.start_utc < window_end.astimezone(UTC),
                    ),
                    and_(
                        TimeOff.end_utc > window_start.astimezone(UTC),
                        TimeOff.end_utc <= window_end.astimezone(UTC),
                    ),
                    and_(
                        TimeOff.start_utc <= window_start.astimezone(UTC),
                        TimeOff.end_utc >= window_end.astimezone(UTC),
                    ),
                ),
            )
        )
        .scalars()
        .all()
    )
    appts = (
        db.execute(
            select(Appointment).where(
                Appointment.owner_id == user.sub,
                Appointment.status.in_(["booked", "completed", "canceled"]),
                or_(
                    and_(
                        Appointment.start_utc >= window_start.astimezone(UTC),
                        Appointment.start_utc < window_end.astimezone(UTC),
                    ),
                    and_(
                        Appointment.end_utc > window_start.astimezone(UTC),
                        Appointment.end_utc <= window_end.astimezone(UTC),
                    ),
                    and_(
                        Appointment.start_utc <= window_start.astimezone(UTC),
                        Appointment.end_utc >= window_end.astimezone(UTC),
                    ),
                ),
            )
        )
        .scalars()
        .all()
    )
    specials = (
        db.execute(
            select(SpecialOpening).where(
                SpecialOpening.owner_id == user.sub,
                or_(
                    and_(
                        SpecialOpening.start_utc >= window_start.astimezone(UTC),
                        SpecialOpening.start_utc < window_end.astimezone(UTC),
                    ),
                    and_(
                        SpecialOpening.end_utc > window_start.astimezone(UTC),
                        SpecialOpening.end_utc <= window_end.astimezone(UTC),
                    ),
                    and_(
                        SpecialOpening.start_utc <= window_start.astimezone(UTC),
                        SpecialOpening.end_utc >= window_end.astimezone(UTC),
                    ),
                ),
            )
        )
        .scalars()
        .all()
    )
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_str)
    weekly_blocks = _expand_weekly_rules(rules, window_start, window_end, tz)
    off_blocks = _collect_time_off(offs, window_start, window_end)
    appt_blocks = [(a.start_utc, a.end_utc, a) for a in appts]
    special_blocks = [
        (_as_aware_utc(s.start_utc), _as_aware_utc(s.end_utc), s) for s in specials
    ]

    owner = db.query(User).filter(User.id == user.sub).first()
    appt_edge_buffer_min = (
        int(getattr(owner, "appt_edge_buffer_min", 5) or 5) if pad_edges else 0
    )

    final_openings, final_off, _appt_spans = _apply_precedence(
        weekly_blocks,
        special_blocks,
        off_blocks,
        appt_blocks,
        appt_edge_buffer_min=appt_edge_buffer_min,
    )

    def _mk_event(
        ev_id: str,
        typ: str,
        title: str,
        s: datetime,
        e: datetime,
        status: Optional[str] = None,
        meta: Optional[dict] = None,
    ):
        return dict(
            id=ev_id,
            type=typ,
            title=title,
            start=s,
            end=e,
            status=status,
            meta=meta or {},
        )

    events: list[dict] = []
    for i, (s, e) in enumerate(final_openings):
        events.append(_mk_event(f"open-{i}", "opening", "Opening", s, e))
    for i, (s, e) in enumerate(final_off):
        events.append(_mk_event(f"off-{i}", "time_off", "Time Off", s, e))
    for a in appts:
        events.append(
            _mk_event(
                f"appt-{a.id}",
                "appointment",
                f"Appt {getattr(a, 'id', '')}",
                a.start_utc,
                a.end_utc,
                status=a.status,
            )
        )

    base["events"] = events
    return SnapshotOut(**base)


# --- Owner Holidays (for UI overlays) ----------------------------------------
class OwnerHolidayOut(BaseModel):
    """Owner-localized holiday overlay event for calendar display."""

    date: str
    name: str
    start_utc: datetime
    end_utc: datetime


@router.get("/owner/holidays", response_model=list[OwnerHolidayOut])
def owner_holidays(
    start: date = Query(..., description="Start date (YYYY-MM-DD) in owner-local"),
    end: date = Query(
        ..., description="End date (YYYY-MM-DD) in owner-local (inclusive)"
    ),
    country_code: str | None = Query(None),
    region_code: str | None = Query(None),
    tz: str | None = None,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> list[OwnerHolidayOut]:
    """Return holidays in the given owner-local date range, expanded to full-day UTC spans."""
    tz_str = tz or getattr(user, "timezone", None) or "America/Toronto"
    from zoneinfo import ZoneInfo

    tzinfo = ZoneInfo(tz_str)
    try:
        from agent.tools_holidays import (
            _fetch_holidays,
            DEFAULT_COUNTRY,
            DEFAULT_REGION,
        )
    except Exception:
        raise HTTPException(500, "Holiday module not available")

    cc = (country_code or DEFAULT_COUNTRY).upper()
    rc = (
        (region_code or DEFAULT_REGION).upper()
        if (region_code or DEFAULT_REGION)
        else None
    )
    years: set[int] = set()
    cur = start
    while cur <= end:
        years.add(cur.year)
        cur = cur + timedelta(days=1)
    rows: list[dict] = []
    for y in sorted(years):
        rows.extend(_fetch_holidays(y, cc))
    out: list[OwnerHolidayOut] = []
    for h in rows:
        dstr = h.get("date")
        if not isinstance(dstr, str):
            continue
        try:
            d = datetime.strptime(dstr, "%Y-%m-%d").date()
        except Exception:
            continue
        if d < start or d > end:
            continue
        counties = h.get("counties")
        applies = (not counties) or (rc and rc in counties)
        if not applies:
            continue
        s_local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tzinfo)
        e_local = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=tzinfo)
        out.append(
            OwnerHolidayOut(
                date=dstr,
                name=str(h.get("localName") or h.get("name") or "Holiday"),
                start_utc=s_local.astimezone(UTC),
                end_utc=e_local.astimezone(UTC),
            )
        )
    out.sort(key=lambda x: x.date)
    return out


class AdminCreateAppt(BaseModel):
    """Payload to create a one-off appointment as the owner/admin.

    Time values are provided in the owner's local timezone (naive or aware).
    When 'allow_override' is false, conflicts will require explicit confirm.
    """

    client_name: str
    client_email: EmailStr
    start_local: datetime
    duration_minutes: int = 30
    status: Literal["booked", "completed", "canceled"] = "booked"
    allow_override: bool = False
    message: Optional[str] = None
    confirm_if_conflicts: bool = False
    lesson_person_id: Optional[int] = None
    lesson_person_name: Optional[str] = None


class AdminCreateRecurringAppts(BaseModel):
    """Payload to create a recurring series of appointments as the owner.

    Either 'occurrences' or 'until_date' must be supplied. Recurrence is
    weekly, spaced by 'repeat_every_weeks'.
    """

    client_name: str
    client_email: EmailStr
    start_local: datetime
    duration_minutes: int = 30
    status: Literal["booked", "completed", "canceled"] = "booked"
    repeat_every_weeks: int = 1
    occurrences: Optional[int] = None
    until_date: Optional[date] = None
    allow_override: bool = False
    confirm_if_conflicts: bool = False
    message: Optional[str] = None
    lesson_person_id: Optional[int] = None
    lesson_person_name: Optional[str] = None

    @model_validator(mode="after")
    def _validate_recurring(self):
        """Enforce valid recurrence parameters for series creation."""
        if self.repeat_every_weeks <= 0:
            raise ValueError("repeat_every_weeks must be >= 1")
        if self.occurrences is None and self.until_date is None:
            raise ValueError(
                "Provide occurrences or until_date for recurring appointments"
            )
        if self.occurrences is not None and self.occurrences <= 0:
            raise ValueError("occurrences must be >= 1")
        if self.occurrences is not None and self.occurrences > 104:
            raise ValueError("occurrences must be <= 104")
        start_date = self.start_local.date()
        if self.until_date is not None and self.until_date < start_date:
            raise ValueError("until_date must be on or after start_local date")
        return self


@router.post("/appointments/admin-create", response_model=dict)
def admin_create_appointment(
    payload: AdminCreateAppt,
    background_tasks: BackgroundTasks,
    request: Request = None,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Create a new appointment for a client, optionally emailing a confirmation."""
    rid = getattr(getattr(request, "state", None), "request_id", None)
    _log_evt(
        "booking_attempt",
        rid=rid,
        owner_id=str(user.sub),
        kind="single",
        client_email=_mask_email(payload.client_email),
    )
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")

    # Compute requested window in owner-local and UTC for conflict checks
    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    end_local = start_local + timedelta(minutes=payload.duration_minutes)
    start_utc, end_utc = start_local.astimezone(UTC), end_local.astimezone(UTC)

    # If not overriding, prevent overlap with existing appointments/timeoff
    if not payload.allow_override:
        conflict_appts, conflict_offs = _collect_conflicts(
            db, owner.id, start_utc, end_utc
        )

        if (conflict_appts or conflict_offs) and not payload.confirm_if_conflicts:
            # Build a machine-parsable confirm payload (so graph.py can replay the same POST with confirm)
            conflicts_human = _format_conflicts(owner, conflict_appts, conflict_offs)

            payload_to_replay = {
                "endpoint": "/api/scheduling/appointments/admin-create",
                "method": "POST",
                "body": {
                    "client_name": payload.client_name,
                    "client_email": payload.client_email,
                    "start_local": start_local.isoformat(timespec="minutes"),
                    "duration_minutes": int(payload.duration_minutes),
                    "status": payload.status,
                    "allow_override": False,
                    "message": payload.message,
                    "confirm_if_conflicts": True,
                },
            }
            _log_evt(
                "booking_conflict",
                rid=rid,
                owner_id=str(user.sub),
                kind="single",
                conflicts=len(conflicts_human),
            )
            raise HTTPException(
                status_code=409,
                detail="CONFIRM_REQUIRED:"
                + json.dumps(
                    {
                        "human": "Booking conflicts detected. Reply 'confirm' to proceed anyway, or adjust the time.",
                        "pending_http": payload_to_replay,
                        "conflicts": conflicts_human,
                    }
                ),
            )

    # Ensure client user exists and has a relationship with this owner
    client = (
        db.query(User)
        .filter(User.email == payload.client_email, User.role == RoleEnum.CLIENT)
        .first()
    )
    if not client:
        raise HTTPException(
            400,
            f"Client with email '{payload.client_email}' not found. Please add the client first in the Clients section.",
        )

    # Check if client has a relationship with this owner
    from app.models import ClientAccount

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
        raise HTTPException(
            400,
            f"Client '{payload.client_email}' is not associated with your account. Please add them as a client first.",
        )

    # Block exact-start duplicates for ACTIVE appts only (allow reusing canceled starts)
    dup_same_start = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.start_utc == start_utc,
            Appointment.status != "canceled",
        )
        .first()
    )
    if dup_same_start:
        raise HTTPException(
            409, "Another appointment already exists at that exact start time."
        )
    appt = Appointment(
        id=uuid_str(),
        owner_id=owner.id,
        client_id=client.id,
        client_name=payload.client_name or client.name or client.email,
        client_email=client.email,
        start_utc=start_utc,
        end_utc=end_utc,
        status=payload.status,
    )
    db.add(appt)
    try:
        if payload.lesson_person_id is not None or (
            payload.lesson_person_name and payload.lesson_person_name.strip()
        ):
            pid = payload.lesson_person_id
            person = None
            from app.models import Person

            if pid is not None:
                person = (
                    db.query(Person)
                    .filter(
                        Person.id == int(pid), Person.account_id == client_account.id
                    )
                    .first()
                )
            if person is None and payload.lesson_person_name:
                nm = payload.lesson_person_name.strip()
                if nm:
                    person = (
                        db.query(Person)
                        .filter(
                            Person.account_id == client_account.id,
                            func.lower(Person.full_name) == func.lower(nm),
                        )
                        .first()
                    )
                    if person is None:
                        person = Person(
                            account_id=client_account.id, full_name=nm, email=None
                        )
                        db.add(person)
                        db.flush()
            if person is not None:
                appt.person_id = person.id
    except Exception:
        pass
    duration_minutes = int((end_utc - start_utc).total_seconds() // 60)
    appt.price_override_cents = get_default_price_cents(
        db, owner_user_id=owner.id, duration_minutes=duration_minutes
    )
    appt.payment_status = "unpaid"
    appt.amount_paid_cents = appt.amount_paid_cents or 0

    try:
        db.commit()
        db.refresh(appt)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            409, "Another appointment already exists at that exact start time."
        )

    # Best-effort: auto-apply wallet funds after booking, if enabled for this owner
    try:
        if get_owner_flag(
            user.sub,
            "auto_apply_wallet_on_book",
            "FEATURE_AUTO_APPLY_WALLET_ON_BOOK",
            default=True,
        ):
            wallet_rows = (
                db.query(PrepaidBundle.id)
                .filter(
                    PrepaidBundle.owner_id == owner.id,
                    PrepaidBundle.client_id == str(client.id),
                    PrepaidBundle.total_credits == 0,
                )
                .all()
            )
            for (wid,) in wallet_rows:
                _log_evt(
                    "wallet_auto_apply_attempt",
                    rid=rid,
                    owner_id=str(user.sub),
                    appt_id=str(appt.id),
                    bundle_id=int(wid),
                )
                auto_apply_wallet_funds(
                    db,
                    owner_id=str(owner.id),
                    bundle_id=int(wid),
                    note_prefix="Auto-apply wallet funds after booking",
                )
            try:
                db.refresh(appt)
            except Exception:
                pass
    except Exception:
        pass

    if client.email:
        email_pkg = build_appt_email(
            audience="client",
            action="created",
            owner=owner,
            start_local=start_local,
            end_local=end_local,
            appointment_id=str(appt.id),
            initiator_label=owner.name or "the owner",
            status_label=appt.status,
            recipient_name=client.name or client.email,
            message=payload.message,
            include_ics=True,
            organizer_email=owner.email,
            attendee_email=client.email,
        )

        background_tasks.add_task(
            send_email,
            client.email,
            email_pkg.subject,
            email_pkg.text,
            email_pkg.html,
            email_pkg.ics_text,
        )

    try:
        _log_evt(
            "booking_ok",
            rid=rid,
            owner_id=str(user.sub),
            kind="single",
            appt_id=str(appt.id),
            start_utc=appt.start_utc.isoformat(),
            end_utc=appt.end_utc.isoformat(),
        )
    except Exception:
        pass
    return {"ok": True, "appointment_id": appt.id, "status": appt.status}


@router.post(
    "/appointments/admin-create/recurring", response_model=dict, status_code=201
)
def admin_create_recurring_appointments(
    payload: AdminCreateRecurringAppts,
    background_tasks: BackgroundTasks,
    request: Request = None,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Create a recurring series of appointments, with best-effort wallet application and emails."""
    rid = getattr(getattr(request, "state", None), "request_id", None)
    _log_evt(
        "booking_attempt",
        rid=rid,
        owner_id=str(user.sub),
        kind="recurring",
        client_email=_mask_email(payload.client_email),
    )
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")

    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    if payload.duration_minutes <= 0:
        raise HTTPException(400, "duration_minutes must be positive")

    step = timedelta(weeks=int(payload.repeat_every_weeks))
    duration_delta = timedelta(minutes=int(payload.duration_minutes))

    occurrences: list[datetime] = []
    current = start_local
    limit_date = payload.until_date
    count = 0
    while True:
        if limit_date is not None and current.date() > limit_date:
            break
        occurrences.append(current)
        count += 1
        if payload.occurrences is not None and count >= payload.occurrences:
            break
        if count >= 104:
            break
        current = current + step
        if limit_date is None and payload.occurrences is None:
            break

    if not occurrences:
        raise HTTPException(
            400, "No occurrences computed for the provided recurrence rule."
        )

    windows = []
    conflicts_summary: list[dict[str, Any]] = []
    for occ_start_local in occurrences:
        occ_end_local = occ_start_local + duration_delta
        occ_start_utc = occ_start_local.astimezone(UTC)
        occ_end_utc = occ_end_local.astimezone(UTC)
        windows.append((occ_start_local, occ_end_local, occ_start_utc, occ_end_utc))

        if not payload.allow_override:
            conflict_appts, conflict_offs = _collect_conflicts(
                db, owner.id, occ_start_utc, occ_end_utc
            )
            if (conflict_appts or conflict_offs) and not payload.confirm_if_conflicts:
                conflicts_human = _format_conflicts(
                    owner, conflict_appts, conflict_offs
                )
                conflicts_summary.append(
                    {
                        "start_local": occ_start_local.isoformat(timespec="minutes"),
                        "conflicts": conflicts_human,
                    }
                )

    if conflicts_summary and not payload.confirm_if_conflicts:
        _log_evt(
            "booking_conflict",
            rid=rid,
            owner_id=str(user.sub),
            kind="recurring",
            conflicts=len(conflicts_summary),
        )
        pending_body = payload.model_dump(mode="json")
        pending_body["start_local"] = start_local.isoformat(timespec="minutes")
        pending_body["confirm_if_conflicts"] = True
        detail = {
            "human": "Booking conflicts detected for one or more occurrences. Reply 'confirm' to proceed anyway, or adjust the time.",
            "pending_http": {
                "endpoint": "/api/scheduling/appointments/admin-create/recurring",
                "method": "POST",
                "body": pending_body,
            },
            "conflicts": conflicts_summary,
        }
        raise HTTPException(
            status_code=409, detail="CONFIRM_REQUIRED:" + json.dumps(detail)
        )

    # Only resolve client identity after conflict handling, so that the caller
    # can receive a uniform 'CONFIRM_REQUIRED' payload before being blocked on
    # client association checks.
    client = (
        db.query(User)
        .filter(User.email == payload.client_email, User.role == RoleEnum.CLIENT)
        .first()
    )
    if not client:
        raise HTTPException(
            400,
            f"Client with email '{payload.client_email}' not found. Please add the client first in the Clients section.",
        )

    from app.models import ClientAccount

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
        raise HTTPException(
            400,
            f"Client '{payload.client_email}' is not associated with your account. Please add them as a client first.",
        )

    created: list[tuple[Appointment, datetime, datetime]] = []
    seen_starts: set[datetime] = set()
    for occ_start_local, occ_end_local, occ_start_utc, occ_end_utc in windows:
        if occ_start_utc in seen_starts:
            raise HTTPException(
                400, "Duplicate occurrence start times detected in the request."
            )
        seen_starts.add(occ_start_utc)

        dup_same_start = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.start_utc == occ_start_utc,
                Appointment.status != "canceled",
            )
            .first()
        )
        if dup_same_start:
            raise HTTPException(
                409,
                "Another appointment already exists at one of the requested start times.",
            )

        appt = Appointment(
            id=uuid_str(),
            owner_id=owner.id,
            client_id=client.id,
            client_name=payload.client_name or client.name or client.email,
            client_email=client.email,
            start_utc=occ_start_utc,
            end_utc=occ_end_utc,
            status=payload.status,
        )
        try:
            with db.begin_nested():
                db.add(appt)
                try:
                    if payload.lesson_person_id is not None or (
                        payload.lesson_person_name
                        and payload.lesson_person_name.strip()
                    ):
                        pid = payload.lesson_person_id
                        person = None
                        from app.models import Person

                        if pid is not None:
                            person = (
                                db.query(Person)
                                .filter(
                                    Person.id == int(pid),
                                    Person.account_id == client_account.id,
                                )
                                .first()
                            )
                        if person is None and payload.lesson_person_name:
                            nm = payload.lesson_person_name.strip()
                            if nm:
                                person = (
                                    db.query(Person)
                                    .filter(
                                        Person.account_id == client_account.id,
                                        func.lower(Person.full_name) == func.lower(nm),
                                    )
                                    .first()
                                )
                                if person is None:
                                    person = Person(
                                        account_id=client_account.id,
                                        full_name=nm,
                                        email=None,
                                    )
                                    db.add(person)
                                    db.flush()
                        if person is not None:
                            appt.person_id = person.id
                except Exception:
                    pass

                duration_minutes = int(
                    (occ_end_utc - occ_start_utc).total_seconds() // 60
                )
                appt.price_override_cents = get_default_price_cents(
                    db, owner_user_id=owner.id, duration_minutes=duration_minutes
                )
                appt.payment_status = "unpaid"
                appt.amount_paid_cents = appt.amount_paid_cents or 0

                db.flush()
                created.append((appt, occ_start_local, occ_end_local))
        except IntegrityError:
            continue
    db.commit()

    response_rows: list[dict[str, Any]] = []
    for appt, occ_start_local, occ_end_local in created:
        try:
            db.refresh(appt)
        except Exception:
            pass

        if client.email:
            email_pkg = build_appt_email(
                audience="client",
                action="created",
                owner=owner,
                start_local=occ_start_local,
                end_local=occ_end_local,
                appointment_id=str(appt.id),
                initiator_label=owner.name or "the owner",
                status_label=appt.status,
                recipient_name=client.name or client.email,
                message=payload.message,
                include_ics=True,
                organizer_email=owner.email,
                attendee_email=client.email,
            )

            background_tasks.add_task(
                send_email,
                client.email,
                email_pkg.subject,
                email_pkg.text,
                email_pkg.html,
                email_pkg.ics_text,
            )

        response_rows.append(
            {
                "appointment_id": str(appt.id),
                "status": appt.status,
                "start_local": occ_start_local.isoformat(timespec="minutes"),
            }
        )
    try:
        if client:
            wallet_ids = (
                db.query(PrepaidBundle.id)
                .filter(
                    PrepaidBundle.owner_id == owner.id,
                    PrepaidBundle.client_id == str(client.id),
                    PrepaidBundle.total_credits == 0,
                )
                .order_by(PrepaidBundle.created_at.desc())
                .all()
            )
            for (wid,) in wallet_ids:
                _log_evt(
                    "wallet_auto_apply_attempt",
                    rid=rid,
                    owner_id=str(user.sub),
                    bundle_id=int(wid),
                )
                auto_apply_wallet_funds(
                    db,
                    owner_id=str(owner.id),
                    bundle_id=int(wid),
                    note_prefix="Auto-apply wallet funds after recurring booking",
                )
    except Exception:
        pass
    try:
        price_map = _service_price_map(db, owner_user_id=owner.id)
        if client:
            wallet = (
                db.query(PrepaidBundle)
                .filter(
                    PrepaidBundle.owner_id == owner.id,
                    PrepaidBundle.client_id == str(client.id),
                    PrepaidBundle.total_credits == 0,
                )
                .order_by(PrepaidBundle.created_at.desc())
                .first()
            )
        else:
            wallet = None
        if wallet:
            for appt, _s, _e in created:
                a = (
                    db.query(Appointment)
                    .filter(Appointment.id == appt.id, Appointment.owner_id == owner.id)
                    .first()
                )
                expected = compute_price_cents(db, a, price_map) or 0
                if int(expected) <= 0:
                    continue
                cash_paid = int(getattr(a, "amount_paid_cents", 0) or 0)
                ledger_total = (
                    db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                    .filter(PrepaidLedger.appointment_id == a.id)
                    .scalar()
                ) or 0
                already_applied = -int(ledger_total) if int(ledger_total) < 0 else 0
                owed = max(int(expected) - cash_paid - already_applied, 0)
                if owed <= 0:
                    continue
                balance = int(
                    db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                    .filter(PrepaidLedger.bundle_id == wallet.id)
                    .scalar()
                    or 0
                )
                use = min(balance, owed)
                if use > 0:
                    _log_evt(
                        "wallet_direct_consume",
                        rid=rid,
                        owner_id=str(user.sub),
                        appt_id=str(a.id),
                        bundle_id=int(wallet.id),
                        used=int(use),
                        owed=int(owed),
                    )
                    a.bundle_id = wallet.id
                    db.add(
                        PrepaidLedger(
                            bundle_id=wallet.id,
                            event="consume",
                            delta_credits=0,
                            amount_cents=-int(use),
                            appointment_id=a.id,
                            note="Auto-apply wallet on recurring booking (direct)",
                        )
                    )
                    if (cash_paid + already_applied + int(use)) >= int(expected):
                        a.payment_status = "paid"
                        if getattr(a, "paid_at", None) is None:
                            a.paid_at = datetime.now(_tz.utc)
                    db.add(a)
            db.commit()
    except Exception:
        pass

    return {"ok": True, "count": len(response_rows), "appointments": response_rows}


class OwnerSettingsOut(BaseModel):
    """Owner settings exposed to the UI.

    - 'appt_edge_buffer_min': Minutes to visually pad appointment edges.
    - 'auto_apply_wallet_on_book': Auto-apply wallet after booking.
    - 'wallet_deposits_as_paid': Treat wallet deposits as paid for reporting.
    - 'group_price_60_cents': Optional fixed price for 60-minute group lessons.
    """

    appt_edge_buffer_min: int
    auto_apply_wallet_on_book: bool = False
    wallet_deposits_as_paid: bool = False
    group_price_60_cents: int | None = None


class OwnerSettingsIn(BaseModel):
    """Incoming owner settings payload.

    Only non-null feature flags are updated; others remain unchanged.
    """

    appt_edge_buffer_min: int = Field(ge=0, le=120)
    auto_apply_wallet_on_book: bool | None = None
    wallet_deposits_as_paid: bool | None = None
    group_price_60_cents: int | None = Field(default=None, ge=0)
    model_config = ConfigDict(extra="forbid")


@router.get("/owner/settings", response_model=OwnerSettingsOut)
def get_owner_settings(
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> OwnerSettingsOut:
    owner = db.query(User).filter(User.id == user.sub).first()
    v = int(getattr(owner, "appt_edge_buffer_min", 5) or 5)
    from services.features import get_owner_flag

    return OwnerSettingsOut(
        appt_edge_buffer_min=v,
        auto_apply_wallet_on_book=get_owner_flag(
            user.sub,
            "auto_apply_wallet_on_book",
            "FEATURE_AUTO_APPLY_WALLET_ON_BOOK",
            default=True,
        ),
        wallet_deposits_as_paid=get_owner_flag(
            user.sub,
            "wallet_deposits_as_paid",
            "FEATURE_WALLET_DEPOSITS_AS_PAID",
            default=False,
        ),
        group_price_60_cents=getattr(owner, "group_price_60_cents", None),
    )


@router.put("/owner/settings", response_model=OwnerSettingsOut)
def update_owner_settings(
    body: OwnerSettingsIn,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> OwnerSettingsOut:
    owner = db.query(User).filter(User.id == user.sub).first()
    owner.appt_edge_buffer_min = int(body.appt_edge_buffer_min)
    if body.group_price_60_cents is not None:
        owner.group_price_60_cents = int(body.group_price_60_cents)
    db.commit()
    db.refresh(owner)
    from services.features import set_owner_flags, get_owner_flag

    flags = {}
    if body.auto_apply_wallet_on_book is not None:
        flags["auto_apply_wallet_on_book"] = bool(body.auto_apply_wallet_on_book)
    if body.wallet_deposits_as_paid is not None:
        flags["wallet_deposits_as_paid"] = bool(body.wallet_deposits_as_paid)
    if flags:
        set_owner_flags(user.sub, **flags)
    return OwnerSettingsOut(
        appt_edge_buffer_min=owner.appt_edge_buffer_min,
        auto_apply_wallet_on_book=get_owner_flag(
            user.sub,
            "auto_apply_wallet_on_book",
            "FEATURE_AUTO_APPLY_WALLET_ON_BOOK",
            default=True,
        ),
        wallet_deposits_as_paid=get_owner_flag(
            user.sub,
            "wallet_deposits_as_paid",
            "FEATURE_WALLET_DEPOSITS_AS_PAID",
            default=False,
        ),
        group_price_60_cents=getattr(owner, "group_price_60_cents", None),
    )


# ---------------------- Group Lesson (owner-only) ----------------------
class GroupCreatePayload(BaseModel):
    """Create a group lesson at a given time with multiple attendees by 'person_id'."""

    start_local: datetime
    duration_minutes: int
    person_ids: list[int]
    status: Literal["booked", "completed", "canceled"] = "booked"
    allow_override: bool = False
    confirm_if_conflicts: bool = False
    message: str | None = None


@router.post("/appointments/admin-create-group", response_model=dict)
def admin_create_group(
    payload: GroupCreatePayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Create or attach a group lesson for the given attendees.

    Reuses existing single seats at the same time when present, otherwise
    creates new seats and groups them under a shared 'group_id'.
    """
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    if not payload.person_ids:
        raise HTTPException(400, "person_ids is required")
    if payload.duration_minutes <= 0:
        raise HTTPException(400, "duration_minutes must be positive")

    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    end_local = start_local + timedelta(minutes=int(payload.duration_minutes))
    start_utc, end_utc = start_local.astimezone(UTC), end_local.astimezone(UTC)

    if not payload.allow_override:
        conflict_appts, conflict_offs = _collect_conflicts(
            db, owner.id, start_utc, end_utc
        )
        if (conflict_appts or conflict_offs) and not payload.confirm_if_conflicts:
            detail = {
                "human": "Booking conflicts detected. Reply 'confirm' to proceed anyway, or adjust the time.",
                "pending_http": {
                    "endpoint": "/api/scheduling/appointments/admin-create-group",
                    "method": "POST",
                    "body": {
                        **payload.model_dump(mode="json"),
                        "start_local": start_local.isoformat(timespec="minutes"),
                        "confirm_if_conflicts": True,
                    },
                },
                "conflicts": _format_conflicts(owner, conflict_appts, conflict_offs),
            }
            raise HTTPException(409, detail="CONFIRM_REQUIRED:" + json.dumps(detail))
    requested_ids: set[int] = {int(pid) for pid in payload.person_ids}
    try:
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
        )

        valid_person_rows = (
            db.query(PersonModel.id)
            .join(ClientAccountModel, ClientAccountModel.id == PersonModel.account_id)
            .filter(
                PersonModel.id.in_(requested_ids),
                ClientAccountModel.owner_user_id == owner.id,
                ClientAccountModel.deleted_at.is_(None),
            )
            .all()
        )
        valid_ids = {int(pid) for (pid,) in valid_person_rows}
        invalid_ids = sorted(list(requested_ids - valid_ids))
        if invalid_ids:
            raise HTTPException(
                400,
                detail={
                    "error": "INVALID_PERSON_IDS",
                    "message": "Some attendees are not found or not associated with your account.",
                    "invalid_person_ids": invalid_ids,
                },
            )
    except HTTPException:
        raise
    except Exception:
        pass
    existing = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.start_utc == start_utc,
            Appointment.end_utc == end_utc,
            Appointment.status != "canceled",
            Appointment.person_id.in_(requested_ids),
        )
        .all()
    )

    # Determine group id: if any existing already in a group, reuse it; else create new
    group_uuid = None
    for r in existing:
        if getattr(r, "group_id", None):
            group_uuid = r.group_id
            break
    if group_uuid is None:
        group_uuid = uuid.UUID(uuid_str())

    attached_count = 0
    appt_ids: list[str] = []
    success_pids: set[int] = set()
    failed_reasons: dict[int, str] = {}

    # Attach existing appointments to the group and adjust price if needed
    for r in existing:
        if r.group_id != group_uuid:
            r.group_id = group_uuid
        gp = getattr(owner, "group_price_60_cents", None)
        if int(payload.duration_minutes) == 60 and gp is not None and int(gp) > 0:
            r.price_override_cents = int(gp)
        else:
            r.price_override_cents = get_default_price_cents(
                db,
                owner_user_id=owner.id,
                duration_minutes=int(payload.duration_minutes),
            )
        r.status = payload.status
        r.payment_status = "unpaid"
        r.amount_paid_cents = r.amount_paid_cents or 0
        db.add(r)
        appt_ids.append(str(r.id))
        attached_count += 1
        if getattr(r, "person_id", None) is not None:
            success_pids.add(int(r.person_id))
    created = 0
    for pid in requested_ids - {
        int(getattr(r, "person_id"))
        for r in existing
        if getattr(r, "person_id", None) is not None
    }:
        a = Appointment(
            id=uuid.uuid4(),
            owner_id=owner.id,
            person_id=int(pid),
            start_utc=start_utc,
            end_utc=end_utc,
            status=payload.status,
            group_id=group_uuid,
        )
        # Attach client linkage/identity for better UI visibility and analytics.
        try:
            from app.models import (
                Person as PersonModel,
                ClientAccount as ClientAccountModel,
            )

            person_row = (
                db.query(PersonModel).filter(PersonModel.id == int(pid)).first()
            )
            acct_row = None
            user_row = None
            if person_row:
                acct_row = (
                    db.query(ClientAccountModel)
                    .filter(
                        ClientAccountModel.id == int(person_row.account_id),
                        ClientAccountModel.owner_user_id == owner.id,
                        ClientAccountModel.deleted_at.is_(None),
                    )
                    .first()
                )
            if acct_row and getattr(acct_row, "client_user_id", None):
                user_row = (
                    db.query(User).filter(User.id == acct_row.client_user_id).first()
                )
            # Set client_id if we found the linked auth user
            if user_row:
                a.client_id = user_row.id
                a.client_name = getattr(user_row, "name", None) or getattr(
                    acct_row, "name", None
                )
                a.client_email = getattr(user_row, "email", None) or getattr(
                    person_row, "email", None
                )
            else:
                # Fall back to denormalized person/account identity when no auth linkage
                a.client_name = getattr(person_row, "full_name", None) or getattr(
                    acct_row, "name", None
                )
                a.client_email = getattr(person_row, "email", None)
        except Exception:
            pass
        # Use group price for 60m if configured, otherwise default price
        gp = getattr(owner, "group_price_60_cents", None)
        if int(payload.duration_minutes) == 60 and gp is not None and int(gp) > 0:
            a.price_override_cents = int(gp)
        else:
            a.price_override_cents = get_default_price_cents(
                db,
                owner_user_id=owner.id,
                duration_minutes=int(payload.duration_minutes),
            )
        a.payment_status = "unpaid"
        a.amount_paid_cents = a.amount_paid_cents or 0
        db.add(a)
        try:
            db.flush()
            created += 1
            appt_ids.append(str(a.id))
            success_pids.add(int(pid))
        except IntegrityError as ie:
            db.rollback()
            exist_row = (
                db.query(Appointment)
                .filter(
                    Appointment.owner_id == owner.id,
                    Appointment.start_utc == start_utc,
                    Appointment.status != "canceled",
                    Appointment.person_id == int(pid),
                )
                .first()
            )
            if exist_row:
                if exist_row.group_id != group_uuid:
                    exist_row.group_id = group_uuid
                exist_row.end_utc = end_utc
                gp2 = getattr(owner, "group_price_60_cents", None)
                if (
                    int(payload.duration_minutes) == 60
                    and gp2 is not None
                    and int(gp2) > 0
                ):
                    exist_row.price_override_cents = int(gp2)
                else:
                    exist_row.price_override_cents = get_default_price_cents(
                        db,
                        owner_user_id=owner.id,
                        duration_minutes=int(payload.duration_minutes),
                    )
                exist_row.status = payload.status
                exist_row.payment_status = "unpaid"
                exist_row.amount_paid_cents = exist_row.amount_paid_cents or 0
                db.add(exist_row)
                try:
                    db.flush()
                    attached_count += 1
                    appt_ids.append(str(exist_row.id))
                    if getattr(exist_row, "person_id", None) is not None:
                        success_pids.add(int(exist_row.person_id))
                except Exception as e:
                    db.rollback()
                    failed_reasons[int(pid)] = f"attach_failed: {getattr(e, 'orig', e)}"
                    continue
            else:
                failed_reasons[int(pid)] = f"insert_failed: {getattr(ie, 'orig', ie)}"
            continue
        except Exception as e:
            db.rollback()
            failed_reasons[int(pid)] = f"insert_failed_other: {e}"
            continue
    db.commit()

    # Best-effort: auto-apply wallet funds for all attendees' wallets
    try:
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
        )

        persons = (
            db.query(PersonModel).filter(PersonModel.id.in_(payload.person_ids)).all()
        )
        acct_ids = {
            int(p.account_id) for p in persons if getattr(p, "account_id", None)
        }
        if acct_ids:
            accounts = (
                db.query(ClientAccountModel)
                .filter(
                    ClientAccountModel.id.in_(acct_ids),
                    ClientAccountModel.owner_user_id == owner.id,
                    ClientAccountModel.deleted_at.is_(None),
                )
                .all()
            )
            client_user_ids = {
                str(a.client_user_id)
                for a in accounts
                if getattr(a, "client_user_id", None)
            }
            if client_user_ids:
                wallet_rows = (
                    db.query(PrepaidBundle.id)
                    .filter(
                        PrepaidBundle.owner_id == owner.id,
                        PrepaidBundle.client_id.in_(client_user_ids),
                        PrepaidBundle.total_credits == 0,
                    )
                    .order_by(PrepaidBundle.created_at.desc())
                    .all()
                )
                for (wid,) in wallet_rows:
                    auto_apply_wallet_funds(
                        db,
                        owner_id=str(owner.id),
                        bundle_id=int(wid),
                        note_prefix="Auto-apply wallet funds after group booking",
                    )
    except Exception:
        pass

    # If the DB still has the legacy unique index on (owner_id, start_utc),
    # only one seat can exist at a time. If we couldn't add everyone, surface
    # a clear error so the owner can run the Alembic migration to enable groups.
    requested_total = len(requested_ids)
    total_added = attached_count + created
    if total_added < requested_total:
        missing_ids = sorted(list(requested_ids - success_pids))
        raise HTTPException(
            409,
            detail={
                "error": "PARTIAL_GROUP_CREATE",
                "message": (
                    "Could not add all attendees. Possible causes: (1) database still has a legacy "
                    "unique index on (owner_id, start_utc), or (2) one or more person_ids are invalid/"
                    "not associated with your account."
                ),
                "hint": "Ensure only ix_owner_start_person_active_unique exists and that all person_ids belong to your clients.",
                "added_person_ids": sorted(list(success_pids)),
                "missing_person_ids": missing_ids,
                "failed_reasons": {str(k): v for k, v in failed_reasons.items()},
            },
        )

    # Send confirmation emails to unique accounts (one per account)
    try:
        recips = _collect_unique_account_recipients_for_person_ids(
            db, owner.id, list(success_pids)
        )
        if recips:
            for to_email, to_name in recips:
                email_pkg = build_appt_email(
                    audience="client",
                    action="created",
                    owner=owner,
                    start_local=start_local,
                    end_local=end_local,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label=payload.status,
                    recipient_name=to_name or to_email,
                    message=payload.message,
                    include_ics=True,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    email_pkg.subject,
                    email_pkg.text,
                    email_pkg.html,
                    email_pkg.ics_text,
                )
    except Exception:
        pass
    return {
        "ok": True,
        "group_id": str(group_uuid),
        "count": total_added,
        "appointment_ids": appt_ids,
    }


class GroupRecurringCreatePayload(BaseModel):
    """Create a recurring series of group lessons for a fixed attendee set."""

    start_local: datetime
    duration_minutes: int
    repeat_every_weeks: int = 1
    occurrences: int | None = None
    until_date: date | None = None
    person_ids: list[int]
    allow_override: bool = False
    confirm_if_conflicts: bool = False
    message: str | None = None


@router.post("/appointments/admin-create-group/recurring", response_model=dict)
def admin_create_group_recurring(
    payload: GroupRecurringCreatePayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Create multiple group lessons according to a weekly recurrence pattern."""
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    if not payload.person_ids:
        raise HTTPException(400, "person_ids is required")
    if payload.duration_minutes <= 0:
        raise HTTPException(400, "duration_minutes must be positive")

    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    step = timedelta(weeks=int(payload.repeat_every_weeks or 1))
    dur = timedelta(minutes=int(payload.duration_minutes))
    limit = payload.until_date
    occs: list[tuple[datetime, datetime, datetime, datetime]] = []
    cur = start_local
    count = 0
    while True:
        if limit is not None and cur.date() > limit:
            break
        end_loc = cur + dur
        occs.append((cur, end_loc, cur.astimezone(UTC), end_loc.astimezone(UTC)))
        count += 1
        if payload.occurrences is not None and count >= payload.occurrences:
            break
        if payload.occurrences is None and count >= 104:
            break
        cur = cur + step

    # Conflicts summary (without override)
    if not payload.allow_override and not payload.confirm_if_conflicts:
        conflicts: list[dict] = []
        for _s_loc, _e_loc, s_utc, e_utc in occs:
            a, t = _collect_conflicts(db, owner.id, s_utc, e_utc)
            if a or t:
                conflicts.append(
                    {
                        "start_local": _s_loc.isoformat(timespec="minutes"),
                        "conflicts": _format_conflicts(owner, a, t),
                    }
                )
        if conflicts:
            detail = {
                "human": "Booking conflicts detected for one or more occurrences. Reply 'confirm' to proceed anyway.",
                "pending_http": {
                    "endpoint": "/api/scheduling/appointments/admin-create-group/recurring",
                    "method": "POST",
                    "body": {
                        **payload.model_dump(mode="json"),
                        "start_local": start_local.isoformat(timespec="minutes"),
                        "confirm_if_conflicts": True,
                    },
                },
                "conflicts": conflicts,
            }
            raise HTTPException(409, detail="CONFIRM_REQUIRED:" + json.dumps(detail))

    created_groups: list[dict] = []
    try:
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
        )

        requested_ids: set[int] = {int(pid) for pid in payload.person_ids}
        valid_person_rows = (
            db.query(PersonModel.id)
            .join(ClientAccountModel, ClientAccountModel.id == PersonModel.account_id)
            .filter(
                PersonModel.id.in_(requested_ids),
                ClientAccountModel.owner_user_id == owner.id,
                ClientAccountModel.deleted_at.is_(None),
            )
            .all()
        )
        valid_ids = {int(pid) for (pid,) in valid_person_rows}
        invalid_ids = sorted(list(requested_ids - valid_ids))
        if invalid_ids:
            raise HTTPException(
                400,
                detail={
                    "error": "INVALID_PERSON_IDS",
                    "message": "Some attendees are not found or not associated with your account.",
                    "invalid_person_ids": invalid_ids,
                },
            )
    except HTTPException:
        raise
    except Exception:
        pass
    for s_loc, e_loc, s_utc, e_utc in occs:
        gid = uuid.uuid4()
        per_group = 0
        for pid in payload.person_ids:
            a = Appointment(
                id=uuid.uuid4(),
                owner_id=owner.id,
                person_id=int(pid),
                start_utc=s_utc,
                end_utc=e_utc,
                status="booked",
                group_id=gid,
            )
            try:
                from app.models import (
                    Person as PersonModel,
                    ClientAccount as ClientAccountModel,
                )

                person_row = (
                    db.query(PersonModel).filter(PersonModel.id == int(pid)).first()
                )
                acct_row = None
                user_row = None
                if person_row:
                    acct_row = (
                        db.query(ClientAccountModel)
                        .filter(
                            ClientAccountModel.id == int(person_row.account_id),
                            ClientAccountModel.owner_user_id == owner.id,
                            ClientAccountModel.deleted_at.is_(None),
                        )
                        .first()
                    )
                if acct_row and getattr(acct_row, "client_user_id", None):
                    user_row = (
                        db.query(User)
                        .filter(User.id == acct_row.client_user_id)
                        .first()
                    )
                if user_row:
                    a.client_id = user_row.id
                    a.client_name = getattr(user_row, "name", None) or getattr(
                        acct_row, "name", None
                    )
                    a.client_email = getattr(user_row, "email", None) or getattr(
                        person_row, "email", None
                    )
                else:
                    a.client_name = getattr(person_row, "full_name", None) or getattr(
                        acct_row, "name", None
                    )
                    a.client_email = getattr(person_row, "email", None)
            except Exception:
                pass
            gp = getattr(owner, "group_price_60_cents", None)
            if int(payload.duration_minutes) == 60 and gp is not None and int(gp) > 0:
                a.price_override_cents = int(gp)
            else:
                a.price_override_cents = get_default_price_cents(
                    db,
                    owner_user_id=owner.id,
                    duration_minutes=int(payload.duration_minutes),
                )
            a.payment_status = "unpaid"
            a.amount_paid_cents = a.amount_paid_cents or 0
            db.add(a)
            try:
                db.flush()
                per_group += 1
            except IntegrityError:
                db.rollback()
                exist_row = (
                    db.query(Appointment)
                    .filter(
                        Appointment.owner_id == owner.id,
                        Appointment.start_utc == s_utc,
                        Appointment.status != "canceled",
                        Appointment.person_id == int(pid),
                    )
                    .first()
                )
                if exist_row:
                    if exist_row.group_id != gid:
                        exist_row.group_id = gid
                    exist_row.end_utc = e_utc
                    gp2 = getattr(owner, "group_price_60_cents", None)
                    if (
                        int(payload.duration_minutes) == 60
                        and gp2 is not None
                        and int(gp2) > 0
                    ):
                        exist_row.price_override_cents = int(gp2)
                    else:
                        exist_row.price_override_cents = get_default_price_cents(
                            db,
                            owner_user_id=owner.id,
                            duration_minutes=int(payload.duration_minutes),
                        )
                    exist_row.status = "booked"
                    exist_row.payment_status = "unpaid"
                    exist_row.amount_paid_cents = exist_row.amount_paid_cents or 0
                    db.add(exist_row)
                    try:
                        db.flush()
                        per_group += 1
                    except Exception:
                        db.rollback()
                        continue
                continue
        if per_group > 0:
            created_groups.append(
                {
                    "group_id": str(gid),
                    "count": per_group,
                    "start_local": s_loc.isoformat(timespec="minutes"),
                }
            )
    db.commit()

    # Best-effort: auto-apply wallets for all attendees across all created groups
    try:
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
        )

        persons = (
            db.query(PersonModel).filter(PersonModel.id.in_(payload.person_ids)).all()
        )
        acct_ids = {
            int(p.account_id) for p in persons if getattr(p, "account_id", None)
        }
        if acct_ids:
            accounts = (
                db.query(ClientAccountModel)
                .filter(
                    ClientAccountModel.id.in_(acct_ids),
                    ClientAccountModel.owner_user_id == owner.id,
                    ClientAccountModel.deleted_at.is_(None),
                )
                .all()
            )
            client_user_ids = {
                str(a.client_user_id)
                for a in accounts
                if getattr(a, "client_user_id", None)
            }
            if client_user_ids:
                wallet_rows = (
                    db.query(PrepaidBundle.id)
                    .filter(
                        PrepaidBundle.owner_id == owner.id,
                        PrepaidBundle.client_id.in_(client_user_ids),
                        PrepaidBundle.total_credits == 0,
                    )
                    .order_by(PrepaidBundle.created_at.desc())
                    .all()
                )
                for (wid,) in wallet_rows:
                    auto_apply_wallet_funds(
                        db,
                        owner_id=str(owner.id),
                        bundle_id=int(wid),
                        note_prefix="Auto-apply wallet funds after group recurring booking",
                    )
    except Exception:
        pass
    try:
        recips = _collect_unique_account_recipients_for_person_ids(
            db, owner.id, [int(p) for p in payload.person_ids]
        )
        for g in created_groups:
            s_loc = (
                datetime.fromisoformat(g["start_local"])
                if isinstance(g.get("start_local"), str)
                else g["start_local"]
            )
            e_loc = s_loc + timedelta(minutes=int(payload.duration_minutes))
            for to_email, to_name in recips:
                email_pkg = build_appt_email(
                    audience="client",
                    action="created",
                    owner=owner,
                    start_local=s_loc,
                    end_local=e_loc,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label="booked",
                    recipient_name=to_name or to_email,
                    message=payload.message,
                    include_ics=True,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    email_pkg.subject,
                    email_pkg.text,
                    email_pkg.html,
                    email_pkg.ics_text,
                )
    except Exception:
        pass

    total_requested = len(payload.person_ids) * len(created_groups)
    total_created = sum(g["count"] for g in created_groups)
    if total_created < total_requested:
        raise HTTPException(
            409,
            detail={
                "error": "PARTIAL_GROUP_CREATE",
                "message": (
                    "Could not add all attendees across recurring occurrences. Possible causes: (1) database still has a legacy "
                    "unique index on (owner_id, start_utc), or (2) one or more person_ids are invalid/not associated with your account."
                ),
                "hint": "Ensure only ix_owner_start_person_active_unique exists and that all person_ids belong to your clients.",
            },
        )
    return {"ok": True, "count": total_created, "groups": created_groups}


class GroupModifyTimePayload(BaseModel):
    """Payload to change the time window for a group lesson."""

    start_local: datetime
    duration_minutes: int
    confirm_if_conflicts: bool = False


@router.put("/appointments/group/{group_id}/time", response_model=dict)
def admin_group_update_time(
    group_id: str,
    payload: GroupModifyTimePayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Reschedule the entire group to a new start and duration, emailing attendees."""
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    rows = (
        db.query(Appointment)
        .filter(Appointment.owner_id == user.sub, Appointment.group_id == gid)
        .all()
    )
    if not rows:
        raise HTTPException(404, "Group not found")

    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    end_local = start_local + timedelta(minutes=int(payload.duration_minutes))
    start_utc, end_utc = start_local.astimezone(UTC), end_local.astimezone(UTC)

    if not payload.confirm_if_conflicts:
        conflict_appts, conflict_offs = _collect_conflicts(
            db, owner.id, start_utc, end_utc
        )
        conflict_appts = [
            a for a in conflict_appts if getattr(a, "group_id", None) != gid
        ]
        if conflict_appts or conflict_offs:
            detail = {
                "human": "Booking conflicts detected. Reply 'confirm' to proceed anyway, or adjust the time.",
                "pending_http": {
                    "endpoint": f"/api/scheduling/appointments/group/{group_id}/time",
                    "method": "PUT",
                    "body": {
                        **payload.model_dump(mode="json"),
                        "confirm_if_conflicts": True,
                    },
                },
                "conflicts": _format_conflicts(owner, conflict_appts, conflict_offs),
            }
            raise HTTPException(409, detail="CONFIRM_REQUIRED:" + json.dumps(detail))
    old_start_local = rows[0].start_utc.astimezone(owner_tz)
    old_end_local = rows[0].end_utc.astimezone(owner_tz)

    for a in rows:
        a.start_utc = start_utc
        a.end_utc = end_utc
        db.add(a)
    db.commit()
    try:
        person_ids = [
            int(a.person_id) for a in rows if getattr(a, "person_id", None) is not None
        ]
        recips = _collect_unique_account_recipients_for_person_ids(
            db, owner.id, person_ids
        )
        for to_email, to_name in recips:
            email_pkg = build_appt_email(
                audience="client",
                action="updated",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=None,
                initiator_label=owner.name or "the owner",
                status_label="booked",
                recipient_name=to_name or to_email,
                message=None,
                old_start_local=old_start_local,
                old_end_local=old_end_local,
                include_ics=True,
                organizer_email=owner.email,
                attendee_email=to_email,
            )
            background_tasks.add_task(
                send_email,
                to_email,
                email_pkg.subject,
                email_pkg.text,
                email_pkg.html,
                email_pkg.ics_text,
            )
    except Exception:
        pass

    return {"ok": True, "group_id": group_id, "updated": len(rows)}


class GroupAttendeesPayload(BaseModel):
    """Payload to add one or more attendees to an existing group lesson."""

    person_ids: list[int]
    appointment_ids: list[str] | None = None


@router.post("/appointments/group/{group_id}/attendees", response_model=dict)
def admin_group_add_attendees(
    group_id: str,
    payload: GroupAttendeesPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Add new attendees to a group, reusing same-time single seats when possible."""
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    sample = (
        db.query(Appointment)
        .filter(Appointment.owner_id == user.sub, Appointment.group_id == gid)
        .first()
    )
    if not sample:
        raise HTTPException(404, "Group not found")
    start_utc, end_utc = sample.start_utc, sample.end_utc
    dur = int((end_utc - start_utc).total_seconds() // 60)
    existing_in_group: list[Appointment] = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.group_id == gid,
            Appointment.status != "canceled",
        )
        .all()
    )
    existing_pids: set[int] = {
        int(a.person_id)
        for a in existing_in_group
        if getattr(a, "person_id", None) is not None
    }

    added = 0
    added_pids: set[int] = set()
    for pid in payload.person_ids:
        exists_in_group = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.group_id == gid,
                Appointment.person_id == int(pid),
                Appointment.status != "canceled",
            )
            .first()
        )
        if exists_in_group:
            continue
        existing_same_time = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.start_utc == start_utc,
                Appointment.status != "canceled",
                Appointment.person_id == int(pid),
            )
            .first()
        )
        if existing_same_time:
            if existing_same_time.group_id != gid:
                existing_same_time.group_id = gid
            existing_same_time.end_utc = end_utc
            gp = getattr(owner, "group_price_60_cents", None)
            if int(dur) == 60 and gp is not None and int(gp) > 0:
                existing_same_time.price_override_cents = int(gp)
            else:
                existing_same_time.price_override_cents = get_default_price_cents(
                    db, owner_user_id=owner.id, duration_minutes=dur
                )
            existing_same_time.payment_status = "unpaid"
            existing_same_time.amount_paid_cents = (
                existing_same_time.amount_paid_cents or 0
            )
            db.add(existing_same_time)
            try:
                db.flush()
                added += 1
                added_pids.add(int(pid))
            except Exception:
                db.rollback()
            continue
        a = Appointment(
            id=uuid.uuid4(),
            owner_id=owner.id,
            person_id=int(pid),
            start_utc=start_utc,
            end_utc=end_utc,
            status="booked",
            group_id=gid,
        )
        gp = getattr(owner, "group_price_60_cents", None)
        if int(dur) == 60 and gp is not None and int(gp) > 0:
            a.price_override_cents = int(gp)
        else:
            a.price_override_cents = get_default_price_cents(
                db, owner_user_id=owner.id, duration_minutes=dur
            )
        a.payment_status = "unpaid"
        a.amount_paid_cents = a.amount_paid_cents or 0
        db.add(a)
        try:
            db.flush()
            added += 1
            added_pids.add(int(pid))
        except IntegrityError:
            db.rollback()
            continue
    requested = {int(p) for p in payload.person_ids}
    missing = [p for p in requested if p not in existing_pids]
    if added == 0 and missing:
        raise HTTPException(
            409,
            detail=(
                "Could not add attendees. Your database likely still enforces a legacy "
                "unique index on (owner_id, start_utc) that prevents multiple attendees at the "
                "same time. Apply the Alembic migration 'sched_0007_group_lessons' to relax the "
                "constraint to (owner_id, start_utc, person_id)."
            ),
        )

    db.commit()
    try:
        if added_pids:
            recips = _collect_unique_account_recipients_for_person_ids(
                db, owner.id, list(added_pids)
            )
            owner_tz = ZoneInfo(owner.timezone)
            s_loc = start_utc.astimezone(owner_tz)
            e_loc = end_utc.astimezone(owner_tz)
            for to_email, to_name in recips:
                email_pkg = build_appt_email(
                    audience="client",
                    action="created",
                    owner=owner,
                    start_local=s_loc,
                    end_local=e_loc,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label="booked",
                    recipient_name=to_name or to_email,
                    message=None,
                    include_ics=True,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    email_pkg.subject,
                    email_pkg.text,
                    email_pkg.html,
                    email_pkg.ics_text,
                )
    except Exception:
        pass

    return {"ok": True, "added": added}


@router.delete("/appointments/group/{group_id}/attendees", response_model=dict)
def admin_group_remove_attendees(
    group_id: str,
    payload: GroupAttendeesPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    q = db.query(Appointment).filter(
        Appointment.owner_id == user.sub,
        Appointment.group_id == gid,
        Appointment.status != "canceled",
    )
    # Support removal by person_ids OR appointment_ids (for attendees without a person record)
    have_pids = [int(p) for p in (payload.person_ids or []) if int(p) > 0]
    have_aids = [str(a) for a in (payload.appointment_ids or []) if a]
    if have_pids and have_aids:
        q = q.filter(
            or_(Appointment.person_id.in_(have_pids), Appointment.id.in_(have_aids))
        )
    elif have_pids:
        q = q.filter(Appointment.person_id.in_(have_pids))
    elif have_aids:
        q = q.filter(Appointment.id.in_(have_aids))
    else:
        return {"ok": True, "removed": 0}
    rows = q.all()
    # Capture person_ids and time for email before status change
    person_ids = [
        int(a.person_id) for a in rows if getattr(a, "person_id", None) is not None
    ]
    owner = db.query(User).filter(User.id == user.sub).first()
    owner_tz = ZoneInfo(owner.timezone) if owner else ZoneInfo("UTC")
    if rows:
        s_loc = rows[0].start_utc.astimezone(owner_tz)
        e_loc = rows[0].end_utc.astimezone(owner_tz)
    for a in rows:
        a.status = "canceled"
        db.add(a)
    db.commit()

    # Send canceled emails to affected accounts (deduped by account)
    try:
        if person_ids and owner:
            recips = _collect_unique_account_recipients_for_person_ids(
                db, owner.id, person_ids
            )
            for to_email, to_name in recips:
                pkg = build_appt_email(
                    audience="client",
                    action="canceled",
                    owner=owner,
                    start_local=s_loc,
                    end_local=e_loc,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label="canceled",
                    recipient_name=to_name or to_email,
                    message=None,
                    include_ics=False,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    pkg.subject,
                    pkg.text,
                    pkg.html,
                    pkg.ics_text,
                )
    except Exception:
        pass

    return {"ok": True, "removed": len(rows)}


@router.put("/appointments/group/{group_id}/cancel", response_model=dict)
def admin_group_cancel(
    group_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    rows = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == user.sub,
            Appointment.group_id == gid,
            Appointment.status != "canceled",
        )
        .all()
    )
    owner = db.query(User).filter(User.id == user.sub).first()
    owner_tz = ZoneInfo(owner.timezone) if owner else ZoneInfo("UTC")
    person_ids = [
        int(a.person_id) for a in rows if getattr(a, "person_id", None) is not None
    ]
    if rows:
        s_loc = rows[0].start_utc.astimezone(owner_tz)
        e_loc = rows[0].end_utc.astimezone(owner_tz)
    for a in rows:
        a.status = "canceled"
        db.add(a)
    db.commit()
    try:
        if person_ids and owner:
            recips = _collect_unique_account_recipients_for_person_ids(
                db, owner.id, person_ids
            )
            for to_email, to_name in recips:
                pkg = build_appt_email(
                    audience="client",
                    action="canceled",
                    owner=owner,
                    start_local=s_loc,
                    end_local=e_loc,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label="canceled",
                    recipient_name=to_name or to_email,
                    message=None,
                    include_ics=False,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    pkg.subject,
                    pkg.text,
                    pkg.html,
                    pkg.ics_text,
                )
    except Exception:
        pass

    return {"ok": True, "canceled": len(rows)}
