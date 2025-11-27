from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.core.auth import require_client, TokenUser
from app.models import Appointment
from app.schemas import ClientApptListOut, ClientApptRow
from fastapi import HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional, Literal
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_, or_
from app.core.auth import get_current_user
from app.models import (
    User,
    RoleEnum,
    ClientAccount,
    PrepaidBundle,
    PrepaidLedger,
)
from app.schemas import (
    AppointmentCancel,
    AppointmentUpdateClient,
)
from app.schemas import ClientPaymentsOut
from ._helpers import (
    UTC,
    uuid_str,
    send_email,
    _compute_final_slots_for_day,
    _get_owner,
    build_appt_email,
)

from services.payments import (
    infer_payment_status,
    expected_price_cents,
)
from services.wallets import auto_apply_wallet_funds
from services.features import get_owner_flag
from sqlalchemy import func, case


router = APIRouter(prefix="/api/client", tags=["client"])
"""Router for client self-service scheduling and payments endpoints."""


@router.get("/appointments", response_model=ClientApptListOut)
def get_my_appointments(
    status: list[Literal["booked", "completed", "canceled"]] | None = Query(None),
    payment_status: (
        list[Literal["unpaid", "paid", "refunded", "waived", "partial", "unknown"]]
        | None
    ) = Query(None),
    date_from: Optional[date] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[date] = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
    me: TokenUser = Depends(require_client),
) -> ClientApptListOut:
    """List client appointments with local→UTC date filtering and payment info.

    Behavior:
    - Interprets 'date_from'/'date_to' in the owner’s timezone and converts to
      UTC bounds '[start_utc >= startLocal@00:00, start_utc < (endLocal+1d)@00:00]'.
    - Validates 'date_to >= date_from' when both provided.
    - Applies 'payment_status' filtering after deriving status from amounts.
    """
    if date_from and date_to and date_to < date_from:
        raise HTTPException(status_code=400, detail="date_to must be >= date_from")

    owner = _get_owner(db)
    owner_tz = ZoneInfo(owner.timezone)
    start_utc_bound = None
    end_utc_bound_excl = None
    if date_from:
        start_local = datetime.combine(date_from, datetime.min.time()).replace(
            tzinfo=owner_tz
        )
        start_utc_bound = start_local.astimezone(UTC)
    if date_to:
        end_local_excl = datetime.combine(date_to, datetime.min.time()).replace(
            tzinfo=owner_tz
        ) + timedelta(days=1)
        end_utc_bound_excl = end_local_excl.astimezone(UTC)
    from app.models import ClientEmail, ClientAccount, Person, User as AuthUser

    emails: set[str] = set()
    me_user = db.query(AuthUser).filter(AuthUser.id == me.sub).first()
    if me_user and me_user.email:
        emails.add(me_user.email.lower())
    acct = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.owner_user_id == owner.id,
            ClientAccount.client_user_id == me.sub,
        )
        .first()
    )
    person_ids: list[int] = []
    if acct:
        rows = db.query(ClientEmail).filter(ClientEmail.account_id == acct.id).all()
        for ce in rows:
            if ce.email:
                emails.add(ce.email.lower())
        person_ids = [
            pid
            for (pid,) in db.query(Person.id).filter(Person.account_id == acct.id).all()
        ]

    identity_or_clauses = [Appointment.client_id == me.sub]
    if emails:
        identity_or_clauses.append(
            and_(
                Appointment.client_email.isnot(None),
                func.lower(Appointment.client_email).in_(list(emails)),
            )
        )
    if person_ids:
        identity_or_clauses.append(Appointment.person_id.in_(person_ids))

    q = (
        db.query(Appointment)
        .filter(Appointment.owner_id == owner.id)
        .filter(or_(*identity_or_clauses))
    )
    if status:
        q = q.filter(Appointment.status.in_(status))
    if start_utc_bound:
        q = q.filter(Appointment.start_utc >= start_utc_bound)
    if end_utc_bound_excl:
        q = q.filter(Appointment.start_utc < end_utc_bound_excl)

    rows_db = q.order_by(Appointment.start_utc.desc()).all()
    from services.payments import (
        _service_price_map,
    )

    price_map = _service_price_map(db, owner_user_id=owner.id)

    out_rows: list[ClientApptRow] = []
    for a in rows_db:
        expected = expected_price_cents(a, price_map)
        derived_status = infer_payment_status(a, expected)
        duration_minutes = (
            int((a.end_utc - a.start_utc).total_seconds() // 60)
            if a.start_utc and a.end_utc
            else None
        )

        out_rows.append(
            ClientApptRow(
                id=a.id,
                start_utc=a.start_utc,
                end_utc=a.end_utc,
                status=a.status,
                payment_status=derived_status,
                amount_paid_cents=a.amount_paid_cents,
                price_cents=expected,
                duration_minutes=duration_minutes,
            )
        )
    if payment_status:
        allowed = set(payment_status)
        out_rows = [r for r in out_rows if r.payment_status in allowed]

    return ClientApptListOut(rows=out_rows)


class MyBookPayload(BaseModel):
    """Client booking request body.

    Provide 'start_local' in the owner's timezone; optionally include
    duration to accept any window that fully contains the requested
    [start, end). If duration is omitted, the window must begin exactly
    at 'start_local'.
    """

    start_local: datetime
    duration_minutes: Optional[int] = None
    message: Optional[str] = None
    lesson_person_name: Optional[str] = None


@router.get("/my-appointments")
def my_appointments(
    scope: str = Query("upcoming", pattern="^(upcoming|history|all)$"),
    tz: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    me: TokenUser = Depends(get_current_user),
) -> list[dict]:
    """Return the authenticated user’s appointments with filtering.

    Scope:
      - 'upcoming': 'end_utc >= now' and 'status != canceled'
      - 'history': 'end_utc < now' or 'status == canceled'
      - 'all': no extra filtering
    """
    from zoneinfo import ZoneInfo

    owner = _get_owner(db)
    tz_name = tz or owner.timezone
    now = datetime.now(ZoneInfo(tz_name)).astimezone(UTC)

    q = db.query(Appointment).filter(Appointment.client_id == me.sub)

    if scope == "upcoming":
        q = q.filter(Appointment.end_utc >= now, Appointment.status != "canceled")
    elif scope == "history":
        q = q.filter((Appointment.end_utc < now) | (Appointment.status == "canceled"))

    appts = q.order_by(Appointment.start_utc.asc()).all()
    from services.payments import (
        _service_price_map,
    )

    price_map = _service_price_map(db, owner_user_id=owner.id)

    out = []
    for a in appts:
        expected = expected_price_cents(a, price_map)
        pstatus = infer_payment_status(a, expected)
        out.append(
            {
                "id": a.id,
                "start_utc": a.start_utc,
                "end_utc": a.end_utc,
                "status": a.status,
                "payment_status": pstatus,
                "client_email": None,
                "client_name": None,
            }
        )
    return out


@router.post("/my/appointments")
def my_book(
    payload: MyBookPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    me: TokenUser = Depends(get_current_user),
) -> dict:
    """Book an appointment as the current client.

    Behavior:
    - Uses the single-owner model via '_get_owner'.
    - Validates availability using unified daily slots (weekly ∪ specials − time off − appts).
    - Reuses a canceled row at the same start if present; otherwise inserts a new appointment.
    - Optionally attaches a 'Person' entry and sends owner/client notifications.
    """
    owner = _get_owner(db)
    owner_tz = ZoneInfo(owner.timezone)

    # Normalize start_local to owner tz
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    final_pairs = _compute_final_slots_for_day(
        db, owner, start_local.date(), owner.timezone
    )

    # If client supplied duration, accept any window that fully contains [start_local, end_local].
    # Otherwise, accept a pair that starts exactly at start_local and take its natural end.
    if payload.duration_minutes is not None:
        end_local = start_local + timedelta(minutes=payload.duration_minutes)
        within = any(s <= start_local and end_local <= e for (s, e) in final_pairs)
        if not within:
            raise HTTPException(409, "Slot is no longer available")
    else:
        match = next(((s, e) for (s, e) in final_pairs if s == start_local), None)
        if not match:
            raise HTTPException(409, "Slot is no longer available")
        _, end_local = match
    # Reuse a canceled row at the same start_utc if one exists (handles legacy unique indexes)
    s_utc = start_local.astimezone(UTC)
    existing_same_start = (
        db.query(Appointment)
        .filter(Appointment.owner_id == owner.id, Appointment.start_utc == s_utc)
        .first()
    )
    if existing_same_start:
        if existing_same_start.status != "canceled":
            raise HTTPException(
                409, "Another appointment already exists at that exact start time."
            )
        a = existing_same_start
        a.end_utc = end_local.astimezone(UTC)
        a.status = "booked"
        a.client_id = me.sub
        a.client_name = getattr(me, "name", None) or getattr(me, "email", None)
        a.client_email = getattr(me, "email", None)
        a.cancel_reason = None
        a.payment_status = "unpaid"
        a.amount_paid_cents = 0
        a.bundle_id = None
        db.add(a)
        try:
            db.commit()
            db.refresh(a)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409, "Another appointment already exists at that exact start time."
            )

        # Attach lesson person if provided
        try:
            if payload.lesson_person_name and payload.lesson_person_name.strip():
                from app.models import ClientAccount, Person

                acct = (
                    db.query(ClientAccount)
                    .filter(
                        ClientAccount.owner_user_id == owner.id,
                        ClientAccount.client_user_id == me.sub,
                        ClientAccount.deleted_at.is_(None),
                    )
                    .first()
                )
                if acct:
                    nm = payload.lesson_person_name.strip()
                    p = (
                        db.query(Person)
                        .filter(
                            Person.account_id == acct.id,
                            func.lower(Person.full_name) == func.lower(nm),
                        )
                        .first()
                    )
                    if p is None:
                        p = Person(account_id=acct.id, full_name=nm, email=None)
                        db.add(p)
                        db.flush()
                    a.person_id = p.id
                    db.add(a)
                    db.commit()
                    db.refresh(a)
        except Exception:
            pass

        # Emails (owner + client)
        if owner.email:
            owner_email = build_appt_email(
                audience="owner",
                action="created",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=str(a.id),
                initiator_label=me.email or "client",
                status_label=a.status,
                message=payload.message,
            )
            background_tasks.add_task(
                send_email,
                owner.email,
                owner_email.subject,
                owner_email.text,
                owner_email.html,
                owner_email.ics_text,
            )

        client_email_addr = getattr(me, "email", None)
        if client_email_addr:
            client_email = build_appt_email(
                audience="client",
                action="created",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=str(a.id),
                initiator_label=owner.name or "the owner",
                status_label=a.status,
                recipient_name=getattr(me, "name", None) or client_email_addr,
                include_ics=True,
                organizer_email=owner.email,
                attendee_email=client_email_addr,
            )
            background_tasks.add_task(
                send_email,
                client_email_addr,
                client_email.subject,
                client_email.text,
                client_email.html,
                client_email.ics_text,
            )

        return {
            "ok": True,
            "appointment_id": a.id,
            "start_utc": a.start_utc,
            "end_utc": a.end_utc,
            "status": a.status,
        }

    appt = Appointment(
        id=uuid_str(),
        owner_id=owner.id,
        client_id=me.sub,
        client_name=getattr(me, "name", None) or getattr(me, "email", None),
        client_email=getattr(me, "email", None),
        start_utc=start_local.astimezone(UTC),
        end_utc=end_local.astimezone(UTC),
        status="booked",
    )
    db.add(appt)
    # Attach lesson person under this client's account if provided
    try:
        if payload.lesson_person_name and payload.lesson_person_name.strip():
            from app.models import ClientAccount, Person

            acct = (
                db.query(ClientAccount)
                .filter(
                    ClientAccount.owner_user_id == owner.id,
                    ClientAccount.client_user_id == me.sub,
                    ClientAccount.deleted_at.is_(None),
                )
                .first()
            )
            if acct:
                nm = payload.lesson_person_name.strip()
                p = (
                    db.query(Person)
                    .filter(
                        Person.account_id == acct.id,
                        func.lower(Person.full_name) == func.lower(nm),
                    )
                    .first()
                )
                if p is None:
                    p = Person(account_id=acct.id, full_name=nm, email=None)
                    db.add(p)
                    db.flush()
                appt.person_id = p.id
    except Exception:
        pass
    try:
        db.commit()
        db.refresh(appt)
        if owner.email:
            owner_email = build_appt_email(
                audience="owner",
                action="created",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=str(appt.id),
                initiator_label=me.email or "client",
                status_label=appt.status,
                message=payload.message,
            )
            background_tasks.add_task(
                send_email,
                owner.email,
                owner_email.subject,
                owner_email.text,
                owner_email.html,
                owner_email.ics_text,
            )

        client_email_addr = getattr(me, "email", None)
        if client_email_addr:
            client_email = build_appt_email(
                audience="client",
                action="created",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=str(appt.id),
                initiator_label=owner.name or "the owner",
                status_label=appt.status,
                recipient_name=getattr(me, "name", None) or client_email_addr,
                include_ics=True,
                organizer_email=owner.email,
                attendee_email=client_email_addr,
            )
            background_tasks.add_task(
                send_email,
                client_email_addr,
                client_email.subject,
                client_email.text,
                client_email.html,
                client_email.ics_text,
            )
        # Best-effort: auto-apply wallet funds after booking, if enabled for this owner
        try:
            if get_owner_flag(
                owner.id,
                "auto_apply_wallet_on_book",
                "FEATURE_AUTO_APPLY_WALLET_ON_BOOK",
                default=True,
            ):
                wallet_rows = (
                    db.query(PrepaidBundle.id)
                    .filter(
                        PrepaidBundle.owner_id == owner.id,
                        PrepaidBundle.client_id == me.sub,
                        PrepaidBundle.total_credits == 0,
                    )
                    .all()
                )
                for (wid,) in wallet_rows:
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
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Double-booking detected, please pick another slot")

    return {
        "ok": True,
        "appointment_id": appt.id,
        "start_utc": appt.start_utc,
        "end_utc": appt.end_utc,
        "status": appt.status,
    }


class MyBookRecurringPayload(BaseModel):
    """Recurring client booking request body (weekly pattern).

    Either 'occurrences' or 'until_date' can cap the series; a hard cap of
    15 occurrences applies for clients.
    """

    start_local: datetime
    duration_minutes: int
    repeat_every_weeks: int = 1
    occurrences: Optional[int] = None
    until_date: Optional[date] = None
    message: Optional[str] = None
    lesson_person_name: Optional[str] = None


@router.post("/my/appointments/recurring")
def my_book_recurring(
    payload: MyBookRecurringPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    me: TokenUser = Depends(get_current_user),
) -> dict:
    """Book a weekly recurring series as the current client.

    Only books occurrences that exactly match available '[start,end]' pairs for
    each day. Enforces a maximum of 15 occurrences per request for clients.
    """
    if int(payload.duration_minutes) <= 0:
        raise HTTPException(400, "duration_minutes must be positive")

    owner = _get_owner(db)
    owner_tz = ZoneInfo(owner.timezone)

    # Normalize start_local to owner tz
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )

    step = timedelta(weeks=int(payload.repeat_every_weeks or 1))
    duration = timedelta(minutes=int(payload.duration_minutes))
    # Enforce max 15 occurrences for clients
    if payload.occurrences is not None and payload.occurrences > 15:
        raise HTTPException(400, "occurrences must be <= 15")

    cur = start_local
    count = 0
    max_iter = payload.occurrences if payload.occurrences is not None else 15
    limit_date = payload.until_date
    occs: list[tuple[datetime, datetime]] = []
    while True:
        if limit_date is not None and cur.date() > limit_date:
            break
        end_loc = cur + duration
        occs.append((cur, end_loc))
        count += 1
        if payload.occurrences is not None and count >= payload.occurrences:
            break
        if limit_date is None and payload.occurrences is None and count >= 15:
            break
        if count >= min(max_iter, 15):
            break
        cur = cur + step

    if not occs:
        raise HTTPException(
            400, "No occurrences were generated for the requested recurrence"
        )

    # Ensure client account exists for optional person linkage
    from app.models import ClientAccount, Person

    acct = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.owner_user_id == owner.id,
            ClientAccount.client_user_id == me.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .first()
    )
    if acct is None:
        acct = ClientAccount(
            owner_user_id=owner.id,
            client_user_id=me.sub,
            name=getattr(me, "name", None),
        )
        db.add(acct)
        db.flush()

    person_id: Optional[int] = None
    if payload.lesson_person_name and payload.lesson_person_name.strip():
        nm = payload.lesson_person_name.strip()
        p = (
            db.query(Person)
            .filter(
                Person.account_id == acct.id,
                func.lower(Person.full_name) == func.lower(nm),
            )
            .first()
        )
        if p is None:
            p = Person(account_id=acct.id, full_name=nm, email=None)
            db.add(p)
            db.flush()
        person_id = p.id

    created: list[dict] = []
    conflicts: list[dict] = []

    for s_loc, e_loc in occs:
        final_pairs = _compute_final_slots_for_day(
            db, owner, s_loc.date(), owner.timezone
        )
        within = any(s <= s_loc and e_loc <= e for (s, e) in final_pairs)
        if not within:
            conflicts.append(
                {
                    "start_local": s_loc.isoformat(timespec="minutes"),
                    "reason": "not available",
                }
            )
            continue

        # Duplicate start guard with reuse of canceled row when present (handles legacy unique indexes)
        s_utc = s_loc.astimezone(UTC)
        existing_same_start = (
            db.query(Appointment)
            .filter(Appointment.owner_id == owner.id, Appointment.start_utc == s_utc)
            .first()
        )
        if existing_same_start:
            if existing_same_start.status != "canceled":
                conflicts.append(
                    {
                        "start_local": s_loc.isoformat(timespec="minutes"),
                        "reason": "conflicts with another appointment",
                    }
                )
                continue
            # Reuse canceled row as a new booking occurrence
            a = existing_same_start
            a.end_utc = e_loc.astimezone(UTC)
            a.status = "booked"
            a.client_id = me.sub
            a.client_name = getattr(me, "name", None) or getattr(me, "email", None)
            a.client_email = getattr(me, "email", None)
            a.cancel_reason = None
            a.payment_status = "unpaid"
            a.amount_paid_cents = 0
            a.bundle_id = None
            db.add(a)
            try:
                db.commit()
                db.refresh(a)
            except IntegrityError:
                db.rollback()
                conflicts.append(
                    {
                        "start_local": s_loc.isoformat(timespec="minutes"),
                        "reason": "conflicts with another appointment",
                    }
                )
                continue
            created.append(
                {
                    "appointment_id": str(a.id),
                    "start_utc": a.start_utc,
                    "end_utc": a.end_utc,
                    "status": a.status,
                }
            )
            continue

        appt = Appointment(
            id=uuid_str(),
            owner_id=owner.id,
            client_id=me.sub,
            client_name=getattr(me, "name", None) or getattr(me, "email", None),
            client_email=getattr(me, "email", None),
            person_id=person_id,
            start_utc=s_loc.astimezone(UTC),
            end_utc=e_loc.astimezone(UTC),
            status="booked",
        )
        db.add(appt)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            conflicts.append(
                {
                    "start_local": s_loc.isoformat(timespec="minutes"),
                    "reason": "conflicts with another appointment",
                }
            )
            continue

        created.append(
            {
                "appointment_id": str(getattr(appt, "id", "")),
                "status": appt.status,
                "start_local": s_loc.isoformat(timespec="minutes"),
            }
        )

        # Emails (queued)
        if owner.email:
            owner_email = build_appt_email(
                audience="owner",
                action="created",
                owner=owner,
                start_local=s_loc,
                end_local=e_loc,
                appointment_id=str(getattr(appt, "id", "")),
                initiator_label=getattr(me, "email", None) or "client",
                status_label=appt.status,
                message=payload.message,
            )
            background_tasks.add_task(
                send_email,
                owner.email,
                owner_email.subject,
                owner_email.text,
                owner_email.html,
                owner_email.ics_text,
            )
        client_email_addr = getattr(me, "email", None)
        if client_email_addr:
            client_email = build_appt_email(
                audience="client",
                action="created",
                owner=owner,
                start_local=s_loc,
                end_local=e_loc,
                appointment_id=str(getattr(appt, "id", "")),
                initiator_label=owner.name or "the owner",
                status_label="booked",
                recipient_name=getattr(me, "name", None) or client_email_addr,
                include_ics=True,
                organizer_email=owner.email,
                attendee_email=client_email_addr,
            )
            background_tasks.add_task(
                send_email,
                client_email_addr,
                client_email.subject,
                client_email.text,
                client_email.html,
                client_email.ics_text,
            )

    try:
        db.commit()
    except IntegrityError:
        db.rollback()

    # auto-apply wallet funds across this client's wallet(s), if enabled
    try:
        if get_owner_flag(
            owner.id,
            "auto_apply_wallet_on_book",
            "FEATURE_AUTO_APPLY_WALLET_ON_BOOK",
            default=True,
        ):
            wallet_rows = (
                db.query(PrepaidBundle.id)
                .filter(
                    PrepaidBundle.owner_id == owner.id,
                    PrepaidBundle.client_id == me.sub,
                    PrepaidBundle.total_credits == 0,
                )
                .all()
            )
            for (wid,) in wallet_rows:
                auto_apply_wallet_funds(
                    db,
                    owner_id=str(owner.id),
                    bundle_id=int(wid),
                    note_prefix="Auto-apply wallet funds after recurring booking",
                )
    except Exception:
        pass

    return {
        "ok": True,
        "count": len(created),
        "appointments": created,
        "conflicts": conflicts,
    }


class MyUpdatePayload(BaseModel):
    """Client update payload for cancel or reschedule actions on own appointment."""

    start_local: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    status: Optional[Literal["canceled"]] = None
    message: Optional[str] = None


@router.post("/appointments/{appt_id}/cancel", response_model=dict)
def cancel_with_reason(
    appt_id: str,
    payload: AppointmentCancel,
    db: Session = Depends(get_db),
    me: TokenUser = Depends(get_current_user),
) -> dict:
    """Cancel an appointment with a reason, with refund rules.

    Applies a ≥24h full-refund policy in owner-local time: restores credits and
    refunds cash to wallet when applicable. Sets 'payment_status' accordingly.
    """
    appt = db.query(Appointment).filter_by(id=appt_id).first()
    if not appt:
        raise HTTPException(404, "Appointment not found")
    if appt.client_id != me.sub and me.role != RoleEnum.OWNER:
        raise HTTPException(403, "Not authorized")
    if appt.status != "canceled":
        # Determine if cancellation is ≥24h before start (owner timezone)
        owner = db.query(User).filter(User.id == appt.owner_id).first()
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
        qualifies_full_refund = (appt_start_local - now_local) >= timedelta(hours=24)
        appt.status = "canceled"
        if qualifies_full_refund:
            # Restore bundle/credits and refund cash to wallet
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
                            note="Auto-restore on client cancel",
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
                            note="Auto-restore on client cancel",
                        )
                    )
            try:
                cash_paid = int(getattr(appt, "amount_paid_cents", 0) or 0)
            except Exception:
                cash_paid = 0
            if cash_paid > 0 and getattr(appt, "client_id", None):
                from app.models import PrepaidBundle, PrepaidLedger

                wallet = (
                    db.query(PrepaidBundle)
                    .filter(
                        PrepaidBundle.owner_id == appt.owner_id,
                        PrepaidBundle.client_id == str(appt.client_id),
                        PrepaidBundle.total_credits == 0,
                    )
                    .order_by(PrepaidBundle.created_at.desc())
                    .first()
                )
                if not wallet:
                    wallet = PrepaidBundle(
                        owner_id=str(appt.owner_id),
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
                appt.payment_status = "refunded"
                try:
                    appt.amount_paid_cents = 0
                except Exception:
                    pass
            # Detach bundle so UI doesn't infer bundle after restore
            try:
                appt.bundle_id = None
            except Exception:
                pass
    appt.cancel_reason = payload.reason
    db.add(appt)
    db.commit()
    return {"ok": True, "notice": "Appointment cancelled"}


@router.put("/my/appointments/{appt_id}")
def my_update(
    appt_id: str,
    body: MyUpdatePayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    me: TokenUser = Depends(get_current_user),
) -> dict:
    """Cancel or reschedule the client’s own appointment.

    Enforces a 24h lock window for rescheduling/cancel in owner-local time;
    owners can still override via owner endpoints.
    """
    appt = db.query(Appointment).filter_by(id=appt_id, client_id=me.sub).first()
    if not appt:
        raise HTTPException(404, "Appointment not found")

    owner = db.query(User).filter(User.id == appt.owner_id).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    owner_tz = ZoneInfo(owner.timezone)

    # Lock window: 24h before start
    now_local = datetime.now(owner_tz)
    appt_start_local = appt.start_utc.astimezone(owner_tz)
    within_24h = (appt_start_local - now_local) < timedelta(hours=24)

    # Cancel flow
    if body.status == "canceled":
        if within_24h:
            raise HTTPException(
                403,
                "Cancellations are locked within 24 hours of start. Please contact the owner.",
            )
        reason = (body.message or "").strip()
        if not reason:
            raise HTTPException(400, "Please provide a cancellation reason.")
        # Track whether any credits/funds were restored (used for payment_status below)
        restored_any = False
        if appt.status != "canceled":
            # Auto-restore wallet/bundle effects if any
            if getattr(appt, "bundle_id", None):
                bid = int(appt.bundle_id)
                from app.models import PrepaidLedger, PrepaidBundle

                # Restore consumed credit if present
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
                            note="Auto-restore on client cancel",
                        )
                    )
                    restored_any = True
                # Restore wallet funds consumed for this appointment
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
                            note="Auto-restore on client cancel",
                        )
                    )
                    restored_any = True
        appt.status = "canceled"
        # Detach bundle to prevent inferred 'bundle' status after restores
        try:
            appt.bundle_id = None
        except Exception:
            pass
        # Full refund to wallet (cash portion); we're outside 24h by guard above
        try:
            cash_paid = int(getattr(appt, "amount_paid_cents", 0) or 0)
        except Exception:
            cash_paid = 0
        if cash_paid > 0 and getattr(appt, "client_id", None):
            from app.models import PrepaidBundle, PrepaidLedger

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
                    note="Full refund to wallet on >=24h cancel",
                )
            )
            appt.payment_status = "refunded"
            try:
                appt.amount_paid_cents = 0
            except Exception:
                pass
        else:
            # No cash paid; if we restored funds/credits, reflect as refunded
            if restored_any:
                appt.payment_status = "refunded"
        if reason:
            appt.cancel_reason = reason
        db.add(appt)
        db.commit()
        db.refresh(appt)

        if owner.email:
            owner_email = build_appt_email(
                audience="owner",
                action="canceled",
                owner=owner,
                start_local=appt_start_local,
                end_local=appt.end_utc.astimezone(owner_tz),
                appointment_id=str(appt.id),
                initiator_label=me.email or "client",
                status_label=appt.status,
                message=reason,
            )
            background_tasks.add_task(
                send_email,
                owner.email,
                owner_email.subject,
                owner_email.text,
                owner_email.html,
                owner_email.ics_text,
            )

        client_email_addr = getattr(me, "email", None)
        if client_email_addr:
            client_email = build_appt_email(
                audience="client",
                action="canceled",
                owner=owner,
                start_local=appt_start_local,
                end_local=appt.end_utc.astimezone(owner_tz),
                appointment_id=str(appt.id),
                initiator_label=owner.name or "our studio",
                status_label=appt.status,
                recipient_name=getattr(me, "name", None) or client_email_addr,
                message=None,
                include_ics=False,
                organizer_email=owner.email,
                attendee_email=client_email_addr,
            )
            background_tasks.add_task(
                send_email,
                client_email_addr,
                client_email.subject,
                client_email.text,
                client_email.html,
                client_email.ics_text,
            )

        return {"ok": True, "appointment_id": appt.id, "status": appt.status}

    # Reschedule flow
    if body.start_local or body.duration_minutes:
        if within_24h:
            raise HTTPException(
                403,
                "Rescheduling is locked within 24 hours of start. Please contact the owner.",
            )

        start_local = (
            body.start_local.replace(tzinfo=owner_tz)
            if (body.start_local and body.start_local.tzinfo is None)
            else (
                body.start_local.astimezone(owner_tz)
                if body.start_local
                else appt.start_utc.astimezone(owner_tz)
            )
        )
        current_duration = int((appt.end_utc - appt.start_utc).total_seconds() // 60)
        duration = (
            body.duration_minutes
            if body.duration_minutes is not None
            else current_duration
        )
        if duration <= 0:
            raise HTTPException(400, "duration_minutes must be positive")
        final_pairs = _compute_final_slots_for_day(
            db, owner, start_local.date(), owner.timezone
        )

        if body.duration_minutes is not None:
            end_local = start_local + timedelta(minutes=duration)
            within = any(s <= start_local and end_local <= e for (s, e) in final_pairs)
            if not within:
                raise HTTPException(409, "New time is not available")
        else:
            match = next(((s, e) for (s, e) in final_pairs if s == start_local), None)
            if not match:
                raise HTTPException(409, "New time is not available")
            _, end_local = match

        # capture old times for email before write
        old_start_local = appt.start_utc.astimezone(owner_tz)
        old_end_local = appt.end_utc.astimezone(owner_tz)

        appt.start_utc = start_local.astimezone(UTC)
        appt.end_utc = end_local.astimezone(UTC)
        if body.message:
            appt.client_change_note = body.message
        db.add(appt)
        try:
            db.commit()
            db.refresh(appt)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409, "Another appointment already exists at that exact start time."
            )

        if owner.email:
            owner_email = build_appt_email(
                audience="owner",
                action="updated",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=str(appt.id),
                initiator_label=me.email or "client",
                status_label=appt.status,
                message=body.message,
                old_start_local=old_start_local,
                old_end_local=old_end_local,
            )
            background_tasks.add_task(
                send_email,
                owner.email,
                owner_email.subject,
                owner_email.text,
                owner_email.html,
                owner_email.ics_text,
            )

        client_email_addr = getattr(me, "email", None)
        if client_email_addr:
            client_email = build_appt_email(
                audience="client",
                action="updated",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=str(appt.id),
                initiator_label=owner.name or "our studio",
                status_label=appt.status,
                recipient_name=getattr(me, "name", None) or client_email_addr,
                old_start_local=old_start_local,
                old_end_local=old_end_local,
                include_ics=True,
                organizer_email=owner.email,
                attendee_email=client_email_addr,
            )
            background_tasks.add_task(
                send_email,
                client_email_addr,
                client_email.subject,
                client_email.text,
                client_email.html,
                client_email.ics_text,
            )
        return {
            "ok": True,
            "appointment_id": appt.id,
            "start_utc": appt.start_utc,
            "end_utc": appt.end_utc,
            "status": appt.status,
        }
    return {"ok": True, "appointment_id": appt.id, "status": appt.status}


@router.patch("/appointments/{appt_id}/client", response_model=dict)
def update_appointment_client(
    appt_id: str,
    payload: AppointmentUpdateClient,
    db: Session = Depends(get_db),
    me: TokenUser = Depends(get_current_user),
):
    """Update client note on an appointment owned by the caller.

    Only updates 'client_previsit_note'. Owner-only fields are not writable
    through this endpoint.
    """
    appt = db.query(Appointment).filter_by(id=appt_id).first()
    if not appt:
        raise HTTPException(404, "Appointment not found")
    if appt.client_id != me.sub and me.role != RoleEnum.OWNER:
        raise HTTPException(403, "Not authorized")
    if payload.client_previsit_note is not None:
        appt.client_previsit_note = payload.client_previsit_note
    db.add(appt)
    db.commit()
    return {"ok": True, "notice": "Note saved"}


class PaymentsSummaryOut(dict):
    pass


@router.get("/payments/summary", response_model=dict)
def my_payments_summary(
    db: Session = Depends(get_db),
    me: TokenUser = Depends(require_client),
) -> dict:
    """Aggregate payment totals for the authenticated client.

    Computes total paid (cash), counts of paid/unpaid appointments, and a
    simple owed count based on past unpaid appointments.
    """
    totals = (
        db.query(
            func.coalesce(
                func.sum(
                    case(
                        (
                            (Appointment.payment_status == "paid"),
                            Appointment.amount_paid_cents,
                        ),
                        else_=0,
                    )
                ),
                0,
            ),
            func.count(case((Appointment.payment_status == "paid", 1))),
            func.count(case((Appointment.payment_status == "unpaid", 1))),
        )
        .filter(Appointment.client_id == me.sub)
        .one()
    )
    total_paid_cents, paid_count, unpaid_count = totals
    now_utc = datetime.utcnow()
    owed_count = (
        db.query(func.count(Appointment.id))
        .filter(
            Appointment.client_id == me.sub,
            Appointment.end_utc < now_utc,
            Appointment.payment_status == "unpaid",
        )
        .scalar()
    )

    return {
        "total_paid_cents": int(total_paid_cents or 0),
        "paid_count": int(paid_count or 0),
        "unpaid_count": int(unpaid_count or 0),
        "owed_count": int(owed_count or 0),
    }


@router.get("/payments", response_model=ClientPaymentsOut)
def client_payments(
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    status: list[Literal["booked", "completed", "canceled"]] | None = Query(None),
    db: Session = Depends(get_db),
    user=Depends(require_client),
) -> ClientPaymentsOut:
    """Return detailed client payments including derived statuses and fees.

    Applies owner-local date bounds, derives payment status for each
    appointment row, and supplements with admin-fee rows when available.
    Returns rows + summary for the authenticated client's account,
    filtered by optional date range and status. Payment status is
    computed from amount/bundle/expected price.

    Date filters are interpreted in the OWNER's timezone and converted
    to UTC bounds:
      start_utc >= (date_from @ 00:00 owner_tz)
      start_utc <  (date_to + 1 day @ 00:00 owner_tz)
    """
    if date_from and date_to and date_to < date_from:
        raise HTTPException(status_code=400, detail="date_to must be >= date_from")

    owner = _get_owner(db)
    owner_tz = ZoneInfo(owner.timezone)

    acct = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.owner_user_id == owner.id,
            ClientAccount.client_user_id == user.sub,
        )
        .first()
    )
    if not acct:
        return {
            "summary": {
                "total_appointments": 0,
                "late_appointments": 0,
                "paid_appointments": 0,
                "unpaid_appointments": 0,
                "total_expected_cents": 0,
                "total_paid_cents": 0,
                "total_owed_cents": 0,
            },
            "rows": [],
        }

    # Compute local->UTC bounds
    start_utc_bound = None
    end_utc_bound_excl = None
    if date_from:
        start_local = datetime.combine(date_from, datetime.min.time()).replace(
            tzinfo=owner_tz
        )
        start_utc_bound = start_local.astimezone(UTC)
    if date_to:
        end_local_excl = datetime.combine(date_to, datetime.min.time()).replace(
            tzinfo=owner_tz
        ) + timedelta(days=1)
        end_utc_bound_excl = end_local_excl.astimezone(UTC)

    price_map = _service_price_map(db, owner.id)

    q = db.query(Appointment).filter(Appointment.client_id == user.sub)
    if status:
        q = q.filter(Appointment.status.in_(status))
    if start_utc_bound:
        q = q.filter(Appointment.start_utc >= start_utc_bound)
    if end_utc_bound_excl:
        q = q.filter(Appointment.start_utc < end_utc_bound_excl)

    rows = q.order_by(Appointment.start_utc.desc()).all()

    out_rows: List[Dict[str, Any]] = []
    roll = {
        "total_appointments": 0,
        "late_appointments": 0,
        "paid_appointments": 0,
        "unpaid_appointments": 0,
        "total_expected_cents": 0,
        "total_paid_cents": 0,
        "total_owed_cents": 0,
    }

    from services.payments import compute_financials
    from app.models import Person as PersonModel

    for a in rows:
        duration_minutes = (
            int((a.end_utc - a.start_utc).total_seconds() // 60)
            if a.start_utc and a.end_utc
            else 0
        )
        fin = compute_financials(db, a, price_map)
        lesson_person_name = None
        try:
            if getattr(a, "person_id", None):
                p = db.query(PersonModel).filter(PersonModel.id == a.person_id).first()
                if p and getattr(p, "full_name", None):
                    lesson_person_name = p.full_name
        except Exception:
            lesson_person_name = None

        out_rows.append(
            {
                "id": str(a.id),
                "start_utc": a.start_utc.isoformat() if a.start_utc else "",
                "duration_minutes": duration_minutes,
                "status": a.status,
                "attendance": getattr(a, "attendance_status", None),
                "lesson_person_name": lesson_person_name,
                "is_group": bool(getattr(a, "group_id", None)),
                "price_cents": fin.get("price_cents"),
                "amount_paid_cents": fin.get("paid_cash_cents", 0),
                "bundle_applied_cents": fin.get("bundle_applied_cents", 0),
                "payment_status": fin.get("payment_status", "unknown"),
            }
        )

        roll["total_appointments"] += 1
        if getattr(a, "attendance_status", None) == "late":
            roll["late_appointments"] += 1
        if fin.get("payment_status") in ("paid", "bundle", "waived", "refunded"):
            roll["paid_appointments"] += 1
        else:
            roll["unpaid_appointments"] += 1

        price_cents = fin.get("price_cents")
        paid_cash = int(fin.get("paid_cash_cents", 0) or 0)
        paid_bundle = int(fin.get("bundle_applied_cents", 0) or 0)
        if price_cents is not None:
            roll["total_expected_cents"] += int(price_cents)
            roll["total_paid_cents"] += min(paid_cash + paid_bundle, int(price_cents))
            roll["total_owed_cents"] += max(
                int(price_cents) - (paid_cash + paid_bundle), 0
            )
        else:
            roll["total_paid_cents"] += paid_cash

    # --- Include Administration Fee charges (rows + totals) ---
    # These are not appointments, so they don't affect the appointment counters.
    # They DO contribute to total_paid_cents and total_owed_cents and appear as rows.
    try:
        from services.admin_fee import list_admin_fee_charges

        fee_rows = list_admin_fee_charges(
            db,
            owner_id=str(owner.id),
            client_account_id=int(acct.id) if acct else None,
        )

        # Filter fees by created_at within the same optional local->UTC bounds
        def _in_bounds(dt) -> bool:
            if not dt:
                return True
            if start_utc_bound and dt < start_utc_bound:
                return False
            if end_utc_bound_excl and dt >= end_utc_bound_excl:
                return False
            return True

        for fee in fee_rows:
            # fee is AdminFeeChargeOut (pydantic); use attributes accordingly
            created = getattr(fee, "created_at", None)
            if not _in_bounds(created):
                continue

            amount = int(getattr(fee, "amount_cents", 0) or 0)
            paid_cash = int(getattr(fee, "paid_cash_cents", 0) or 0)
            paid_bundle = int(getattr(fee, "bundle_applied_cents", 0) or 0)
            paid_total = max(paid_cash + paid_bundle, 0)
            owed = max(amount - paid_total, 0)

            status_val = str(getattr(fee, "status", "")).lower()
            if status_val in ("paid",):
                pstatus = "paid"
            elif status_val in ("bundle",):
                pstatus = "bundle"
            elif status_val in ("waived", "refunded"):
                pstatus = "paid"
                owed = 0
            else:
                if paid_total <= 0:
                    pstatus = "unpaid"
                elif paid_total < amount:
                    pstatus = "partial"
                else:
                    pstatus = "paid"

            out_rows.append(
                {
                    "id": f"fee:{getattr(fee, 'id', '')}",
                    "start_utc": created.isoformat() if created else "",
                    "duration_minutes": 0,
                    "status": "completed",
                    "attendance": None,
                    "price_cents": amount,
                    "amount_paid_cents": paid_cash,
                    "bundle_applied_cents": paid_bundle,
                    "payment_status": pstatus,
                }
            )
            roll["total_paid_cents"] += min(paid_total, amount)
            roll["total_owed_cents"] += owed
    except Exception as _e:
        print(f"[CLIENT_PAYMENTS] admin fee rows error: {getattr(_e, 'message', _e)}")

    return {"summary": roll, "rows": out_rows}


from app.schemas import FinancialSummary
from services.payments import _service_price_map


@router.get("/appointments/summary", response_model=FinancialSummary)
def client_appointments_summary(
    start: date = Query(...),
    end: date = Query(...),
    status: list[Literal["booked", "completed", "canceled"]] | None = Query(None),
    payment_status: list[str] | None = Query(None),
    db: Session = Depends(get_db),
    me: TokenUser = Depends(require_client),
) -> FinancialSummary:
    """Return a financial summary over the client’s appointments.

    Computes totals (expected, paid cash, bundle, owed) for appointments that
    overlap the inclusive date window. Supports optional filtering by
    appointment `status` and computed `payment_status`.

    Args:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD); must be >= start.
        status: Optional status filter.
        payment_status: Optional computed payment status filter.
        db: SQLAlchemy session.
        me: Authenticated client.

    Returns:
        `FinancialSummary` with aggregate totals.
    """
    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")
    owner = _get_owner(db)
    price_map = _service_price_map(db, owner_user_id=owner.id)

    start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = (datetime.combine(end, datetime.min.time()) + timedelta(days=1)).replace(
        tzinfo=timezone.utc
    )

    q = db.query(Appointment).filter(
        Appointment.client_id == me.sub,
        Appointment.end_utc > start_dt,
        Appointment.start_utc < end_dt,
    )
    if status:
        q = q.filter(Appointment.status.in_(status))

    rows = q.order_by(Appointment.start_utc.desc()).all()

    out_rows = []
    from services.payments import compute_financials

    for a in rows:
        fin = compute_financials(db, a, price_map)
        if payment_status and fin["payment_status"] not in set(payment_status):
            continue
        out_rows.append(
            {
                "price_cents": fin["price_cents"],
                "paid_cash_cents": fin["paid_cash_cents"],
                "bundle_applied_cents": fin["bundle_applied_cents"],
                "owed_cents": fin["owed_cents"],
            }
        )

    from services.payments import summarize_financial_rows

    return summarize_financial_rows(
        [
            {
                "price_cents": r["price_cents"],
                "paid_cash_cents": r["paid_cash_cents"],
                "bundle_applied_cents": r["bundle_applied_cents"],
                "owed_cents": r["owed_cents"],
            }
            for r in out_rows
        ]
    )


# ---- Client Wallet (balance + transactions) ----
@router.get("/wallet", response_model=dict)
def client_wallet(
    limit: int = Query(20, ge=1, le=200),
    date_from: date | None = Query(
        None, description="Filter from this date (YYYY-MM-DD) inclusive"
    ),
    date_to: date | None = Query(
        None, description="Filter until this date (YYYY-MM-DD) inclusive"
    ),
    db: Session = Depends(get_db),
    me: TokenUser = Depends(require_client),
) -> dict:
    """
    Returns the current wallet balance and recent transactions for the authenticated client
    under the current owner. Wallets are represented by PrepaidBundle rows where total_credits == 0.
    """
    owner = _get_owner(db)
    acct = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.owner_user_id == owner.id,
            ClientAccount.client_user_id == me.sub,
        )
        .first()
    )
    if not acct:
        return {"balance_cents": 0, "transactions": [], "appointments_count": 0}
    wallet = (
        db.query(PrepaidBundle)
        .filter(
            PrepaidBundle.owner_id == owner.id,
            PrepaidBundle.client_id == me.sub,
            PrepaidBundle.total_credits == 0,
        )
        .order_by(PrepaidBundle.created_at.desc())
        .first()
    )
    if not wallet:
        return {"balance_cents": 0, "transactions": [], "appointments_count": 0}
    balance = int(
        db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
        .filter(PrepaidLedger.bundle_id == wallet.id)
        .scalar()
        or 0
    )
    q = db.query(PrepaidLedger).filter(PrepaidLedger.bundle_id == wallet.id)
    if date_from:
        from datetime import datetime, timezone as _tz

        start_dt = datetime.combine(date_from, datetime.min.time()).replace(
            tzinfo=_tz.utc
        )
        q = q.filter(PrepaidLedger.created_at >= start_dt)
    if date_to:
        from datetime import datetime, timedelta, timezone as _tz

        end_excl = (
            datetime.combine(date_to, datetime.min.time()) + timedelta(days=1)
        ).replace(tzinfo=_tz.utc)
        q = q.filter(PrepaidLedger.created_at < end_excl)
    rows = q.order_by(PrepaidLedger.created_at.desc()).limit(limit).all()
    tx = [
        {
            "event": r.event,
            "amount_cents": int(getattr(r, "amount_cents", 0) or 0),
            "appointment_id": str(getattr(r, "appointment_id", "") or ""),
            "note": getattr(r, "note", None),
            "created_at": r.created_at,
        }
        for r in rows
    ]
    appts_count = int(
        db.query(func.count(Appointment.id))
        .filter(
            Appointment.client_id == me.sub,
            Appointment.owner_id == owner.id,
            Appointment.status.in_(["booked", "completed"]),
        )
        .scalar()
        or 0
    )

    return {
        "balance_cents": balance,
        "transactions": tx,
        "appointments_count": appts_count,
    }
