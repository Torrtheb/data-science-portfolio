from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import (
    User,
    AvailabilityRule,
    Appointment,
    RoleEnum,
)
from app.schemas import (
    BookRequest,
)
from ._helpers import (
    UTC,
    uuid_str,
    send_email,
    build_appt_email,
)
from services.services_scheduling import (
    generate_daily_slots,
)

from services.payments import get_default_price_cents


router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])
"""Router for admin-side scheduling actions (owner-authenticated)."""


@router.post("/book", response_model=dict)
def book(
    payload: BookRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict:
    """Create an appointment on behalf of a client.

    Behavior:
    - Ensures the authenticated owner exists.
    - Ensures the client user exists (creates with CLIENT role if missing).
    - Normalizes the requested start to the owner’s timezone and derives slot
      length from the containing weekly rule (fallback 30 minutes).
    - Revalidates availability just-in-time using generated daily slots.
    - Inserts the appointment in UTC, sets default price and unpaid status.
    - Schedules a confirmation email to the client (with ICS).

    Args:
        payload: Booking request containing client identity, start_local, message.
        background_tasks: FastAPI background task runner for email send.
        db: SQLAlchemy session.
        user: Authenticated owner.

    Returns:
        Dict with `ok: True` and `appointment_id` on success.

    Raises:
        HTTPException: 404 when owner missing; 400 for invalid email or times;
        409 when slot is unavailable or double-booked.
    """
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")

    if "@" not in (payload.client_email or ""):
        raise HTTPException(400, "client_email must be a valid email")

    # 1) ensure client user exists
    client = db.query(User).filter(User.email == payload.client_email).first()
    if not client:
        client = User(
            id=uuid_str(),
            name=payload.client_name,
            email=payload.client_email,
            role=RoleEnum.CLIENT,
            timezone=owner.timezone,
            createdAt=datetime.utcnow(),
            updatedAt=datetime.utcnow(),
        )
        db.add(client)
        db.commit()
        db.refresh(client)

    # 2) normalize start time to owner tz
    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )

    # 3) figure slot length from containing rule (fallback 30)
    rules = (
        db.query(AvailabilityRule)
        .filter_by(owner_id=owner.id, weekday=start_local.weekday())
        .all()
    )

    def _local_dt(d, t):
        return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=owner_tz)

    containing_rule = next(
        (
            r
            for r in rules
            if _local_dt(start_local.date(), r.start_local)
            <= start_local
            < _local_dt(start_local.date(), r.end_local)
        ),
        None,
    )
    slot_minutes = containing_rule.slot_minutes if containing_rule else 30
    end_local = start_local + timedelta(minutes=slot_minutes)

    # 4) re-check availability just in time
    slots_today = [
        (s, e) for s, e in generate_daily_slots(db, owner, start_local.date())
    ]
    if (start_local, end_local) not in slots_today:
        raise HTTPException(409, "Slot is no longer available")

    # 5) write appointment in UTC
    appt = Appointment(
        id=uuid_str(),
        owner_id=owner.id,
        client_id=client.id,
        client_name=payload.client_name or client.name or client.email,
        client_email=client.email,
        start_utc=start_local.astimezone(UTC),
        end_utc=end_local.astimezone(UTC),
        status="booked",
    )
    db.add(appt)
    duration_minutes = int((end_local - start_local).total_seconds() // 60)
    appt.price_override_cents = get_default_price_cents(
        db, owner_user_id=owner.id, duration_minutes=duration_minutes
    )
    appt.payment_status = "unpaid"
    appt.amount_paid_cents = appt.amount_paid_cents or 0
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Double-booking detected, please pick another slot")

    # 6) email (OWNER acted -> notify CLIENT)
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

    return {"ok": True, "appointment_id": appt.id}
