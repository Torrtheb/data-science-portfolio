from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Response, Query
from pydantic import BaseModel
from datetime import datetime, timedelta, date, time
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from typing import Any, Optional
from sqlalchemy.exc import IntegrityError
from uuid import UUID  # add
import re  # add

from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import (
    User,
    AvailabilityRule,
    TimeOff,
    SpecialOpening,
)
from app.schemas import (
    SpecialOpeningCreate,
    SpecialOpeningOut,
)
from ._helpers import UTC, uuid_str, resolve_tz

router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])

DISPLAY_OPENING_RE = re.compile(r"^open-(.+)$")


def _resolve_opening_from_path(
    db: Session,
    owner_id: str,
    opening_id_path: str,
    owner_tz: str,
) -> Optional[SpecialOpening]:
    """Resolve a human-friendly opening identifier into a SpecialOpening.

    Supports the following forms:
    - Exact UUID string → that opening.
    - "open-<uuid>-<anything>" → extracts the UUID and resolves that opening.
    - "open-<n>" → resolves to the nth opening today (0-based), ordered by start.
    """
    s = (opening_id_path or "").strip()

    # 1) direct UUID match
    try:
        UUID(s)
        row = db.query(SpecialOpening).filter_by(owner_id=owner_id, id=s).first()
        if row:
            return row
    except Exception:
        pass

    # 2) 'open-<...>' variants
    m = DISPLAY_OPENING_RE.match(s)
    if m:
        tail = m.group(1)

        # 2a) If it starts with a UUID (36 chars, contains dashes), strip it out
        maybe_uuid = tail[:36]
        try:
            UUID(maybe_uuid)
            row = (
                db.query(SpecialOpening)
                .filter_by(owner_id=owner_id, id=maybe_uuid)
                .first()
            )
            if row:
                return row
        except Exception:
            pass

        # 2b) If it's a pure index like 'open-0' / 'open-3' → map to today's nth opening
        if tail.isdigit():
            idx = int(tail)
            tz = ZoneInfo(owner_tz)
            today = datetime.now(tz).date()
            day_start_local = datetime(
                today.year, today.month, today.day, 0, 0, tzinfo=tz
            )
            day_end_local = datetime(
                today.year, today.month, today.day, 23, 59, tzinfo=tz
            )
            from ._helpers import UTC

            u_start = day_start_local.astimezone(UTC)
            u_end = day_end_local.astimezone(UTC)
            rows = (
                db.query(SpecialOpening)
                .filter(
                    SpecialOpening.owner_id == owner_id,
                    SpecialOpening.start_utc < u_end,
                    SpecialOpening.end_utc > u_start,
                )
                .order_by(SpecialOpening.start_utc.asc())
                .all()
            )
            if 0 <= idx < len(rows):
                return rows[idx]

    # 3) last chance: literal string match (if ids stored as text in other shapes)
    row = db.query(SpecialOpening).filter_by(owner_id=owner_id, id=s).first()
    return row


class UpdateOpeningPayload(BaseModel):
    """Partial update shape for a SpecialOpening.

    Times are owner-local (naive or tz-aware); unspecified fields are left
    unchanged. Set 'allow_overlap=True' to skip conflict checks.
    """

    start: datetime | None = None
    end: datetime | None = None
    slot_minutes: int | None = None
    buffer_minutes: int | None = None
    note: str | None = None
    allow_overlap: bool = False


# -----------------------------------------------------------------------------
# One-off Openings (absolute window, stored in UTC)
# -----------------------------------------------------------------------------


@router.get("/openings", response_model=list[SpecialOpeningOut])
def list_openings(
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
    tz: str | None = Query(None, description="IANA timezone for local timestamps"),
) -> list[dict[str, Any]]:
    """List the owner's one-off openings, including owner-local timestamps."""
    owner = db.query(User).filter(User.id == user.sub).first()
    display_tz = resolve_tz(tz, getattr(owner, "timezone", None) or "America/Toronto")
    tz_name = getattr(display_tz, "key", str(display_tz))

    rows = (
        db.query(SpecialOpening)
        .filter_by(owner_id=user.sub)
        .order_by(SpecialOpening.start_utc.asc())
        .all()
    )
    return [
        {
            "id": row.id,
            "start_utc": row.start_utc,
            "end_utc": row.end_utc,
            "start_local": row.start_utc.astimezone(display_tz),
            "end_local": row.end_utc.astimezone(display_tz),
            "timezone": tz_name,
            "slot_minutes": row.slot_minutes,
            "buffer_minutes": row.buffer_minutes,
            "note": row.note,
        }
        for row in rows
    ]


@router.post("/openings", response_model=SpecialOpeningOut, status_code=201)
def add_opening(
    payload: SpecialOpeningCreate,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict[str, Any]:
    """Create a new one-off opening. Prevents overlap with time off and weekly rules."""
    start_utc = payload.start.astimezone(UTC)
    end_utc = payload.end.astimezone(UTC)
    if end_utc <= start_utc:
        raise HTTPException(400, "end must be after start")
    off = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == user.sub,
            TimeOff.start_utc < end_utc,
            TimeOff.end_utc > start_utc,
        )
        .first()
    )
    if off:
        raise HTTPException(409, "This window overlaps time off.")
    if not payload.allow_overlap:
        owner = db.query(User).filter(User.id == user.sub).first()
        owner_tz = resolve_tz(
            None, getattr(owner, "timezone", None) or "America/Toronto"
        )
        sl = start_utc.astimezone(owner_tz)
        el = end_utc.astimezone(owner_tz)
        cur = sl
        while cur.date() <= el.date():
            day_start = cur.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start.replace(hour=23, minute=59)
            span_start = max(cur, day_start)
            span_end = min(el, day_end)
            if span_end > span_start:
                weekday_mon0 = span_start.weekday()
                start_min = span_start.hour * 60 + span_start.minute
                end_min = span_end.hour * 60 + span_end.minute
                rules = (
                    db.query(AvailabilityRule)
                    .filter(
                        AvailabilityRule.owner_id == user.sub,
                        AvailabilityRule.weekday == weekday_mon0,
                        AvailabilityRule.start_local
                        < time(hour=end_min // 60, minute=end_min % 60),
                        AvailabilityRule.end_local
                        > time(hour=start_min // 60, minute=start_min % 60),
                    )
                    .all()
                )
                if rules:
                    raise HTTPException(
                        409, "This window overlaps weekly availability."
                    )
            cur = day_start + timedelta(days=1)
    overlapping_openings = (
        db.query(SpecialOpening)
        .filter(
            SpecialOpening.owner_id == user.sub,
            SpecialOpening.start_utc < end_utc,
            SpecialOpening.end_utc > start_utc,
        )
        .all()
    )
    for opening in overlapping_openings:
        db.delete(opening)

    o = SpecialOpening(
        id=uuid_str(),
        owner_id=user.sub,
        start_utc=start_utc,
        end_utc=end_utc,
        slot_minutes=payload.slot_minutes,
        buffer_minutes=payload.buffer_minutes,
        note=payload.note,
    )
    db.add(o)
    try:
        db.commit()
        db.refresh(o)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "This opening overlaps another opening")
    return o


# -----------------------------------------------------------------------------
# Recurring weekly openings (create as individual one-off specials)
# -----------------------------------------------------------------------------


class RecurringOpeningsCreate(BaseModel):
    """Request body for creating weekly recurring openings as individual rows."""

    weekday: int
    start_hhmm: str
    end_hhmm: str
    slot_minutes: int
    buffer_minutes: int = 0
    weeks: int = 8
    start_date: Optional[date] = None
    note: Optional[str] = None


def _parse_hhmm_to_time(hhmm: str) -> time:
    """Parse "HH:MM" into a time or raise HTTP 400 if invalid."""
    s = (hhmm or "").strip()
    parts = s.split(":")
    if len(parts) < 2:
        raise HTTPException(400, "Invalid time; expected HH:MM")
    try:
        h, m = int(parts[0]), int(parts[1])
    except Exception:
        raise HTTPException(400, "Invalid time; expected HH:MM")
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise HTTPException(400, "Invalid time; expected HH:MM")
    return time(h, m)


@router.post(
    "/openings/recurring", response_model=list[SpecialOpeningOut], status_code=201
)
def add_recurring_openings(
    payload: RecurringOpeningsCreate,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
    tz: Optional[str] = Query(None),
) -> list[SpecialOpening]:
    """
    Create N weekly openings as individual SpecialOpening rows, so each can be edited independently.

    - Anchor by owner-local weekday + HH:MM range.
    - First occurrence = the next date on/after start_date that matches weekday.
    - For each occurrence, carve through TimeOff and merge touching/overlapping specials.
    """
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(401, "Owner not found")
    owner_tz = ZoneInfo(tz or owner.timezone)

    if not (0 <= int(payload.weekday) <= 6):
        raise HTTPException(400, "weekday must be 0..6 (Mon=0..Sun=6)")

    start_t = _parse_hhmm_to_time(payload.start_hhmm)
    end_t = _parse_hhmm_to_time(payload.end_hhmm)
    if end_t <= start_t:
        raise HTTPException(400, "end_hhmm must be after start_hhmm")
    if payload.weeks <= 0:
        raise HTTPException(400, "weeks must be >= 1")
    base = payload.start_date or datetime.now(owner_tz).date()
    delta = (int(payload.weekday) - base.weekday()) % 7
    first_day = base if delta == 0 else (base + timedelta(days=delta))

    created: list[SpecialOpening] = []
    from services.services_scheduling import (
        carve_opening_through_timeoff,
        merge_or_get_special_opening,
    )

    for i in range(int(payload.weeks)):
        d = first_day + timedelta(days=7 * i)
        start_local = datetime(
            d.year, d.month, d.day, start_t.hour, start_t.minute, tzinfo=owner_tz
        )
        end_local = datetime(
            d.year, d.month, d.day, end_t.hour, end_t.minute, tzinfo=owner_tz
        )
        start_utc = start_local.astimezone(UTC)
        end_utc = end_local.astimezone(UTC)
        carve_opening_through_timeoff(db, owner.id, start_utc, end_utc)
        sp = merge_or_get_special_opening(
            db,
            owner.id,
            start_utc,
            end_utc,
            slot_minutes=int(payload.slot_minutes),
            buffer_minutes=int(payload.buffer_minutes or 0),
            note=payload.note or "Weekly opening",
        )
        created.append(sp)
    db.commit()
    ids = [c.id for c in created]
    rows = (
        db.query(SpecialOpening)
        .filter(SpecialOpening.owner_id == owner.id, SpecialOpening.id.in_(ids))
        .order_by(SpecialOpening.start_utc.asc())
        .all()
    )
    return rows


@router.delete("/openings/{opening_id}", status_code=204)
def delete_opening(
    opening_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> Response:
    """Delete an opening by id or display token."""
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(401, "Owner not found")

    o = _resolve_opening_from_path(
        db, owner_id=user.sub, opening_id_path=opening_id, owner_tz=owner.timezone
    )
    if not o:
        raise HTTPException(404, "Opening not found")

    db.delete(o)
    db.commit()
    return Response(status_code=204)


@router.put("/openings/{opening_id}", response_model=SpecialOpeningOut)
def update_opening(
    opening_id: str,
    payload: UpdateOpeningPayload,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> SpecialOpening:
    """Update a one-off opening, with optional overlap checks against time off and other openings."""
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(401, "Owner not found")
    owner_tz = ZoneInfo(owner.timezone)

    o = _resolve_opening_from_path(
        db, owner_id=user.sub, opening_id_path=opening_id, owner_tz=owner.timezone
    )
    if not o:
        raise HTTPException(404, "Opening not found")
    start_local = (
        payload.start.replace(tzinfo=owner_tz)
        if payload.start and payload.start.tzinfo is None
        else (
            payload.start.astimezone(owner_tz)
            if payload.start
            else o.start_utc.astimezone(owner_tz)
        )
    )
    end_local = (
        payload.end.replace(tzinfo=owner_tz)
        if payload.end and payload.end.tzinfo is None
        else (
            payload.end.astimezone(owner_tz)
            if payload.end
            else o.end_utc.astimezone(owner_tz)
        )
    )
    if end_local <= start_local:
        raise HTTPException(400, "end must be after start")

    new_start_utc, new_end_utc = start_local.astimezone(UTC), end_local.astimezone(UTC)
    if not payload.allow_overlap:
        off = (
            db.query(TimeOff)
            .filter(
                TimeOff.owner_id == owner.id,
                TimeOff.start_utc < new_end_utc,
                TimeOff.end_utc > new_start_utc,
            )
            .first()
        )
        if off:
            raise HTTPException(409, "This window overlaps time off.")
        other = (
            db.query(SpecialOpening)
            .filter(
                SpecialOpening.owner_id == owner.id,
                SpecialOpening.id != o.id,
                SpecialOpening.start_utc < new_end_utc,
                SpecialOpening.end_utc > new_start_utc,
            )
            .first()
        )
        if other:
            raise HTTPException(409, "This window overlaps another opening.")
    o.start_utc, o.end_utc = new_start_utc, new_end_utc
    if payload.slot_minutes is not None:
        o.slot_minutes = payload.slot_minutes
    if payload.buffer_minutes is not None:
        o.buffer_minutes = payload.buffer_minutes
    if payload.note is not None:
        o.note = payload.note

    db.commit()
    db.refresh(o)
    return o
