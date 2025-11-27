from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Response, Query
from pydantic import BaseModel
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import json
import sqlalchemy as sa
from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import (
    User,
    TimeOff,
    Appointment,
)
from app.schemas import (
    TimeOffCreate,
    TimeOffOut,
)
from ._helpers import UTC, uuid_str, resolve_tz

router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])
"""Router for owner-facing time-off endpoints."""

NONCANCELLED_STATUSES = {"booked", "confirmed", "pending"}


def _pretty_local(dt_utc: datetime, tz_name: str) -> str:
    """Format a UTC datetime into a short owner-local string for logs/UI."""
    tz = ZoneInfo(tz_name)
    return dt_utc.astimezone(tz).strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


@router.get("/timeoff", response_model=list[TimeOffOut])
def list_timeoff(
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
    tz: str | None = Query(None, description="IANA timezone for local timestamps"),
):
    """List time-off windows for the owner with owner-local fields included."""
    owner = db.query(User).filter(User.id == user.sub).first()
    display_tz = resolve_tz(tz, getattr(owner, "timezone", None) or "America/Toronto")
    tz_name = getattr(display_tz, "key", str(display_tz))

    rows = (
        db.query(TimeOff)
        .filter_by(owner_id=user.sub)
        .order_by(TimeOff.start_utc.asc())
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
            "note": row.note,
        }
        for row in rows
    ]


@router.post("/timeoff", response_model=TimeOffOut)
def add_timeoff(
    payload: TimeOffCreate,
    confirm_if_conflicts: bool = Query(
        False,
        description="Acknowledge conflicts; still requires moving/canceling appointments first",
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Create a time-off window; blocks if overlaps active appointments.

    If overlaps are detected and 'confirm_if_conflicts' is false, returns a
    '409 CONFIRM_REQUIRED' with a machine-parsable 'pending_http' body to
    allow explicit confirmation by the caller.
    """
    start_utc = payload.start.astimezone(UTC)
    end_utc = payload.end.astimezone(UTC)
    if end_utc <= start_utc:
        raise HTTPException(400, "end must be after start")

    # 1) BLOCK on overlapping active appointments (no auto-cancel)
    owner = db.query(User).filter(User.id == user.sub).first()
    owner_tz = owner.timezone if owner and owner.timezone else "America/Toronto"

    conflicting_appointments = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == user.sub,
            Appointment.start_utc < end_utc,
            Appointment.end_utc > start_utc,
            sa.or_(
                Appointment.status.is_(None),
                Appointment.status.in_(NONCANCELLED_STATUSES),
            ),
        )
        .order_by(Appointment.start_utc.asc())
        .all()
    )
    if conflicting_appointments and not confirm_if_conflicts:
        preview = "; ".join(
            f"{_pretty_local(a.start_utc, owner_tz)} → {_pretty_local(a.end_utc, owner_tz)}"
            for a in conflicting_appointments[:5]
        )
        more = (
            f" (+{len(conflicting_appointments) - 5} more)"
            if len(conflicting_appointments) > 5
            else ""
        )
        detail = {
            "human": (
                "Time off overlaps existing appointments. Cancel or move those appointments first."
            ),
            "conflicts": [preview + more] if preview else [],
            "pending_http": {
                "endpoint": "/api/scheduling/timeoff",
                "method": "POST",
                "body": {
                    **payload.model_dump(mode="json"),
                    "confirm_if_conflicts": True,
                },
            },
        }
        raise HTTPException(409, detail="CONFIRM_REQUIRED:" + json.dumps(detail))

    # 2) LATEST-WINS for overlapping time-off blocks (safe to keep)
    overlapping_timeoffs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == user.sub,
            TimeOff.start_utc < end_utc,
            TimeOff.end_utc > start_utc,
        )
        .all()
    )
    for timeoff in overlapping_timeoffs:
        db.delete(timeoff)

    # 3) Exact-match fast-path
    exact_match = (
        db.query(TimeOff).filter_by(owner_id=user.sub, start_utc=start_utc).first()
    )
    if exact_match:
        return exact_match

    off = TimeOff(
        id=uuid_str(),
        owner_id=user.sub,
        start_utc=start_utc,
        end_utc=end_utc,
        note=payload.note,
    )
    db.add(off)
    try:
        db.commit()
        db.refresh(off)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Database constraint error creating time-off")
    return off


@router.delete("/timeoff/{timeoff_id}", status_code=204)
def delete_timeoff(
    timeoff_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Delete a time-off window by id for the owner."""
    t = (
        db.query(TimeOff)
        .filter(TimeOff.id == timeoff_id, TimeOff.owner_id == user.sub)
        .first()
    )
    if not t:
        # Matches your frontend's observed 404 {"detail":"Not Found"}
        raise HTTPException(status_code=404, detail="Not Found")
    db.delete(t)
    db.commit()
    return Response(status_code=204)


class UpdateTimeOffPayload(BaseModel):
    """Partial update payload for a time-off window."""

    start: datetime | None = None
    end: datetime | None = None
    note: str | None = None


@router.put("/timeoff/{timeoff_id}", response_model=TimeOffOut)
def update_timeoff(
    timeoff_id: str,
    payload: UpdateTimeOffPayload,
    confirm_if_conflicts: bool = Query(
        False,
        description="Acknowledge conflicts; still requires moving/canceling appointments first",
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Update time-off window; enforces no overlap with active appointments.

    On conflicts and without confirmation, returns '409 CONFIRM_REQUIRED' with
    a replayable 'pending_http' body to allow explicit confirmation.
    """
    t = db.query(TimeOff).filter_by(id=timeoff_id, owner_id=user.sub).first()
    if not t:
        raise HTTPException(404, "Time off not found")

    owner = db.query(User).filter(User.id == user.sub).first()
    owner_tz = (
        ZoneInfo(owner.timezone)
        if owner and owner.timezone
        else ZoneInfo("America/Toronto")
    )
    owner_tz_name = (
        owner_tz.key if isinstance(owner_tz, ZoneInfo) else "America/Toronto"
    )
    start_local = (
        payload.start.replace(tzinfo=owner_tz)
        if payload.start and payload.start.tzinfo is None
        else (
            payload.start.astimezone(owner_tz)
            if payload.start
            else t.start_utc.astimezone(owner_tz)
        )
    )
    end_local = (
        payload.end.replace(tzinfo=owner_tz)
        if payload.end and payload.end.tzinfo is None
        else (
            payload.end.astimezone(owner_tz)
            if payload.end
            else t.end_utc.astimezone(owner_tz)
        )
    )
    if end_local <= start_local:
        raise HTTPException(400, "end must be after start")

    new_start_utc, new_end_utc = start_local.astimezone(UTC), end_local.astimezone(UTC)
    conflicting_appointments = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == user.sub,
            Appointment.start_utc < new_end_utc,
            Appointment.end_utc > new_start_utc,
            sa.or_(
                Appointment.status.is_(None),
                Appointment.status.in_(NONCANCELLED_STATUSES),
            ),
        )
        .order_by(Appointment.start_utc.asc())
        .all()
    )
    if conflicting_appointments and not confirm_if_conflicts:
        preview = "; ".join(
            f"{_pretty_local(a.start_utc, owner_tz_name)} → {_pretty_local(a.end_utc, owner_tz_name)}"
            for a in conflicting_appointments[:5]
        )
        more = (
            f" (+{len(conflicting_appointments) - 5} more)"
            if len(conflicting_appointments) > 5
            else ""
        )
        detail = {
            "human": (
                "Time off overlaps existing appointments. Cancel or move those appointments first."
            ),
            "conflicts": [preview + more] if preview else [],
            "pending_http": {
                "endpoint": f"/api/scheduling/timeoff/{timeoff_id}",
                "method": "PUT",
                "body": {
                    **(payload.model_dump(mode="json") if payload else {}),
                    "confirm_if_conflicts": True,
                },
            },
        }
        raise HTTPException(409, detail="CONFIRM_REQUIRED:" + json.dumps(detail))

    # Persist
    t.start_utc, t.end_utc = new_start_utc, new_end_utc
    if payload.note is not None:
        t.note = payload.note

    db.commit()
    db.refresh(t)
    return t
