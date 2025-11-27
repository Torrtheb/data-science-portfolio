from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Any, Dict

from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import (
    AvailabilityRule,
)
from app.schemas import (
    AvailabilityCreate,
    AvailabilityOut,
)
from ._helpers import uuid_str

router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])


# -----------------------------------------------------------------------------
# Availability (weekly, local wall-clock)
# -----------------------------------------------------------------------------


@router.get("/availability", response_model=list[AvailabilityOut])
def list_availability(
    db: Session = Depends(get_db), user: TokenUser = Depends(require_owner)
) -> list[AvailabilityOut]:
    """List all weekly availability rules for the authenticated owner."""
    rules = db.query(AvailabilityRule).filter_by(owner_id=user.sub).all()
    return [AvailabilityOut.from_db_model(rule) for rule in rules]


@router.post("/availability", response_model=AvailabilityOut)
def add_availability(
    payload: AvailabilityCreate,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> AvailabilityOut:
    """Create a weekly availability rule for the owner.

    Enforces simple validation (end after start) and applies a "latest wins"
    policy by deleting overlapping rules on the same weekday before insert.

    Raises:
        HTTPException: 400 when 'end_local <= start_local'; 409 on constraint issues.
    """
    start_time = payload.get_start_time()
    end_time = payload.get_end_time()
    if end_time <= start_time:
        raise HTTPException(400, "end_local must be after start_local")
    exact_match = (
        db.query(AvailabilityRule)
        .filter_by(
            owner_id=user.sub,
            weekday=payload.weekday,
            start_local=start_time,
            end_local=end_time,
            slot_minutes=payload.slot_minutes,
            buffer_minutes=payload.buffer_minutes,
        )
        .first()
    )
    if exact_match:
        return AvailabilityOut.from_db_model(exact_match)
    overlapping_rules = (
        db.query(AvailabilityRule)
        .filter(
            AvailabilityRule.owner_id == user.sub,
            AvailabilityRule.weekday == payload.weekday,
            AvailabilityRule.start_local < end_time,
            AvailabilityRule.end_local > start_time,
        )
        .all()
    )
    for rule in overlapping_rules:
        db.delete(rule)

    rule = AvailabilityRule(
        id=uuid_str(),
        owner_id=user.sub,
        weekday=payload.weekday,
        start_local=start_time,
        end_local=end_time,
        slot_minutes=payload.slot_minutes,
        buffer_minutes=payload.buffer_minutes,
    )
    db.add(rule)
    try:
        db.commit()
        db.refresh(rule)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Database constraint error creating availability rule")
    return AvailabilityOut.from_db_model(rule)


@router.delete("/availability/{rule_id}", status_code=204)
def delete_availability(
    rule_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> Response:
    """Delete a single availability rule by id for the owner.

    Raises:
        HTTPException: 404 if the rule does not exist for this owner.
    """
    r = db.query(AvailabilityRule).filter_by(id=rule_id, owner_id=user.sub).first()
    if not r:
        raise HTTPException(404, "Availability rule not found")
    db.delete(r)
    db.commit()
    return Response(status_code=204)


@router.get("/availability/debug/all")
def debug_list_all_availability(
    db: Session = Depends(get_db), user: TokenUser = Depends(require_owner)
) -> list[Dict[str, Any]]:
    """Debug endpoint: list all rules with raw fields for the owner."""
    rows = (
        db.query(AvailabilityRule)
        .filter(AvailabilityRule.owner_id == user.sub)
        .order_by(AvailabilityRule.weekday.asc(), AvailabilityRule.start_local.asc())
        .all()
    )
    return [
        {
            "id": r.id,
            "weekday": r.weekday,
            "start_local": str(r.start_local),
            "end_local": str(r.end_local),
            "slot_minutes": r.slot_minutes,
            "buffer_minutes": r.buffer_minutes,
        }
        for r in rows
    ]


@router.delete("/availability")
def bulk_delete_availability(
    weekday: int | None = None,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> Dict[str, int | bool]:
    """Bulk delete availability rules for the owner.

    If 'weekday' is provided (0..6), only rules for that weekday are deleted.

    Raises:
        HTTPException: 400 when 'weekday' is out of range.
    """
    q = db.query(AvailabilityRule).filter(AvailabilityRule.owner_id == user.sub)
    if weekday is not None:
        if weekday < 0 or weekday > 6:
            raise HTTPException(400, "weekday must be 0..6")
        q = q.filter(AvailabilityRule.weekday == weekday)
    count = q.count()
    if count == 0:
        return {"ok": True, "deleted": 0}
    q.delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": count}
