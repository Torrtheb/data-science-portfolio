from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import (
    AvailabilityRule,
    TimeOff,
    Appointment,
    SpecialOpening,
)


router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])
"""Router containing non-production debug endpoints for scheduling."""


@router.delete("/_debug/purge-owner-data")
def purge_owner_data(
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict:
    """Purge all scheduling objects for the authenticated owner.

    Deletes in FK-safe order: appointments, time off, special openings, and
    weekly availability rules. Intended for development/testing only.

    Returns:
        Dict with counts of deleted rows per table and 'ok: True'.
    """
    appts = (
        db.query(Appointment)
        .filter(Appointment.owner_id == user.sub)
        .delete(synchronize_session=False)
    )
    offs = (
        db.query(TimeOff)
        .filter(TimeOff.owner_id == user.sub)
        .delete(synchronize_session=False)
    )
    opens = (
        db.query(SpecialOpening)
        .filter(SpecialOpening.owner_id == user.sub)
        .delete(synchronize_session=False)
    )
    rules = (
        db.query(AvailabilityRule)
        .filter(AvailabilityRule.owner_id == user.sub)
        .delete(synchronize_session=False)
    )
    db.commit()
    return {
        "ok": True,
        "deleted": {
            "appointments": appts,
            "timeoffs": offs,
            "openings": opens,
            "rules": rules,
        },
    }
