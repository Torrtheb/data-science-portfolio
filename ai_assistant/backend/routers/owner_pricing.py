from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Sequence
from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import ServiceOption
from app.schemas import (
    AdminFeeChargeCreate,
    AdminFeeChargeOut,
    AdminFeeChargeUpdate,
    AdminFeeSettingsIn,
    AdminFeeSettingsOut,
    ServiceOptionCreate,
    ServiceOptionOut,
)
from services.admin_fee import (
    create_admin_fee_charge,
    get_admin_fee_setting,
    list_admin_fee_charges,
    set_admin_fee_setting,
    update_admin_fee_charge,
    delete_admin_fee_charge,
)

router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])


@router.get("/owner/service-options", response_model=List[ServiceOptionOut])
def get_service_options(
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Return all service options for the owner sorted by duration."""
    return (
        db.query(ServiceOption)
        .filter(ServiceOption.owner_id == user.sub)
        .order_by(ServiceOption.duration_minutes.asc())
        .all()
    )


@router.put("/owner/service-options", response_model=List[ServiceOptionOut])
def replace_service_options(
    payload: List[ServiceOptionCreate],
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Replace all service options for the owner with the provided list.

    Validates duration against the allowed set before inserting. Marks
    'is_active' based on the payload flag.
    """
    db.query(ServiceOption).filter(ServiceOption.owner_id == user.sub).delete(
        synchronize_session=False
    )
    allowed = {15, 30, 45, 60, 120}
    for p in payload:
        if p.duration_minutes not in allowed:
            raise HTTPException(
                400, f"duration_minutes must be one of {sorted(allowed)}"
            )
        db.add(
            ServiceOption(
                owner_id=user.sub,
                duration_minutes=p.duration_minutes,
                price_cents=p.price_cents,
                currency=p.currency,
                is_active=1 if p.is_active else 0,
            )
        )
    db.commit()
    return (
        db.query(ServiceOption)
        .filter(ServiceOption.owner_id == user.sub)
        .order_by(ServiceOption.duration_minutes.asc())
        .all()
    )


@router.patch("/owner/service-options/{option_id}", response_model=ServiceOptionOut)
def update_one_service_option(
    option_id: int,
    patch: ServiceOptionCreate,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Update a single service option for the owner by id."""
    row = (
        db.query(ServiceOption)
        .filter(ServiceOption.id == option_id, ServiceOption.owner_id == user.sub)
        .first()
    )
    if not row:
        raise HTTPException(404, "Service option not found")
    allowed = {15, 30, 45, 60, 120}
    if patch.duration_minutes not in allowed:
        raise HTTPException(400, f"duration_minutes must be one of {sorted(allowed)}")
    row.duration_minutes = patch.duration_minutes
    row.price_cents = patch.price_cents
    row.currency = patch.currency
    row.is_active = 1 if patch.is_active else 0
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.delete("/owner/service-options/{option_id}", response_model=dict)
def delete_one_service_option(
    option_id: int,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Delete a service option by id for the current owner."""
    n = (
        db.query(ServiceOption)
        .filter(ServiceOption.id == option_id, ServiceOption.owner_id == user.sub)
        .delete(synchronize_session=False)
    )
    db.commit()
    if n == 0:
        raise HTTPException(404, "Service option not found")
    return {"ok": True}


@router.get("/owner/admin-fee", response_model=AdminFeeSettingsOut)
def get_admin_fee_settings(
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Return the current administration fee setting for the owner."""
    return get_admin_fee_setting(db, user.sub)


@router.put("/owner/admin-fee", response_model=AdminFeeSettingsOut)
def update_admin_fee_settings(
    payload: AdminFeeSettingsIn,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Update the administration fee amount (in cents)."""
    try:
        return set_admin_fee_setting(db, user.sub, payload.admin_fee_cents)
    except Exception as exc:
        raise HTTPException(400, f"Failed to update administration fee: {exc}")


@router.get("/owner/admin-fee/charges", response_model=List[AdminFeeChargeOut])
def list_admin_fee_charges_endpoint(
    status: Optional[Sequence[str]] = Query(None),
    client_account_id: Optional[int] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=200),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """List administration fee charges with optional filters and limit."""
    try:
        return list_admin_fee_charges(
            db,
            owner_id=user.sub,
            status=status,
            client_account_id=client_account_id,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(400, f"Failed to list administration fee charges: {exc}")


@router.post("/owner/admin-fee/charges", response_model=AdminFeeChargeOut)
def create_admin_fee_charge_endpoint(
    payload: AdminFeeChargeCreate,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Create a new administration fee charge for a client account."""
    try:
        return create_admin_fee_charge(
            db,
            owner_id=user.sub,
            client_account_id=payload.client_account_id,
            amount_cents=payload.amount_cents,
            note=payload.note,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Failed to create administration fee charge: {exc}")


@router.put("/owner/admin-fee/charges/{charge_id}", response_model=AdminFeeChargeOut)
def update_admin_fee_charge_endpoint(
    charge_id: int,
    payload: AdminFeeChargeUpdate,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Update fields on an administration fee charge (status, cash paid, note)."""
    try:
        return update_admin_fee_charge(
            db,
            owner_id=user.sub,
            charge_id=charge_id,
            status=payload.status,
            paid_cash_cents=payload.paid_cash_cents,
            note=payload.note,
            apply_wallet=payload.apply_wallet,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Failed to update administration fee charge: {exc}")


@router.delete("/owner/admin-fee/charges/{charge_id}", response_model=dict)
def delete_admin_fee_charge_endpoint(
    charge_id: int,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Delete an administration fee charge when allowed by business rules."""
    try:
        delete_admin_fee_charge(db, owner_id=user.sub, charge_id=int(charge_id))
        return {"ok": True}
    except ValueError as exc:
        msg = str(exc)
        code = 404 if msg == "Charge not found" else 400
        raise HTTPException(code, msg)
    except Exception as exc:
        raise HTTPException(400, f"Failed to delete administration fee charge: {exc}")
