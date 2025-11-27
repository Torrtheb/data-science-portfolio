from __future__ import annotations
from typing import Optional, Sequence
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.models import (
    AdminFeeCharge,
    AdminFeeStatus,
    ClientAccount,
    OwnerFeeSetting,
    PrepaidBundle,
    PrepaidLedger,
    User,
)
from app.schemas import (
    AdminFeeChargeOut,
    AdminFeeSettingsOut,
)
from services.wallets import auto_apply_wallet_funds
from services.emailer import send_email as _send_email, render_basic_html
from services.services_scheduling import _account_primary_email

DEFAULT_ADMIN_FEE_CENTS = 1500


def get_admin_fee_setting(db: Session, owner_id: str) -> AdminFeeSettingsOut:
    """Fetch the current admin-fee amount configured for an owner.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner identifier.

    Returns:
        'AdminFeeSettingsOut' containing the fee amount in cents. Falls back to
        'DEFAULT_ADMIN_FEE_CENTS' when no setting is stored.
    """
    row = db.query(OwnerFeeSetting).filter(OwnerFeeSetting.owner_id == owner_id).first()
    cents = int(
        getattr(row, "admin_fee_cents", DEFAULT_ADMIN_FEE_CENTS)
        or DEFAULT_ADMIN_FEE_CENTS
    )
    return AdminFeeSettingsOut(admin_fee_cents=cents)


def set_admin_fee_setting(
    db: Session, owner_id: str, amount_cents: int
) -> AdminFeeSettingsOut:
    """Create or update the admin-fee amount for an owner.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner identifier.
        amount_cents: New fee amount (will be clamped to >= 0).

    Returns:
        Persisted 'AdminFeeSettingsOut' reflecting the stored amount.
    """
    amount = max(int(amount_cents), 0)
    row = db.query(OwnerFeeSetting).filter(OwnerFeeSetting.owner_id == owner_id).first()
    if row:
        row.admin_fee_cents = amount
        db.add(row)
    else:
        row = OwnerFeeSetting(owner_id=owner_id, admin_fee_cents=amount)
        db.add(row)
    db.commit()
    db.refresh(row)
    return AdminFeeSettingsOut(admin_fee_cents=int(row.admin_fee_cents))


def _client_label(db: Session, acct: ClientAccount | None) -> Optional[str]:
    """Return a friendly display label for a client account.

    Prefers the account name; otherwise uses the linked user's name or email.

    Args:
        db: SQLAlchemy session.
        acct: Optional 'ClientAccount' instance.

    Returns:
        Display string, or None if no label can be determined.
    """
    if acct is None:
        return None
    if getattr(acct, "name", None):
        return acct.name
    if getattr(acct, "client_user_id", None):
        user = db.query(User).filter(User.id == acct.client_user_id).first()
        if user:
            return user.name or user.email
    return None


def serialize_charge(db: Session, charge: AdminFeeCharge) -> AdminFeeChargeOut:
    """Convert a charge row to the API schema shape.

    Args:
        db: SQLAlchemy session.
        charge: 'AdminFeeCharge' instance.

    Returns:
        'AdminFeeChargeOut' suitable for responses.
    """
    acct = (
        db.query(ClientAccount)
        .filter(ClientAccount.id == charge.client_account_id)
        .first()
    )
    label = _client_label(db, acct)
    status_value = (
        charge.status.value
        if isinstance(charge.status, AdminFeeStatus)
        else str(charge.status)
    )
    return AdminFeeChargeOut(
        id=int(charge.id),
        owner_id=str(charge.owner_id),
        client_account_id=int(charge.client_account_id),
        client_user_id=getattr(charge, "client_user_id", None),
        amount_cents=int(charge.amount_cents),
        status=status_value,
        paid_cash_cents=int(charge.paid_cash_cents or 0),
        bundle_applied_cents=int(charge.bundle_applied_cents or 0),
        note=getattr(charge, "note", None),
        created_at=charge.created_at,
        updated_at=charge.updated_at,
        client_label=label,
    )


def create_admin_fee_charge(
    db: Session,
    owner_id: str,
    client_account_id: int,
    amount_cents: Optional[int] = None,
    note: Optional[str] = None,
) -> AdminFeeChargeOut:
    """Create an admin-fee charge for a client account.

    Validates ownership, uses the owner’s current admin-fee amount when
    'amount_cents' is not provided, and auto-applies wallet funds when a wallet
    exists. Attempts to notify the client by email (best-effort).

    Args:
        db: SQLAlchemy session.
        owner_id: Owner identifier.
        client_account_id: Target client account under the owner.
        amount_cents: Optional override amount (>= 0).
        note: Optional note attached to the charge.

    Returns:
        Serialized 'AdminFeeChargeOut' for the newly created charge.

    Raises:
        ValueError: If the client account is not found for the owner.
    """
    acct = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.id == int(client_account_id),
            ClientAccount.owner_user_id == owner_id,
            ClientAccount.deleted_at.is_(None),
        )
        .first()
    )
    if not acct:
        raise ValueError("Client account not found for owner")

    default_amount = get_admin_fee_setting(db, owner_id).admin_fee_cents
    amount = int(amount_cents) if amount_cents is not None else default_amount
    amount = max(amount, 0)

    charge = AdminFeeCharge(
        owner_id=owner_id,
        client_account_id=int(acct.id),
        client_user_id=getattr(acct, "client_user_id", None),
        amount_cents=amount,
        status=AdminFeeStatus.UNPAID,
        paid_cash_cents=0,
        bundle_applied_cents=0,
        note=note,
    )
    db.add(charge)
    db.commit()
    db.refresh(charge)

    wallet = _wallet_for_charge(db, charge)
    if wallet:
        auto_apply_wallet_funds(
            db,
            owner_id=str(owner_id),
            bundle_id=wallet.id,
            note_prefix="Auto-apply wallet funds to admin fee",
        )
        db.refresh(charge)

    try:
        owner: User | None = db.query(User).filter(User.id == owner_id).first()
        to_email: str | None = _account_primary_email(db, acct.id)
        if to_email:
            owner_label = (owner.name if owner and owner.name else None) or "our studio"
            amt = max(int(charge.amount_cents or 0), 0)
            paid = max(int(charge.paid_cash_cents or 0), 0) + max(
                int(charge.bundle_applied_cents or 0), 0
            )
            outstanding = max(amt - paid, 0)

            def _fmt(cents: int) -> str:
                return f"${cents / 100:.2f}"

            lines: list[str] = [
                f"An administration fee has been added to your account with {owner_label}.",
                "",
                f"Amount: {_fmt(amt)}",
            ]
            if charge.bundle_applied_cents:
                lines.append(
                    f"Applied from wallet: {_fmt(int(charge.bundle_applied_cents))}"
                )
            if charge.note:
                lines.append(f"Note: {charge.note}")
            if outstanding > 0:
                lines.append(f"Balance due: {_fmt(outstanding)}")
            else:
                lines.append("This fee is fully covered.")
            lines.append("")
            lines.append("If you have any questions, just reply to this email.")

            subject = f"Administration fee added ({_fmt(amt)})"
            html = render_basic_html("Administration Fee", lines)
            _send_email(to=to_email, subject=subject, text="\n".join(lines), html=html)
    except Exception as _e:
        print(
            f"[ADMIN_FEE_EMAIL] Failed to notify client: {getattr(_e, 'message', _e)}"
        )

    return serialize_charge(db, charge)


def list_admin_fee_charges(
    db: Session,
    owner_id: str,
    status: Optional[Sequence[str]] = None,
    client_account_id: Optional[int] = None,
    limit: Optional[int] = None,
) -> list[AdminFeeChargeOut]:
    """List admin-fee charges for an owner with optional filters.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner identifier.
        status: Optional list of statuses (strings or AdminFeeStatus values).
        client_account_id: Optional account filter.
        limit: Optional maximum number of rows.

    Returns:
        List of 'AdminFeeChargeOut' records matching the criteria.
    """
    q = (
        db.query(AdminFeeCharge)
        .filter(AdminFeeCharge.owner_id == owner_id)
        .order_by(AdminFeeCharge.created_at.desc())
    )
    if status:
        statuses: list[AdminFeeStatus] = []
        for item in status:
            if isinstance(item, AdminFeeStatus):
                statuses.append(item)
                continue
            try:
                statuses.append(AdminFeeStatus(str(item).lower()))
            except ValueError:
                continue
        if statuses:
            q = q.filter(AdminFeeCharge.status.in_(statuses))
    if client_account_id is not None:
        q = q.filter(AdminFeeCharge.client_account_id == int(client_account_id))
    if limit is not None:
        q = q.limit(int(limit))
    rows = q.all()
    return [serialize_charge(db, row) for row in rows]


def _wallet_for_charge(db: Session, charge: AdminFeeCharge) -> Optional[PrepaidBundle]:
    """Return the most recent wallet bundle for the charge’s client, if any.

    Args:
        db: SQLAlchemy session.
        charge: The admin fee charge.

    Returns:
        Most recent 'PrepaidBundle' with 'total_credits == 0' or None if none.
    """
    if not getattr(charge, "client_user_id", None):
        return None
    return (
        db.query(PrepaidBundle)
        .filter(
            PrepaidBundle.owner_id == charge.owner_id,
            PrepaidBundle.client_id == str(charge.client_user_id),
            PrepaidBundle.total_credits == 0,
        )
        .order_by(PrepaidBundle.created_at.desc())
        .first()
    )


def _wallet_balance(db: Session, bundle_id: int) -> int:
    """Compute the current wallet balance (cents) for a bundle.

    Args:
        db: SQLAlchemy session.
        bundle_id: Wallet bundle ID.

    Returns:
        Integer cents balance (can be 0).
    """
    bal = (
        db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
        .filter(PrepaidLedger.bundle_id == bundle_id)
        .scalar()
    )
    return int(bal or 0)


def _apply_wallet_to_charge(db: Session, charge: AdminFeeCharge) -> int:
    """Consume wallet funds up to the outstanding amount for the charge.

    Args:
        db: SQLAlchemy session.
        charge: Admin fee charge to apply funds toward.

    Returns:
        Applied amount in cents.
    """
    wallet = _wallet_for_charge(db, charge)
    if not wallet:
        return 0
    outstanding = max(
        int(charge.amount_cents)
        - int(charge.paid_cash_cents or 0)
        - int(charge.bundle_applied_cents or 0),
        0,
    )
    if outstanding <= 0:
        return 0
    balance = _wallet_balance(db, wallet.id)
    use = min(int(balance), int(outstanding))
    if use <= 0:
        return 0
    db.add(
        PrepaidLedger(
            bundle_id=wallet.id,
            event="consume",
            delta_credits=0,
            amount_cents=-use,
            appointment_id=None,
            note=f"Admin fee charge {charge.id}",
        )
    )
    charge.bundle_applied_cents = int(charge.bundle_applied_cents or 0) + use
    if charge.amount_cents <= (charge.paid_cash_cents + charge.bundle_applied_cents):
        charge.status = AdminFeeStatus.BUNDLE
    db.add(charge)
    return use


def _refund_wallet_for_charge(db: Session, charge: AdminFeeCharge) -> int:
    """Return previously consumed wallet funds for this charge back to the wallet.

    Args:
        db: SQLAlchemy session.
        charge: Admin fee charge whose wallet application is being refunded.

    Returns:
        Refunded amount in cents.
    """
    refund = int(charge.bundle_applied_cents or 0)
    if refund <= 0:
        return 0
    wallet = _wallet_for_charge(db, charge)
    if not wallet:
        return 0
    db.add(
        PrepaidLedger(
            bundle_id=wallet.id,
            event="restore",
            delta_credits=0,
            amount_cents=refund,
            appointment_id=None,
            note=f"Refund admin fee charge {charge.id}",
        )
    )
    charge.bundle_applied_cents = 0
    db.add(charge)
    return refund


def update_admin_fee_charge(
    db: Session,
    owner_id: str,
    charge_id: int,
    *,
    status: Optional[str] = None,
    paid_cash_cents: Optional[int] = None,
    note: Optional[str] = None,
    apply_wallet: bool | None = None,
) -> AdminFeeChargeOut:
    """Update charge fields: cash paid, note, status, and wallet application.

    Coerces status to enum with validation. Automatically transitions to
    'PAID' if covered by cash + wallet. Handles 'REFUNDED' and 'WAIVED'
    semantics including wallet refunds.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner identifier; must own the charge.
        charge_id: Charge identifier.
        status: Optional new status (string or enum).
        paid_cash_cents: Optional cash paid amount to set (>= 0).
        note: Optional note to set.
        apply_wallet: If true, apply available wallet funds.

    Returns:
        Updated 'AdminFeeChargeOut'.

    Raises:
        ValueError: If the charge is not found or the status is invalid.
    """
    charge = (
        db.query(AdminFeeCharge)
        .filter(
            AdminFeeCharge.id == int(charge_id), AdminFeeCharge.owner_id == owner_id
        )
        .first()
    )
    if not charge:
        raise ValueError("Charge not found")

    if paid_cash_cents is not None:
        charge.paid_cash_cents = max(int(paid_cash_cents), 0)
    if note is not None:
        charge.note = note

    if apply_wallet:
        _apply_wallet_to_charge(db, charge)

    new_status: Optional[AdminFeeStatus] = None
    if status is not None:
        try:
            new_status = (
                status
                if isinstance(status, AdminFeeStatus)
                else AdminFeeStatus(str(status).lower())
            )
        except ValueError:
            raise ValueError("Invalid status")
        if new_status is AdminFeeStatus.REFUNDED:
            _refund_wallet_for_charge(db, charge)
            charge.paid_cash_cents = 0
        if new_status is AdminFeeStatus.WAIVED:
            charge.paid_cash_cents = 0
            charge.bundle_applied_cents = 0
        charge.status = new_status

    # Auto-close if fully paid via cash + bundle
    if (charge.paid_cash_cents + charge.bundle_applied_cents) >= charge.amount_cents:
        if charge.status not in {
            AdminFeeStatus.BUNDLE,
            AdminFeeStatus.REFUNDED,
            AdminFeeStatus.WAIVED,
        }:
            charge.status = AdminFeeStatus.PAID
    elif charge.status == AdminFeeStatus.PAID:
        charge.status = AdminFeeStatus.UNPAID

    db.add(charge)
    db.commit()
    db.refresh(charge)
    return serialize_charge(db, charge)


def delete_admin_fee_charge(
    db: Session,
    owner_id: str,
    charge_id: int,
) -> None:
    """Hard-delete an admin fee charge that has no payments applied.

    Rules:
      - Only the owning owner can delete.
      - Disallowed if any cash or wallet funds have been applied, or if status is
        not 'UNPAID'.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner identifier; must own the charge.
        charge_id: Charge identifier to delete.

    Raises:
        ValueError: If the charge is not found, has payments applied, or is not unpaid.
    """
    charge = (
        db.query(AdminFeeCharge)
        .filter(
            AdminFeeCharge.id == int(charge_id), AdminFeeCharge.owner_id == owner_id
        )
        .first()
    )
    if not charge:
        raise ValueError("Charge not found")

    if (int(charge.paid_cash_cents or 0) > 0) or (
        int(charge.bundle_applied_cents or 0) > 0
    ):
        raise ValueError(
            "Cannot delete a charge with payments applied; refund or waive instead"
        )
    if charge.status != AdminFeeStatus.UNPAID:
        raise ValueError("Only unpaid charges can be deleted; refund or waive instead")

    db.delete(charge)
    db.commit()
