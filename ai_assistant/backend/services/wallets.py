from __future__ import annotations
from datetime import datetime, timezone as _tz
from typing import Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from app.models import (
    Appointment,
    AdminFeeCharge,
    AdminFeeStatus,
    ClientAccount,
    Person,
    PrepaidBundle,
    PrepaidLedger,
)


class WalletAdjustmentError(ValueError):
    """Raised when a wallet adjustment request is invalid or unsafe."""


def _wallet_balance(db: Session, bundle_id: int) -> int:
    """Return the current wallet balance for a bundle in cents.

    Args:
        db: SQLAlchemy session.
        bundle_id: Wallet bundle identifier.

    Returns:
        Integer cents balance (sum of 'PrepaidLedger.amount_cents').
    """
    total = (
        db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
        .filter(PrepaidLedger.bundle_id == int(bundle_id))
        .scalar()
    )
    return int(total or 0)


def auto_apply_wallet_funds(
    db: Session,
    *,
    owner_id: str,
    bundle_id: int,
    note_prefix: str = "Auto-apply wallet funds",
) -> Dict[str, int]:
    """Apply wallet funds to outstanding appointments and admin-fee charges.

    Applies funds FIFO by date: first appointments, then admin fee charges.
    Only considers items for the wallet’s client and owner. Updates appointment
    'payment_status' to "paid" when fully covered.

    Transaction semantics: uses the provided 'Session'; commits only when
    changes are made, otherwise rolls back to avoid open transactions.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner identifier; must match the wallet’s owner.
        bundle_id: Wallet bundle id ('PrepaidBundle' with 'total_credits == 0').
        note_prefix: Prefix for ledger entry notes.

    Returns:
        Summary dict with keys:
          - 'applied_cents': Total cents applied
          - 'appointments': Count of appointments affected
          - 'admin_fee_charges': Count of admin fee charges affected
          - 'remaining_balance_cents': Wallet balance after application
    """

    summary = {
        "applied_cents": 0,
        "appointments": 0,
        "admin_fee_charges": 0,
        "remaining_balance_cents": 0,
    }

    wallet = db.get(PrepaidBundle, int(bundle_id))
    if not wallet or str(wallet.owner_id) != str(owner_id) or wallet.total_credits != 0:
        summary["remaining_balance_cents"] = _wallet_balance(db, bundle_id)
        return summary

    balance = _wallet_balance(db, wallet.id)
    summary["remaining_balance_cents"] = balance
    if balance <= 0:
        return summary
    account_rows = (
        db.query(ClientAccount.id)
        .filter(
            ClientAccount.owner_user_id == str(owner_id),
            ClientAccount.client_user_id == str(wallet.client_id),
            ClientAccount.deleted_at.is_(None),
        )
        .all()
    )
    account_ids = [int(row[0]) for row in account_rows]

    funds_remaining = int(balance)
    changes_made = False
    from services.payments import _service_price_map, compute_price_cents

    price_map = _service_price_map(db, owner_user_id=str(owner_id))

    # ---------- Apply to appointments (oldest first) ----------
    client_match_filters = [Appointment.client_id == str(wallet.client_id)]
    if account_ids:
        client_match_filters.append(Person.account_id.in_(account_ids))
    if len(client_match_filters) == 1:
        client_match_clause = client_match_filters[0]
    else:
        client_match_clause = or_(*client_match_filters)

    # Include canceled appointments too so wallet deposits can cover
    # cancellation-owed balances automatically.
    appt_query = (
        db.query(Appointment)
        .outerjoin(Person, Person.id == Appointment.person_id)
        .filter(
            Appointment.owner_id == str(owner_id),
            client_match_clause,
            or_(
                Appointment.payment_status.is_(None),
                Appointment.payment_status == "unpaid",
            ),
        )
        .order_by(Appointment.start_utc.asc())
    )

    for appt in appt_query.all():
        if funds_remaining <= 0:
            break
        if appt.bundle_id and int(appt.bundle_id) != int(wallet.id):
            continue

        expected = compute_price_cents(db, appt, price_map)
        if expected is None or int(expected) <= 0:
            continue

        cash_paid = int(getattr(appt, "amount_paid_cents", 0) or 0)
        ledger_total = (
            db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
            .filter(PrepaidLedger.appointment_id == appt.id)
            .scalar()
        )
        already_applied = -int(ledger_total or 0) if int(ledger_total or 0) < 0 else 0
        owed = max(int(expected) - cash_paid - already_applied, 0)
        if owed <= 0:
            continue

        use = min(funds_remaining, owed)
        if use <= 0:
            continue

        db.add(
            PrepaidLedger(
                bundle_id=wallet.id,
                event="consume",
                delta_credits=0,
                amount_cents=-int(use),
                appointment_id=appt.id,
                note=f"{note_prefix} (appointment {appt.id})",
            )
        )

        funds_remaining -= int(use)
        summary["applied_cents"] += int(use)
        summary["appointments"] += 1
        changes_made = True

        # Attach wallet if not already linked
        if not getattr(appt, "bundle_id", None):
            appt.bundle_id = wallet.id

        if (cash_paid + already_applied + int(use)) >= int(expected):
            appt.payment_status = "paid"
            if getattr(appt, "paid_at", None) is None:
                appt.paid_at = datetime.now(_tz.utc)
        db.add(appt)

    if funds_remaining > 0:
        # ---------- Apply to admin fee charges (oldest first) ----------
        charge_filters = [
            AdminFeeCharge.owner_id == str(owner_id),
            AdminFeeCharge.status == AdminFeeStatus.UNPAID,
        ]
        if account_ids:
            charge_filters.append(AdminFeeCharge.client_account_id.in_(account_ids))
        else:
            charge_filters.append(
                AdminFeeCharge.client_user_id == str(wallet.client_id)
            )

        charge_query = (
            db.query(AdminFeeCharge)
            .filter(*charge_filters)
            .order_by(AdminFeeCharge.created_at.asc())
        )

        for charge in charge_query.all():
            if funds_remaining <= 0:
                break
            outstanding = max(
                int(charge.amount_cents)
                - int(charge.paid_cash_cents or 0)
                - int(charge.bundle_applied_cents or 0),
                0,
            )
            if outstanding <= 0:
                continue

            use = min(funds_remaining, outstanding)
            if use <= 0:
                continue

            db.add(
                PrepaidLedger(
                    bundle_id=wallet.id,
                    event="consume",
                    delta_credits=0,
                    amount_cents=-int(use),
                    appointment_id=None,
                    note=f"{note_prefix} (admin fee {charge.id})",
                )
            )

            charge.bundle_applied_cents = int(charge.bundle_applied_cents or 0) + int(
                use
            )
            if int(charge.paid_cash_cents or 0) + int(
                charge.bundle_applied_cents or 0
            ) >= int(charge.amount_cents):
                charge.status = AdminFeeStatus.BUNDLE

            db.add(charge)

            funds_remaining -= int(use)
            summary["applied_cents"] += int(use)
            summary["admin_fee_charges"] += 1
            changes_made = True

    if changes_made:
        db.commit()
    else:
        db.rollback()

    summary["remaining_balance_cents"] = _wallet_balance(db, wallet.id)
    return summary


def adjust_wallet_balance(
    db: Session,
    *,
    owner_id: str,
    bundle_id: int,
    amount_cents: int,
    note: Optional[str] = None,
    client_user_id: Optional[str] = None,
    client_account_id: Optional[int] = None,
) -> int:
    """Adjust a wallet (store-credit bundle) balance by +/- amount.

    Validates bundle ownership and optional client constraints, prevents
    overdrafts, and records an 'adjust' ledger row. On credits (positive
    adjustments), automatically attempts to apply funds to outstanding items.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner identifier; must match wallet owner.
        bundle_id: Wallet bundle identifier (must be a wallet, not credits).
        amount_cents: Positive to credit, negative to debit.
        note: Optional ledger note; defaults to "Manual credit/debit".
        client_user_id: Optional client ownership assertion.
        client_account_id: Optional account ownership assertion.

    Returns:
        The new wallet balance in cents after the adjustment and any auto-apply.

    Raises:
        WalletAdjustmentError: On zero amount, wrong owner, non-wallet bundle,
        mismatched client/account, or overdraft attempt.
    """
    delta = int(amount_cents)
    if delta == 0:
        raise WalletAdjustmentError("amount_cents must be non-zero")

    bundle = db.get(PrepaidBundle, int(bundle_id))
    if not bundle or str(bundle.owner_id) != str(owner_id):
        raise WalletAdjustmentError("Bundle not found for this owner")
    if bundle.total_credits != 0:
        raise WalletAdjustmentError("Adjustments are supported only for wallets")

    if client_user_id and str(bundle.client_id) != str(client_user_id):
        raise WalletAdjustmentError("Bundle belongs to a different client")
    if client_account_id is not None:
        acct = (
            db.query(ClientAccount)
            .filter(
                ClientAccount.id == int(client_account_id),
                ClientAccount.owner_user_id == owner_id,
                ClientAccount.deleted_at.is_(None),
            )
            .first()
        )
        if not acct or str(acct.client_user_id) != str(bundle.client_id):
            raise WalletAdjustmentError(
                "Bundle does not match the supplied client account"
            )

    balance = _wallet_balance(db, bundle.id)

    if delta < 0 and balance + delta < 0:
        raise WalletAdjustmentError(
            "That would overdraft the wallet. Reduce the amount and try again."
        )

    entry_note = note or ("Manual debit" if delta < 0 else "Manual credit")
    db.add(
        PrepaidLedger(
            bundle_id=bundle.id,
            event="adjust",
            delta_credits=0,
            amount_cents=delta,
            appointment_id=None,
            note=entry_note,
        )
    )
    db.commit()

    if delta > 0:
        summary = auto_apply_wallet_funds(
            db,
            owner_id=str(owner_id),
            bundle_id=bundle.id,
            note_prefix="Auto-apply wallet funds after manual adjustment",
        )
        return int(
            summary.get("remaining_balance_cents", _wallet_balance(db, bundle.id))
        )

    return int(_wallet_balance(db, bundle.id))
