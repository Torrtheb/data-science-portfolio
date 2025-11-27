from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, ToolException
from sqlalchemy import func

from app.db import SessionLocal
from app.models import (
    User,
    Appointment,
    Person,
    ClientAccount,
    ClientEmail,
    PrepaidBundle,
    PrepaidLedger,
)
from services.payments import compute_price_cents, _service_price_map
from services.admin_fee import create_admin_fee_charge as svc_create_admin_fee_charge
from services.wallets import (
    adjust_wallet_balance,
    WalletAdjustmentError,
    auto_apply_wallet_funds,
)

from agent.tool_utils import _owner_id_from_config
from agent.tool_ops import _fmt_cents as _fmt_cents_symbol
from agent.schemas import ToolAdjustWalletIn, ToolAdjustWalletOut
from agent.schemas import (
    ToolAdminFeeChargeIn,
    ToolAdminFeeChargeOut,
    ToolCreateBundleIn,
    ToolCreateBundleOut,
)

log = logging.getLogger(__name__)


@tool("attach_wallet", return_direct=False)
def attach_wallet_tool(
    appointment_id: str,
    bundle_id: int,
    config: RunnableConfig | None = None,
) -> Dict[str, object]:
    """
    Attach an existing wallet (bundle) to an appointment.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            appt = (
                db.query(Appointment)
                .filter(
                    Appointment.id == appointment_id, Appointment.owner_id == owner_id
                )
                .first()
            )
            if not appt:
                raise ToolException("Appointment not found")
            b = db.get(PrepaidBundle, int(bundle_id))
            if not b or str(b.owner_id) != str(owner_id):
                raise ToolException("Wallet not found")
            if getattr(appt, "client_id", None) and str(b.client_id) != str(
                appt.client_id
            ):
                raise ToolException("Wallet belongs to a different client")
            appt.bundle_id = int(bundle_id)
            db.add(appt)
            db.commit()
            db.refresh(appt)
            return {
                "ok": True,
                "appointment_id": str(appt.id),
                "bundle_id": appt.bundle_id,
            }
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"attach_wallet failed: {e}")


@tool("apply_wallet", return_direct=False)
def apply_wallet_tool(
    appointment_id: str,
    config: RunnableConfig | None = None,
) -> Dict[str, int | bool]:
    """
    Apply wallet funds up to the owed amount for this appointment. Does not change payment_status.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            appt = (
                db.query(Appointment)
                .filter(
                    Appointment.id == appointment_id, Appointment.owner_id == owner_id
                )
                .first()
            )
            if not appt:
                raise ToolException("Appointment not found")
            if not getattr(appt, "bundle_id", None):
                raise ToolException("No wallet attached")
            bid = int(appt.bundle_id)

            # lock bundle row to avoid races
            db.query(PrepaidBundle).filter(
                PrepaidBundle.id == bid
            ).with_for_update().first()

            # compute expected and owed
            try:
                price_map = _service_price_map(db, owner_user_id=owner_id)
                expected = compute_price_cents(db, appt, price_map)
            except Exception:
                expected = None
            cash = int(getattr(appt, "amount_paid_cents", 0) or 0)

            # already consumed for this appointment (negative)
            already = (
                db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                .filter(
                    PrepaidLedger.bundle_id == bid,
                    PrepaidLedger.appointment_id == appt.id,
                    PrepaidLedger.amount_cents < 0,
                )
                .scalar()
            ) or 0
            already = -int(already) if int(already) < 0 else 0

            owed = max(int(expected or 0) - cash - already, 0)
            if owed <= 0:
                return {"ok": True, "applied_cents": 0}

            # wallet balance (sum of all amount_cents)
            bal = (
                db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                .filter(PrepaidLedger.bundle_id == bid)
                .scalar()
            ) or 0
            use = min(int(bal), int(owed))
            if use <= 0:
                return {"ok": True, "applied_cents": 0}
            db.add(
                PrepaidLedger(
                    bundle_id=bid,
                    event="consume",
                    delta_credits=0,
                    amount_cents=-use,
                    appointment_id=appt.id,
                    note="Agent apply wallet",
                )
            )
            db.commit()
            return {"ok": True, "applied_cents": use}
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"apply_wallet failed: {e}")


@tool("restore_wallet", return_direct=False)
def restore_wallet_tool(
    appointment_id: str,
    config: RunnableConfig | None = None,
) -> Dict[str, int | bool]:
    """
    Restore any wallet funds previously applied for this appointment back to the wallet.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            appt = (
                db.query(Appointment)
                .filter(
                    Appointment.id == appointment_id, Appointment.owner_id == owner_id
                )
                .first()
            )
            if not appt:
                raise ToolException("Appointment not found")
            if not getattr(appt, "bundle_id", None):
                return {"ok": True, "restored_cents": 0}
            bid = int(appt.bundle_id)
            spent = (
                db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                .filter(
                    PrepaidLedger.bundle_id == bid,
                    PrepaidLedger.appointment_id == appt.id,
                    PrepaidLedger.amount_cents < 0,
                )
                .scalar()
            ) or 0
            spent = -int(spent) if int(spent) < 0 else 0
            if spent > 0:
                db.add(
                    PrepaidLedger(
                        bundle_id=bid,
                        event="restore",
                        delta_credits=0,
                        amount_cents=+spent,
                        appointment_id=appt.id,
                        note="Agent restore wallet",
                    )
                )
                db.commit()
            return {"ok": True, "restored_cents": spent}
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"restore_wallet failed: {e}")


@tool("top_up_wallet", return_direct=False)
def top_up_wallet_tool(
    client_user_id: str,
    bundle_id: int,
    amount_cents: int,
    note: str | None = None,
    config: RunnableConfig | None = None,
) -> Dict[str, bool]:
    """
    Add funds to an existing wallet for a client.
    """
    try:
        if int(amount_cents) <= 0:
            raise ToolException("amount_cents must be positive")
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            b = db.get(PrepaidBundle, int(bundle_id))
            if not b or str(b.owner_id) != str(owner_id):
                raise ToolException("Wallet not found")
            # Resolve client_user_id if it's an email/name
            if client_user_id and not db.get(User, str(client_user_id)):
                sel = str(client_user_id).strip()
                resolved: Optional[str] = None
                try:
                    if "@" in sel:
                        acct = (
                            db.query(ClientAccount)
                            .join(
                                ClientEmail, ClientEmail.account_id == ClientAccount.id
                            )
                            .filter(
                                ClientAccount.owner_user_id == owner_id,
                                ClientAccount.deleted_at.is_(None),
                                func.lower(ClientEmail.email) == func.lower(sel),
                            )
                            .first()
                        )
                        if acct and getattr(acct, "client_user_id", None):
                            resolved = str(acct.client_user_id)
                    else:
                        acct = (
                            db.query(ClientAccount)
                            .filter(
                                ClientAccount.owner_user_id == owner_id,
                                ClientAccount.deleted_at.is_(None),
                                func.lower(ClientAccount.name) == sel.lower(),
                            )
                            .first()
                        )
                        if acct and getattr(acct, "client_user_id", None):
                            resolved = str(acct.client_user_id)
                        if not resolved:
                            p = (
                                db.query(Person)
                                .join(
                                    ClientAccount, ClientAccount.id == Person.account_id
                                )
                                .filter(
                                    ClientAccount.owner_user_id == owner_id,
                                    func.lower(Person.full_name) == sel.lower(),
                                )
                                .first()
                            )
                            if p:
                                acct = (
                                    db.query(ClientAccount)
                                    .filter(ClientAccount.id == p.account_id)
                                    .first()
                                )
                                if acct and getattr(acct, "client_user_id", None):
                                    resolved = str(acct.client_user_id)
                except Exception:
                    resolved = None
                if resolved:
                    client_user_id = resolved
            if str(b.client_id) != str(client_user_id):
                raise ToolException("Wallet belongs to a different client")
            db.add(
                PrepaidLedger(
                    bundle_id=b.id,
                    event="purchase",
                    delta_credits=0,
                    amount_cents=int(amount_cents),
                    appointment_id=None,
                    note=note or "Agent top up",
                )
            )
            db.commit()
            # After a successful top-up, auto-apply wallet funds to any outstanding items
            try:
                auto_apply_wallet_funds(
                    db,
                    owner_id=str(owner_id),
                    bundle_id=int(b.id),
                    note_prefix="Auto-apply wallet funds after agent top-up",
                )
            except Exception:
                pass
            return {"ok": True}
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"top_up_wallet failed: {e}")


@tool("adjust_wallet", args_schema=ToolAdjustWalletIn, return_direct=False)
def adjust_wallet_tool(
    bundle_id: int,
    amount_cents: int,
    note: Optional[str] = None,
    client_user_id: Optional[str] = None,
    client_account_id: Optional[int] = None,
    owner_user_id: Optional[str] = None,
    config: RunnableConfig | None = None,
) -> ToolAdjustWalletOut:
    """Manually adjust a client's wallet balance (positive to add funds, negative to remove)."""
    try:
        owner_id = owner_user_id or _owner_id_from_config(config)
        if not owner_id:
            raise ToolException("owner_user_id is required")

        with SessionLocal() as db:
            # Resolve client_user_id from account id if needed
            if not client_user_id and client_account_id is not None:
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
                    raise ToolException("Client account not found")
                client_user_id = acct.client_user_id

            # If client_user_id looks like an email or name, resolve to a real user id
            if client_user_id and not db.get(User, str(client_user_id)):
                sel = str(client_user_id).strip()
                resolved: Optional[str] = None
                try:
                    if "@" in sel:
                        acct = (
                            db.query(ClientAccount)
                            .join(
                                ClientEmail, ClientEmail.account_id == ClientAccount.id
                            )
                            .filter(
                                ClientAccount.owner_user_id == owner_id,
                                ClientAccount.deleted_at.is_(None),
                                func.lower(ClientEmail.email) == func.lower(sel),
                            )
                            .first()
                        )
                        if acct and getattr(acct, "client_user_id", None):
                            resolved = str(acct.client_user_id)
                    else:
                        acct = (
                            db.query(ClientAccount)
                            .filter(
                                ClientAccount.owner_user_id == owner_id,
                                ClientAccount.deleted_at.is_(None),
                                func.lower(ClientAccount.name) == sel.lower(),
                            )
                            .first()
                        )
                        if acct and getattr(acct, "client_user_id", None):
                            resolved = str(acct.client_user_id)
                        if not resolved:
                            p = (
                                db.query(Person)
                                .join(
                                    ClientAccount, ClientAccount.id == Person.account_id
                                )
                                .filter(
                                    ClientAccount.owner_user_id == owner_id,
                                    func.lower(Person.full_name) == sel.lower(),
                                )
                                .first()
                            )
                            if p:
                                acct = (
                                    db.query(ClientAccount)
                                    .filter(ClientAccount.id == p.account_id)
                                    .first()
                                )
                                if acct and getattr(acct, "client_user_id", None):
                                    resolved = str(acct.client_user_id)
                except Exception:
                    resolved = None
                if resolved:
                    client_user_id = resolved

            new_balance = adjust_wallet_balance(
                db,
                owner_id=owner_id,
                bundle_id=int(bundle_id),
                amount_cents=int(amount_cents),
                note=note,
                client_user_id=client_user_id,
                client_account_id=client_account_id,
            )

            return ToolAdjustWalletOut(
                ok=True, bundle_id=int(bundle_id), balance_cents=int(new_balance)
            )
    except WalletAdjustmentError as exc:
        raise ToolException(str(exc))
    except ToolException:
        raise
    except Exception as exc:
        raise ToolException(f"adjust_wallet failed: {exc}")


@tool("create_admin_fee_charge", args_schema=ToolAdminFeeChargeIn, return_direct=False)
def create_admin_fee_charge_tool(
    payload: ToolAdminFeeChargeIn,
    config: RunnableConfig | None = None,
) -> ToolAdminFeeChargeOut:
    """Create an administration fee charge for a specific client account."""
    try:
        owner_id = payload.owner_user_id or _owner_id_from_config(config)
        if not owner_id:
            raise ToolException("owner_user_id is required")
        payload.require_target()

        with SessionLocal() as db:
            acct = None
            if payload.client_account_id:
                acct = (
                    db.query(ClientAccount)
                    .filter(
                        ClientAccount.id == int(payload.client_account_id),
                        ClientAccount.owner_user_id == owner_id,
                        ClientAccount.deleted_at.is_(None),
                    )
                    .first()
                )
                if not acct:
                    raise ToolException("Client account not found for this owner")
            elif payload.client_user_id:
                acct = (
                    db.query(ClientAccount)
                    .filter(
                        ClientAccount.owner_user_id == owner_id,
                        ClientAccount.client_user_id == payload.client_user_id,
                        ClientAccount.deleted_at.is_(None),
                    )
                    .first()
                )
                if not acct:
                    raise ToolException("Client account not found for that client")
            elif payload.client_email:
                email = str(payload.client_email).strip().lower()
                acct = (
                    db.query(ClientAccount)
                    .join(ClientEmail, ClientEmail.account_id == ClientAccount.id)
                    .filter(
                        ClientAccount.owner_user_id == owner_id,
                        ClientAccount.deleted_at.is_(None),
                        func.lower(ClientEmail.email) == email,
                    )
                    .first()
                )
                if not acct:
                    raise ToolException("Client email not found for this owner")
            else:
                raise ToolException(
                    "Provide client_account_id, client_user_id, or client_email"
                )

            charge = svc_create_admin_fee_charge(
                db,
                owner_id=owner_id,
                client_account_id=int(acct.id),
                amount_cents=payload.amount_cents,
                note=payload.note,
            )
            return ToolAdminFeeChargeOut(
                charge_id=int(charge.id),
                client_account_id=int(charge.client_account_id),
                amount_cents=int(charge.amount_cents),
                status=str(charge.status),
            )
    except ToolException:
        raise
    except ValueError as exc:
        raise ToolException(str(exc))
    except Exception as exc:
        raise ToolException(f"create_admin_fee_charge failed: {exc}")


@tool("create_bundle", args_schema=ToolCreateBundleIn, return_direct=False)
def create_bundle_tool(
    client_user_id: str,
    name: str = "Bundle",
    total_credits: int = 0,
    price_cents: int = 0,
    currency: str = "USD",
    expires_at: Optional[datetime] = None,
    owner_user_id: Optional[str] = None,
    config: RunnableConfig | None = None,
) -> ToolCreateBundleOut:
    """Create a prepaid bundle (wallet or credit pack) for a client.

    total_credits == 0 creates a wallet (store credit) funded by
    'price_cents'. total_credits > 0 creates a credit pack, with
    'price_cents' as the bundle price.

    Args:
        client_user_id: Client user id.
        name: Bundle name label.
        total_credits: 0 for wallet; >0 for credit packs.
        price_cents: Price or initial deposit in cents.
        currency: Currency code.
        expires_at: Optional expiry datetime.
        owner_user_id: Explicit owner id (optional; otherwise from config).
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolCreateBundleOut' with bundle details and remaining balance.

    Raises:
        ToolException: If owner/client cannot be resolved or ownership invalid.
    """
    try:
        owner_id = owner_user_id or _owner_id_from_config(config)
        with SessionLocal() as db:
            owner = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")
            client = db.query(User).filter(User.id == client_user_id).first()
            if not client:
                raise ToolException("Client not found")
            acct = (
                db.query(ClientAccount)
                .filter(
                    ClientAccount.owner_user_id == owner.id,
                    ClientAccount.client_user_id == client.id,
                    ClientAccount.deleted_at.is_(None),
                )
                .first()
            )
            if not acct:
                raise ToolException("Client is not associated with this owner")

            b = PrepaidBundle(
                owner_id=owner.id,
                client_id=client.id,
                name=name,
                total_credits=int(total_credits or 0),
                remaining_credits=int(total_credits or 0),
                price_cents=int(price_cents or 0),
                currency=currency or "USD",
                expires_at=expires_at,
            )
            db.add(b)
            db.flush()
            db.add(
                PrepaidLedger(
                    bundle_id=b.id,
                    event="purchase",
                    delta_credits=int(total_credits or 0),
                    amount_cents=int(price_cents or 0),
                    appointment_id=None,
                    note="Agent-created bundle",
                )
            )
            db.commit()
            db.refresh(b)

            summary = None
            if b.total_credits == 0:
                summary = auto_apply_wallet_funds(
                    db,
                    owner_id=str(owner.id),
                    bundle_id=b.id,
                    note_prefix="Auto-apply wallet funds after bundle creation",
                )
                db.refresh(b)

            if summary is not None:
                rem_balance = summary.get("remaining_balance_cents")
            else:
                try:
                    bal = (
                        db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                        .filter(PrepaidLedger.bundle_id == b.id)
                        .scalar()
                    )
                    rem_balance = int(bal or 0)
                except Exception:
                    rem_balance = None

            return ToolCreateBundleOut(
                id=b.id,
                client_id=b.client_id,
                name=b.name,
                total_credits=b.total_credits,
                remaining_credits=b.remaining_credits,
                price_cents=b.price_cents,
                currency=b.currency,
                status=b.status,
                expires_at=b.expires_at,
                created_at=b.created_at,
                remaining_balance_cents=rem_balance,
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"create_bundle failed: {e}")


@tool("list_wallets", return_direct=False)
def list_wallets_tool(
    client_user_id: Optional[str] = None,
    client_account_id: Optional[int] = None,
    config: RunnableConfig | None = None,
) -> List[Dict[str, int | str]]:
    """
    List wallet bundles (store credit, total_credits == 0) for a client under this owner.
    Accepts either client_user_id or client_account_id.
    Returns: [{id, name, remaining_balance_cents, currency, status}]
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            if not client_user_id and client_account_id is not None:
                acct = (
                    db.query(ClientAccount)
                    .filter(ClientAccount.id == int(client_account_id))
                    .first()
                )
                if not acct or str(acct.owner_user_id) != str(owner_id):
                    raise ToolException("Account not found for this owner")
                client_user_id = str(acct.client_user_id)
            if client_user_id and not db.get(User, str(client_user_id)):
                sel = str(client_user_id).strip()
                resolved: Optional[str] = None
                try:
                    if "@" in sel:
                        acct = (
                            db.query(ClientAccount)
                            .join(
                                ClientEmail, ClientEmail.account_id == ClientAccount.id
                            )
                            .filter(
                                ClientAccount.owner_user_id == owner_id,
                                ClientAccount.deleted_at.is_(None),
                                func.lower(ClientEmail.email) == func.lower(sel),
                            )
                            .first()
                        )
                        if acct and getattr(acct, "client_user_id", None):
                            resolved = str(acct.client_user_id)
                    else:
                        acct = (
                            db.query(ClientAccount)
                            .filter(
                                ClientAccount.owner_user_id == owner_id,
                                ClientAccount.deleted_at.is_(None),
                                func.lower(ClientAccount.name) == sel.lower(),
                            )
                            .first()
                        )
                        if acct and getattr(acct, "client_user_id", None):
                            resolved = str(acct.client_user_id)
                        if not resolved:
                            p = (
                                db.query(Person)
                                .join(
                                    ClientAccount, ClientAccount.id == Person.account_id
                                )
                                .filter(
                                    ClientAccount.owner_user_id == owner_id,
                                    func.lower(Person.full_name) == sel.lower(),
                                )
                                .first()
                            )
                            if p:
                                acct = (
                                    db.query(ClientAccount)
                                    .filter(ClientAccount.id == p.account_id)
                                    .first()
                                )
                                if acct and getattr(acct, "client_user_id", None):
                                    resolved = str(acct.client_user_id)
                except Exception:
                    resolved = None
                if resolved:
                    client_user_id = resolved
            if not client_user_id:
                raise ToolException("Provide client_user_id or client_account_id")

            q = (
                db.query(PrepaidBundle)
                .filter(
                    PrepaidBundle.owner_id == owner_id,
                    PrepaidBundle.client_id == str(client_user_id),
                    PrepaidBundle.total_credits == 0,
                )
                .order_by(PrepaidBundle.created_at.desc())
            )
            bundles = q.all()
            out = []
            for b in bundles:
                bal = (
                    db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                    .filter(PrepaidLedger.bundle_id == b.id)
                    .scalar()
                ) or 0
                # Always present a pretty CAD symbol format to keep responses consistent
                out.append(
                    {
                        "id": int(b.id),
                        "name": getattr(b, "name", "Wallet"),
                        "remaining_balance_cents": int(bal),
                        "currency": getattr(b, "currency", "CAD") or "CAD",
                        "status": getattr(b, "status", "active") or "active",
                        "remaining_balance_pretty": _fmt_cents_symbol(int(bal), "$"),
                    }
                )
            return out
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"list_wallets failed: {e}")
