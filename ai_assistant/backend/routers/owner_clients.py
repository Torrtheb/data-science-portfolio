from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, date
import os
from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError
from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import (
    User,
    RoleEnum,
    ClientAccount,
    ClientEmail,
    PrepaidBundle,
    PrepaidLedger,
    Person,
)

from app.schemas import (
    OwnerCreate,
    OwnerOut,
    PrepaidBundleCreate,
    PrepaidBundleOut,
    ClientAccountSummary,
)
from ._helpers import (
    uuid_str,
    send_email,
    _html_from_text,
)

from sqlalchemy import func
from services.wallets import (
    adjust_wallet_balance,
    WalletAdjustmentError,
    auto_apply_wallet_funds,
)


router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])
MAX_BROADCAST_RECIPIENTS = int(os.getenv("MAX_BROADCAST_RECIPIENTS", "200") or 200)
REQUIRE_BROADCAST_CONFIRM = os.getenv("BROADCAST_REQUIRE_CONFIRM", "1") == "1"


def _ensure_wallet(db: Session, *, owner_id: str, client_id: str) -> PrepaidBundle:
    """
    Guarantee a base wallet (PrepaidBundle with total_credits == 0) exists for this owner/client.

    Creates a zero-credit, USD wallet if none exists. Returns the wallet.
    """
    wallet = (
        db.query(PrepaidBundle)
        .filter(
            PrepaidBundle.owner_id == owner_id,
            PrepaidBundle.client_id == client_id,
            PrepaidBundle.total_credits == 0,
            PrepaidBundle.status == "active",
        )
        .order_by(PrepaidBundle.created_at.desc())
        .first()
    )
    if wallet:
        return wallet
    wallet = PrepaidBundle(
        owner_id=owner_id,
        client_id=client_id,
        name="Wallet",
        total_credits=0,
        remaining_credits=0,
        price_cents=0,
        currency="USD",
        status="active",
    )
    db.add(wallet)
    db.commit()
    db.refresh(wallet)
    return wallet


def _sendable_email_for_account(
    db: Session, account_id: int
) -> tuple[str | None, bool]:
    """
    Pick a sendable email for a client account, respecting unsubscribe flags.

    Returns:
        (email, unsubscribed_found)
        - email: first non-unsubscribed address (primary-first order) or None
        - unsubscribed_found: True if the account has only unsubscribed emails
    """
    rows = (
        db.query(ClientEmail.email, ClientEmail.is_primary, ClientEmail.unsubscribed)
        .filter(ClientEmail.account_id == account_id)
        .order_by(ClientEmail.is_primary.desc(), ClientEmail.id.asc())
        .all()
    )
    unsubscribed_only = False
    for email, is_primary, unsubscribed in rows:
        if unsubscribed:
            unsubscribed_only = True
            continue
        return email, unsubscribed_only
    return None, unsubscribed_only


class ClientOut(BaseModel):
    """Minimal client identity for responses."""

    id: str
    name: Optional[str] = None
    email: EmailStr


class OwnerClientUpdate(BaseModel):
    """Owner update payload for client account fields and emails."""

    name: Optional[str] = None
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None
    emails: Optional[list[dict]] = None


@router.get("/owner/clients", response_model=list[ClientAccountSummary])
def owner_list_clients(
    search: Optional[str] = Query(
        None, description="Filter by account name, client name or email (icontains)"
    ),
    limit: int = Query(25, ge=1, le=200),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> list[ClientAccountSummary]:
    """
    List client ACCOUNTS for this owner with people_count and client details.
    Search supports: account name, client user name, client user email.
    """
    term = (search or "").strip()
    like = f"%{term}%" if term else None
    q = (
        db.query(
            ClientAccount.id.label("account_id"),
            ClientAccount.client_user_id.label("client_user_id"),
            ClientAccount.name.label("acct_name"),
            User.email.label("client_email"),
            User.name.label("client_name"),
            func.count(Person.id).label("people_count"),
        )
        .join(User, User.id == ClientAccount.client_user_id)
        .outerjoin(Person, Person.account_id == ClientAccount.id)
        .filter(
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
    )
    if like:
        q = q.filter(
            or_(
                ClientAccount.name.ilike(like),
                User.name.ilike(like),
                User.email.ilike(like),
            )
        )
    rows = (
        q.group_by(
            ClientAccount.id,
            ClientAccount.client_user_id,
            ClientAccount.name,
            User.email,
            User.name,
        )
        .order_by(func.lower(User.name).asc(), func.lower(User.email).asc())
        .limit(limit)
        .all()
    )

    return [
        ClientAccountSummary(
            account_id=row.account_id,
            client_user_id=row.client_user_id,
            client_email=row.client_email,
            client_name=row.client_name,
            name=row.acct_name,
            people_count=int(row.people_count or 0),
        )
        for row in rows
    ]


class ClientAccountDetailOut(BaseModel):
    """Owner-facing client account detail with emails and people."""

    account_id: int
    client_user_id: str
    client_email: Optional[str] = None
    client_name: Optional[str] = None
    name: Optional[str] = None
    emails: list[dict] = Field(default_factory=list)
    people: list[dict] = Field(default_factory=list)


@router.get("/owner/clients/{account_id}")
def owner_get_client_detail(
    account_id: int,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict:
    """Return an owner-scoped client account with emails and people arrays."""
    acct = (
        db.query(ClientAccount)
        .filter(ClientAccount.id == account_id, ClientAccount.owner_user_id == user.sub)
        .first()
    )
    if not acct:
        raise HTTPException(404, "Client account not found")
    u = db.query(User).filter(User.id == acct.client_user_id).first()
    emails = [
        {
            "id": e.id,
            "email": e.email,
            "is_primary": bool(e.is_primary),
            "unsubscribed": bool(getattr(e, "unsubscribed", 0)),
        }
        for e in db.query(ClientEmail).filter(ClientEmail.account_id == acct.id).all()
    ]
    people = [
        {"id": p.id, "full_name": p.full_name, "email": p.email}
        for p in db.query(Person)
        .filter(Person.account_id == acct.id)
        .order_by(Person.id.asc())
        .all()
    ]
    return {
        "account_id": acct.id,
        "client_user_id": acct.client_user_id,
        "client_email": getattr(u, "email", None),
        "client_name": getattr(u, "name", None),
        "name": acct.name,
        "emails": emails,
        "people": people,
    }


class OwnerCreatePersonPayload(BaseModel):
    """Create a person (attendee) under a client account."""

    full_name: str
    email: Optional[str] = None


@router.post("/owner/clients/{account_id}/people", response_model=dict)
def owner_create_person(
    account_id: int,
    payload: OwnerCreatePersonPayload,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict:
    """Create a person under this owner's client account and return its id.

    Use this to seed attendees for group lessons when the account has no people yet.
    """
    acct = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.id == account_id,
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .first()
    )
    if not acct:
        raise HTTPException(404, "Client account not found")
    full_name = (payload.full_name or "").strip()
    if not full_name:
        raise HTTPException(400, "full_name is required")
    person = Person(
        account_id=acct.id, full_name=full_name, email=(payload.email or None)
    )
    db.add(person)
    db.flush()
    db.commit()
    return {"id": person.id, "full_name": person.full_name, "email": person.email}


class BroadcastEmailPayload(BaseModel):
    """Broadcast email request to one or many client users.

    If 'client_user_ids' is omitted, all client users for the owner are
    targeted. When 'preview_only' is true, the endpoint returns counts
    without sending messages.
    """

    subject: str
    text: Optional[str] = None
    html: Optional[str] = None
    client_user_ids: Optional[List[str]] = None
    preview_only: bool = False
    confirm_send: bool = False


@router.post("/owner/email/broadcast", response_model=dict)
def owner_broadcast_email(
    payload: BroadcastEmailPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict:
    """Send or preview a broadcast email to selected/all client users."""
    owner: User = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    q = (
        db.query(User, ClientAccount.id.label("account_id"))
        .join(ClientAccount, ClientAccount.client_user_id == User.id)
        .filter(
            ClientAccount.owner_user_id == owner.id,
            ClientAccount.deleted_at.is_(None),
            User.role == RoleEnum.CLIENT,
        )
    )

    if payload.client_user_ids:
        q = q.filter(User.id.in_(payload.client_user_ids))

    recipients: List[tuple[User, int]] = q.all()
    emails: list[str] = []
    skipped_unsubscribed = 0
    skipped_missing = 0
    for user_row, account_id in recipients:
        chosen, unsub_only = _sendable_email_for_account(db, int(account_id))
        if chosen:
            emails.append(chosen)
            continue
        if unsub_only:
            skipped_unsubscribed += 1
        else:
            # fall back to auth.User email if available and not unsubscribed anywhere
            if user_row.email:
                emails.append(user_row.email)
            else:
                skipped_missing += 1
    # Deduplicate to avoid multi-send if account has multiple entries
    emails = list(dict.fromkeys(emails))

    if not emails:
        return {
            "ok": True,
            "recipients": 0,
            "skipped_unsubscribed": skipped_unsubscribed,
            "skipped_missing": skipped_missing,
            "preview_only": bool(payload.preview_only),
        }
    subject = payload.subject.strip()
    if not subject:
        raise HTTPException(400, "Subject is required")

    plain = (payload.text or "").strip()
    html = payload.html or _html_from_text(
        owner.name or "Owner", plain or "(no content)"
    )

    if not payload.preview_only:
        if REQUIRE_BROADCAST_CONFIRM and not payload.confirm_send:
            raise HTTPException(
                400,
                "Set confirm_send=true to proceed with broadcast (preview_only=false).",
            )
        if MAX_BROADCAST_RECIPIENTS and len(emails) > MAX_BROADCAST_RECIPIENTS:
            raise HTTPException(
                400,
                f"Too many recipients ({len(emails)}). Limit is {MAX_BROADCAST_RECIPIENTS}; narrow your selection or send in batches.",
            )

    # — Preview vs Send —
    if payload.preview_only:
        return {
            "ok": True,
            "recipients": len(emails),
            "skipped_unsubscribed": skipped_unsubscribed,
            "skipped_missing": skipped_missing,
            "preview_only": True,
        }
    for to in emails:
        background_tasks.add_task(
            send_email,
            to,
            subject,
            plain or "(no content)",
            html,
            None,
        )

    return {
        "ok": True,
        "recipients": len(emails),
        "skipped_unsubscribed": skipped_unsubscribed,
        "skipped_missing": skipped_missing,
        "preview_only": False,
    }


@router.get("/owner/clients/{client_id}/bundles", response_model=list[PrepaidBundleOut])
def list_bundles(
    client_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> list[PrepaidBundleOut]:
    """List bundles/wallets for a client under this owner, newest first."""
    _ensure_wallet(db, owner_id=user.sub, client_id=client_id)
    rows = (
        db.query(PrepaidBundle)
        .filter(
            PrepaidBundle.owner_id == user.sub, PrepaidBundle.client_id == client_id
        )
        .order_by(PrepaidBundle.created_at.desc())
        .all()
    )
    out = []
    for b in rows:
        rem_balance = None
        try:
            total = (
                db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                .filter(PrepaidLedger.bundle_id == b.id)
                .scalar()
            )
            rem_balance = int(total or 0)
        except Exception:
            rem_balance = None
        item = PrepaidBundleOut.model_validate(b, from_attributes=True)
        item.remaining_balance_cents = rem_balance
        out.append(item)
    return out


@router.get(
    "/owner/clients/{client_id}/bundles/{bundle_id}/ledger", response_model=list[dict]
)
def bundle_ledger(
    client_id: str,
    bundle_id: int,
    limit: int = Query(5, ge=1, le=200),
    date_from: date | None = Query(
        None, description="Filter ledger from this date (YYYY-MM-DD) inclusive"
    ),
    date_to: date | None = Query(
        None, description="Filter ledger until this date (YYYY-MM-DD) inclusive"
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> list[dict]:
    """Return recent ledger entries for a specific wallet/bundle.

    Date filters are treated as UTC dates and apply to 'created_at'.
    """
    b = db.get(PrepaidBundle, int(bundle_id))
    if not b or b.owner_id != user.sub or b.client_id != client_id:
        raise HTTPException(404, "Bundle not found")
    q = db.query(PrepaidLedger).filter(PrepaidLedger.bundle_id == b.id)
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
    return [
        {
            "event": r.event,
            "delta_credits": int(getattr(r, "delta_credits", 0) or 0),
            "amount_cents": int(getattr(r, "amount_cents", 0) or 0),
            "appointment_id": str(getattr(r, "appointment_id", "") or ""),
            "note": getattr(r, "note", None),
            "created_at": r.created_at,
        }
        for r in rows
    ]


@router.post("/owner/clients/{client_id}/bundles", response_model=PrepaidBundleOut)
def create_bundle(
    client_id: str,
    payload: PrepaidBundleCreate,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> PrepaidBundleOut:
    """Create a credit bundle or wallet for a client.

    Enforces a single active wallet per client. Records an initial purchase
    ledger row and triggers wallet auto-apply for wallets.
    """
    if payload.client_id != client_id:
        raise HTTPException(400, "client_id mismatch")
    existing_wallet = (
        db.query(PrepaidBundle)
        .filter(
            PrepaidBundle.owner_id == user.sub,
            PrepaidBundle.client_id == client_id,
            PrepaidBundle.total_credits == 0,
            PrepaidBundle.status == "active",
        )
        .first()
    )
    if existing_wallet and payload.total_credits == 0:
        raise HTTPException(409, "Client already has an active wallet")

    try:
        b = PrepaidBundle(
            owner_id=user.sub,
            client_id=client_id,
            name=payload.name,
            total_credits=payload.total_credits,
            remaining_credits=payload.total_credits,
            price_cents=payload.price_cents,
            currency=payload.currency,
            expires_at=payload.expires_at,
        )
        db.add(b)
        db.flush()
        db.add(
            PrepaidLedger(
                bundle_id=b.id,
                event="purchase",
                delta_credits=payload.total_credits,
                amount_cents=payload.price_cents,
                note="New bundle",
            )
        )
        db.commit()
        db.refresh(b)
    except IntegrityError:
        db.rollback()
        # Likely hit the partial unique index for active wallet creation race
        raise HTTPException(409, "Client already has an active wallet")
    except Exception:
        db.rollback()
        raise HTTPException(500, "Failed to create bundle")

    if b.total_credits == 0:
        summary = auto_apply_wallet_funds(
            db,
            owner_id=user.sub,
            bundle_id=b.id,
            note_prefix="Auto-apply wallet funds after bundle creation",
        )
        setattr(b, "remaining_balance_cents", summary.get("remaining_balance_cents"))

    return b


class TopUpPayload(BaseModel):
    amount_cents: int
    note: Optional[str] = None


class WalletAdjustPayload(BaseModel):
    amount_cents: int = Field(
        ..., description="Positive to add funds, negative to remove funds"
    )
    note: Optional[str] = None


@router.post(
    "/owner/clients/{client_id}/bundles/{bundle_id}/topup", response_model=dict
)
def topup_bundle(
    client_id: str,
    bundle_id: int,
    payload: TopUpPayload,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    if payload.amount_cents <= 0:
        raise HTTPException(400, "amount_cents must be positive")
    b = db.get(PrepaidBundle, int(bundle_id))
    if not b or b.owner_id != user.sub or b.client_id != client_id:
        raise HTTPException(404, "Bundle not found")
    db.add(
        PrepaidLedger(
            bundle_id=b.id,
            event="purchase",
            delta_credits=0,
            amount_cents=int(payload.amount_cents),
            appointment_id=None,
            note=payload.note or "Top up",
        )
    )
    db.commit()
    summary = auto_apply_wallet_funds(
        db,
        owner_id=user.sub,
        bundle_id=b.id,
        note_prefix="Auto-apply wallet funds after top-up",
    )
    return {
        "ok": True,
        "remaining_balance_cents": summary.get("remaining_balance_cents"),
    }


@router.post(
    "/owner/clients/{client_id}/bundles/{bundle_id}/adjust", response_model=dict
)
def adjust_wallet_bundle(
    client_id: str,
    bundle_id: int,
    payload: WalletAdjustPayload,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Adjust wallet balance by a signed amount with optional note.

    Positive amounts credit the wallet; negative amounts debit it. Returns the
    new wallet balance. Normalizes overdraft errors to a friendly message.
    """
    try:
        new_balance = adjust_wallet_balance(
            db,
            owner_id=user.sub,
            bundle_id=int(bundle_id),
            amount_cents=int(payload.amount_cents),
            note=payload.note,
            client_user_id=client_id,
        )
    except WalletAdjustmentError as exc:
        msg = str(exc)
        if "overdraft" in msg.lower() or "insufficient wallet" in msg.lower():
            msg = "That would overdraft the wallet. Reduce the amount and try again."
        raise HTTPException(400, msg)
    except Exception:
        raise HTTPException(500, "Failed to adjust wallet")

    return {"ok": True, "balance_cents": int(new_balance)}


@router.post(
    "/owner/clients/{client_id}/bundles/migrate-to-wallet", response_model=dict
)
def migrate_credit_bundles_to_wallet(
    client_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """
    Convert all credit-based bundles (total_credits>0) for this client into wallets by:
      - Depositing remaining_credits * (price_cents/total_credits) as amount_cents
      - Setting total_credits = 0 and remaining_credits = 0
    """
    bundles = (
        db.query(PrepaidBundle)
        .filter(
            PrepaidBundle.owner_id == user.sub,
            PrepaidBundle.client_id == client_id,
            PrepaidBundle.total_credits > 0,
        )
        .all()
    )
    wallet = (
        db.query(PrepaidBundle)
        .filter(
            PrepaidBundle.owner_id == user.sub,
            PrepaidBundle.client_id == client_id,
            PrepaidBundle.total_credits == 0,
        )
        .order_by(PrepaidBundle.created_at.desc())
        .first()
    )
    if not wallet:
        wallet = PrepaidBundle(
            owner_id=user.sub,
            client_id=client_id,
            name="Wallet",
            total_credits=0,
            remaining_credits=0,
            price_cents=0,
            currency="USD",
        )
        db.add(wallet)
        db.flush()

    migrated = 0
    deposited_total = 0
    for b in bundles:
        per = 0
        try:
            if int(b.total_credits) > 0:
                per = round(int(b.price_cents) / int(b.total_credits))
        except Exception:
            per = 0
        deposit = int(getattr(b, "remaining_credits", 0) or 0) * per
        if deposit > 0:
            db.add(
                PrepaidLedger(
                    bundle_id=wallet.id,
                    event="adjust",
                    delta_credits=0,
                    amount_cents=deposit,
                    appointment_id=None,
                    note=f"Migrate credits from bundle {b.id}",
                )
            )
            deposited_total += deposit
        b.total_credits = 0
        b.remaining_credits = 0
        b.status = "canceled"
        db.add(b)
        migrated += 1
    db.commit()
    summary = auto_apply_wallet_funds(
        db,
        owner_id=user.sub,
        bundle_id=wallet.id,
        note_prefix="Auto-apply wallet funds after bundle migration",
    )
    return {
        "ok": True,
        "migrated": migrated,
        "deposited_cents": deposited_total,
        "remaining_balance_cents": summary.get("remaining_balance_cents"),
    }


@router.put("/owner/clients/{account_id}")
def owner_update_client(
    account_id: int,
    payload: OwnerClientUpdate,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> dict:
    """Update owner-managed fields on a client account, including emails.

    Normalizes 10-digit phones to '###-###-####'. When emails are provided,
    they replace existing emails and enforce at most one primary.
    """
    acct = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.id == int(account_id),
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .first()
    )
    if not acct:
        raise HTTPException(404, "Client account not found")

    if payload.name is not None:
        acct.name = payload.name or None
    if payload.phone is not None:
        raw = (payload.phone or "").strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            if len(digits) != 10:
                raise HTTPException(
                    400, "INVALID_PHONE: expected ###-###-#### (10 digits)"
                )
            acct.phone = f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
        else:
            acct.phone = None
    if payload.emergency_contact is not None:
        acct.emergency_contact = payload.emergency_contact or None

    if payload.emails is not None:
        new_emails = []
        for item in payload.emails:
            email = (item.get("email") or "").strip()
            if not email:
                continue
            is_primary = bool(item.get("is_primary"))
            unsub = bool(item.get("unsubscribed"))
            new_emails.append(
                {"email": email, "is_primary": is_primary, "unsubscribed": unsub}
            )
        if new_emails:
            if not any(e.get("is_primary") for e in new_emails):
                new_emails[0]["is_primary"] = True
            else:
                seen_primary = False
                for e in new_emails:
                    if e.get("is_primary") and not seen_primary:
                        seen_primary = True
                    else:
                        e["is_primary"] = False
        db.query(ClientEmail).filter(ClientEmail.account_id == acct.id).delete()
        for e in new_emails:
            db.add(
                ClientEmail(
                    account_id=acct.id,
                    email=e["email"],
                    is_primary=1 if e.get("is_primary") else 0,
                    unsubscribed=1 if e.get("unsubscribed") else 0,
                )
            )

    db.add(acct)
    db.commit()
    return {"ok": True}


@router.post("/owner", response_model=OwnerOut)
def create_owner(payload: OwnerCreate, db: Session = Depends(get_db)) -> OwnerOut:
    """Create a new owner user with the provided name, email, and timezone."""
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(400, "User with this email already exists")
    owner = User(
        id=uuid_str(),
        name=payload.name,
        email=payload.email,
        role=RoleEnum.OWNER,
        timezone=payload.timezone,
        createdAt=datetime.utcnow(),
        updatedAt=datetime.utcnow(),
    )
    db.add(owner)
    db.commit()
    db.refresh(owner)
    return OwnerOut(
        id=owner.id, name=owner.name, email=owner.email, timezone=payload.timezone
    )


from sqlalchemy import case


@router.get("/owner/clients/resolve", response_model=ClientAccountSummary)
def owner_resolve_client_account(
    query: str = Query(..., description="Name or email to resolve"),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> ClientAccountSummary:
    """
    Resolve a free-text query to ONE client account belonging to this owner.
    Searches: client user name, client user email, account name.
    Returns:
      - 200: exact/best match (ClientAccountSummary)
      - 404: no match
      - 409: multiple plausible matches (detail.candidates[])
    """
    term = (query or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="query is required")

    like = f"%{term}%"
    term_lower = term.lower()

    q = (
        db.query(
            ClientAccount.id.label("account_id"),
            ClientAccount.client_user_id.label("client_user_id"),
            ClientAccount.name.label("acct_name"),
            User.email.label("client_email"),
            User.name.label("client_name"),
            func.count(Person.id).label("people_count"),
            case(
                (func.lower(User.email) == term_lower, 0),
                (func.lower(User.name) == term_lower, 1),
                (func.lower(ClientAccount.name) == term_lower, 2),
                (func.lower(User.name).like(term_lower + "%"), 3),
                (func.lower(ClientAccount.name).like(term_lower + "%"), 4),
                (func.lower(User.email).like(term_lower + "%"), 5),
                else_=6,
            ).label("rank"),
        )
        .join(User, User.id == ClientAccount.client_user_id)
        .outerjoin(Person, Person.account_id == ClientAccount.id)
        .filter(
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
            or_(
                User.email.ilike(like),
                User.name.ilike(like),
                ClientAccount.name.ilike(like),
            ),
        )
        .group_by(
            ClientAccount.id,
            ClientAccount.client_user_id,
            ClientAccount.name,
            User.email,
            User.name,
        )
        .order_by(
            "rank",
            func.lower(User.name).asc(),
            func.lower(User.email).asc(),
            ClientAccount.id.desc(),
        )
        .limit(5)
    )

    rows = q.all()
    if not rows:
        raise HTTPException(status_code=404, detail="No client matched")

    best_rank = min(r.rank for r in rows)
    top = [r for r in rows if r.rank == best_rank]

    def to_summary(r):
        return ClientAccountSummary(
            account_id=r.account_id,
            client_user_id=r.client_user_id,
            client_email=r.client_email,
            client_name=r.client_name,
            name=r.acct_name,
            people_count=int(r.people_count or 0),
        )

    if len(rows) == 1 or len(top) == 1:
        return to_summary(top[0])

    candidates = [to_summary(r).model_dump() for r in rows]
    raise HTTPException(
        status_code=409,
        detail={"message": "Multiple matches", "candidates": candidates},
    )


@router.post("/owner/bundles/cleanup-orphans", response_model=dict)
def cleanup_orphan_wallet_balances(
    dry_run: bool = Query(
        True,
        description="If true, does not write adjustments; returns what would change.",
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """
    Zero out wallet balances (total_credits == 0) for bundles whose linked client account
    was deleted or no longer exists for this owner. Adds an 'adjust' ledger entry per bundle.

    Returns: { affected: number, total_adjusted_cents: number, details: [{bundle_id, balance_cents}] }
    """
    wallets = (
        db.query(PrepaidBundle, ClientAccount)
        .outerjoin(
            ClientAccount,
            and_(
                ClientAccount.client_user_id == PrepaidBundle.client_id,
                ClientAccount.owner_user_id == PrepaidBundle.owner_id,
            ),
        )
        .filter(
            PrepaidBundle.owner_id == user.sub,
            PrepaidBundle.total_credits == 0,
        )
        .all()
    )

    affected = 0
    total_adjust = 0
    details = []

    for b, acct in wallets:
        orphan = (acct is None) or (getattr(acct, "deleted_at", None) is not None)
        if not orphan:
            continue
        bal = int(
            db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
            .filter(PrepaidLedger.bundle_id == b.id)
            .scalar()
            or 0
        )
        if bal == 0:
            continue
        details.append({"bundle_id": int(b.id), "balance_cents": bal})
        affected += 1
        total_adjust += bal
        if not dry_run:
            db.add(
                PrepaidLedger(
                    bundle_id=b.id,
                    event="adjust",
                    delta_credits=0,
                    amount_cents=-bal,
                    appointment_id=None,
                    note="Cleanup orphan wallet balance",
                )
            )

    if not dry_run and affected:
        db.commit()

    return {
        "affected": affected,
        "total_adjusted_cents": int(total_adjust),
        "details": details,
        "dry_run": bool(dry_run),
    }


@router.post("/owner/bundles/delete-orphans", response_model=dict)
def delete_orphan_bundles(
    dry_run: bool = Query(
        True, description="If true, preview deletions without writing"
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """
    Hard-delete orphan bundles (where the linked client account no longer exists or is deleted).
    This removes PrepaidLedger rows for those bundles and then deletes the PrepaidBundle rows.

    WARNING: This is destructive and removes audit history for those bundles.
             Prefer the cleanup-orphans endpoint to zero balances while preserving history.
    """
    rows = (
        db.query(PrepaidBundle, ClientAccount)
        .outerjoin(
            ClientAccount,
            and_(
                ClientAccount.client_user_id == PrepaidBundle.client_id,
                ClientAccount.owner_user_id == PrepaidBundle.owner_id,
            ),
        )
        .filter(PrepaidBundle.owner_id == user.sub)
        .all()
    )
    orphans: list[PrepaidBundle] = []
    for b, acct in rows:
        orphan = (acct is None) or (getattr(acct, "deleted_at", None) is not None)
        if orphan:
            orphans.append(b)

    if dry_run:
        balances = []
        for b in orphans:
            bal = int(
                db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
                .filter(PrepaidLedger.bundle_id == b.id)
                .scalar()
                or 0
            )
            balances.append(
                {
                    "bundle_id": int(b.id),
                    "client_id": str(b.client_id),
                    "is_wallet": int(b.total_credits or 0) == 0,
                    "balance_cents": bal,
                }
            )
        return {"dry_run": True, "count": len(orphans), "bundles": balances}
    deleted = 0
    for b in orphans:
        db.query(PrepaidLedger).filter(PrepaidLedger.bundle_id == b.id).delete()
        db.delete(b)
        deleted += 1
    if deleted:
        db.commit()
    return {"ok": True, "deleted": deleted}


@router.post("/owner/bundles/reset-wallets", response_model=dict)
def reset_wallet_balances(
    dry_run: bool = Query(True, description="If true, preview without writing"),
    client_account_id: int | None = Query(
        None, description="Optional: limit to a single client account id"
    ),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """
    Zero current balances for ALL wallets (store credit bundles, i.e., total_credits == 0) for this owner,
    or only for a single client account if "client_account_id" is provided.

    Implementation: inserts a single PrepaidLedger(event='adjust', amount_cents=-balance) per wallet so that
    the sum for that bundle becomes zero. Credit-based bundles (total_credits > 0) are not changed.

    WARNING: This is irreversible. Use dry_run first to see what will change.
    """
    q = (
        db.query(PrepaidBundle, ClientAccount)
        .outerjoin(
            ClientAccount,
            and_(
                ClientAccount.client_user_id == PrepaidBundle.client_id,
                ClientAccount.owner_user_id == PrepaidBundle.owner_id,
            ),
        )
        .filter(
            PrepaidBundle.owner_id == user.sub,
            PrepaidBundle.total_credits == 0,
        )
    )
    if client_account_id is not None:
        q = q.filter(ClientAccount.id == int(client_account_id))
    wallets = q.all()

    details = []
    total_adjust = 0
    for b, acct in wallets:
        bal = int(
            db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
            .filter(PrepaidLedger.bundle_id == b.id)
            .scalar()
            or 0
        )
        if bal == 0:
            continue
        details.append(
            {
                "bundle_id": int(b.id),
                "client_id": str(b.client_id),
                "client_account_id": int(getattr(acct, "id", 0)) if acct else None,
                "balance_cents": bal,
            }
        )
        total_adjust += bal
        if not dry_run:
            db.add(
                PrepaidLedger(
                    bundle_id=b.id,
                    event="adjust",
                    delta_credits=0,
                    amount_cents=-bal,
                    appointment_id=None,
                    note="Admin reset wallet balance",
                )
            )

    if not dry_run and total_adjust:
        db.commit()

    return {
        "dry_run": bool(dry_run),
        "affected": len(details),
        "total_adjusted_cents": int(total_adjust),
        "details": details,
    }
