from __future__ import annotations
from fastapi import APIRouter, Depends, Query, HTTPException, Response, Body
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
import uuid

from app.db import SessionLocal
from app.core.auth import require_owner, TokenUser
from app.models import ClientAccount, User, Person, ClientEmail, PrepaidBundle
from app.schemas import (
    ClientAccountSummary,
    ClientAccountDetail,
    ClientCreate,
    ClientEmailOut,
    ClientProfileUpdate,
    PersonCreate,
    PersonOut,
)

router = APIRouter(prefix="/owner", tags=["owner"])


def get_db():
    """FastAPI dependency yielding a SQLAlchemy session and ensuring cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----------------------------
# Helpers
# ----------------------------
def _uuid() -> str:
    """Return a random UUID string (helper for new ids)."""
    return str(uuid.uuid4())


def _owner_timezone(db: Session, owner_id: str) -> str:
    """Fetch the owner's timezone or a sensible default."""
    owner = db.query(User).filter(User.id == owner_id).first()
    return owner.timezone if owner and owner.timezone else "America/Toronto"


def _ensure_wallet(db: Session, *, owner_id: str, client_id: str) -> PrepaidBundle:
    """Ensure a zero-credit wallet exists for this owner/client pair."""
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


# ----------------------------
# List clients (optionally include email/name + search by email)
# ----------------------------
@router.get("/clients", response_model=list[ClientAccountSummary])
def list_clients(
    include_people: bool = Query(False),
    q: str | None = Query(
        None, description="Search by client email (contains, case-insensitive)"
    ),
    limit: int = 50,
    offset: int = 0,
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """List client accounts for the owner with optional people and search.

    - 'include_people': loads people to compute counts in a single query.
    - 'q': case-insensitive filter on client email.
    - Supports simple pagination via 'limit' and 'offset'.
    """
    base = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .order_by(ClientAccount.id.desc())
    )

    if include_people:
        base = base.options(joinedload(ClientAccount.people))

    accounts = base.limit(limit).offset(offset).all()
    client_ids = [acc.client_user_id for acc in accounts]
    users_by_id = {}
    if client_ids:
        q_users = db.query(User).filter(User.id.in_(client_ids)).all()
        users_by_id = {u.id: u for u in q_users}
    if q:
        q_lower = q.lower()
        accounts = [
            acc
            for acc in accounts
            if (
                users_by_id.get(acc.client_user_id)
                and users_by_id[acc.client_user_id].email
                and q_lower in users_by_id[acc.client_user_id].email.lower()
            )
        ]

    out: list[ClientAccountSummary] = []
    for acc in accounts:
        u = users_by_id.get(acc.client_user_id)
        out.append(
            ClientAccountSummary(
                account_id=acc.id,
                client_user_id=acc.client_user_id or None,
                client_email=(u.email if u else None),
                client_name=(u.name if u else None),
                people_count=(len(acc.people) if include_people else len(acc.people)),
                name=acc.name,
            )
        )
    return out


# ----------------------------
# Get one client (detail)
# ----------------------------
@router.get("/clients/{account_id}", response_model=ClientAccountDetail)
def get_client_detail(
    account_id: int,
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Return owner-scoped client account detail including emails and people."""
    acc = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.id == account_id,
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .options(joinedload(ClientAccount.people), joinedload(ClientAccount.emails))
        .first()
    )
    if not acc:
        raise HTTPException(status_code=404, detail="Client account not found")

    u = db.query(User).filter(User.id == acc.client_user_id).first()
    emails_list = [
        ClientEmailOut(id=e.id, email=e.email, is_primary=bool(e.is_primary))
        for e in acc.emails
    ]
    if not emails_list and u and u.email:
        emails_list = [ClientEmailOut(id=0, email=u.email, is_primary=True)]

    return ClientAccountDetail(
        account_id=acc.id,
        client_user_id=acc.client_user_id,
        client_email=(u.email if u else None),
        client_name=(u.name if u else None),
        name=acc.name,
        phone=acc.phone,
        emergency_contact=acc.emergency_contact,
        emails=emails_list,
        people=[
            {"id": p.id, "full_name": p.full_name, "email": p.email} for p in acc.people
        ],
    )


@router.put("/clients/{account_id}", response_model=ClientAccountDetail)
def update_client_detail(
    account_id: int,
    payload: ClientProfileUpdate = Body(...),
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Update owner-managed fields on a client account and replace emails.

    Phone numbers are normalized to '###-###-####' when 10 digits are provided.
    At most two emails are accepted and a single primary is enforced.
    """
    acc = (
        db.query(ClientAccount)
        .options(joinedload(ClientAccount.people), joinedload(ClientAccount.emails))
        .filter(
            ClientAccount.id == account_id,
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .first()
    )
    if not acc:
        raise HTTPException(status_code=404, detail="Client account not found")
    if payload.name is not None:
        acc.name = payload.name.strip() or None
    if payload.phone is not None:
        raw = (payload.phone or "").strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            if len(digits) != 10:
                raise HTTPException(
                    status_code=400,
                    detail="INVALID_PHONE: expected ###-###-#### (10 digits)",
                )
            acc.phone = f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
        else:
            acc.phone = None
    if payload.emergency_contact is not None:
        acc.emergency_contact = payload.emergency_contact.strip() or None
    if payload.emails is not None:
        if len(payload.emails) > 2:
            raise HTTPException(
                status_code=400, detail="At most two emails are allowed"
            )
        db.query(ClientEmail).filter(ClientEmail.account_id == acc.id).delete()
        for e in payload.emails:
            db.add(
                ClientEmail(
                    account_id=acc.id,
                    email=str(e.email).lower(),
                    is_primary=1 if e.is_primary else 0,
                )
            )

    db.commit()
    db.refresh(acc)
    return get_client_detail(account_id, user, db)


# ----------------------------
# Create (or revive) a client under this owner
# ----------------------------
@router.post("/clients", response_model=ClientAccountSummary, status_code=201)
def create_client(
    payload: ClientCreate,
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Attach an existing auth user as a client under this owner (or revive).

    If a soft-deleted account mapping exists, it is revived; otherwise a new
    mapping is created. Ensures there is a wallet bundle for the pair.
    """
    client = db.query(User).filter(User.email == payload.email).first()
    if not client:
        raise HTTPException(
            status_code=400,
            detail="Client not provisioned. Create user in NextAuth first, then attach.",
        )

    if payload.name and (client.name or "") != payload.name:
        client.name = payload.name
        db.flush()
    acc = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.client_user_id == client.id,
        )
        .first()
    )

    if acc and acc.deleted_at is None:
        raise HTTPException(
            status_code=409, detail="Client already exists for this owner"
        )

    if acc and acc.deleted_at is not None:
        acc.deleted_at = None
    else:
        acc = ClientAccount(owner_user_id=user.sub, client_user_id=client.id)
        db.add(acc)

    db.commit()
    db.refresh(acc)
    _ensure_wallet(db, owner_id=user.sub, client_id=client.id)

    people_count = db.query(Person).filter(Person.account_id == acc.id).count()

    return ClientAccountSummary(
        account_id=acc.id,
        client_user_id=acc.client_user_id,
        client_email=client.email,
        client_name=client.name,
        people_count=people_count,
    )


# ----------------------------
# Delete (soft-delete) a client account under this owner
# ----------------------------
@router.delete("/clients/{account_id}", status_code=204)
def delete_client(
    account_id: int,
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Soft-delete a client account for this owner and remove associated people."""
    acc = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.id == account_id,
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .first()
    )
    if not acc:
        raise HTTPException(status_code=404, detail="Client account not found")
    db.query(Person).filter(Person.account_id == acc.id).delete()
    acc.deleted_at = datetime.utcnow()
    db.commit()
    return Response(status_code=204)


# ----------------------------
# Owner can add a person to a client account
# ----------------------------
@router.post("/clients/{account_id}/people", response_model=PersonOut, status_code=201)
def owner_add_person(
    account_id: int,
    payload: PersonCreate,
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Add a person (attendee) under a client account and return the record."""
    acc = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.id == account_id,
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .first()
    )
    if not acc:
        raise HTTPException(status_code=404, detail="Client account not found")

    p = Person(account_id=acc.id, full_name=payload.full_name, email=payload.email)
    db.add(p)
    db.commit()
    db.refresh(p)
    return PersonOut(id=p.id, full_name=p.full_name, email=p.email)


# ----------------------------
# Owner can remove a person from a client account
# ----------------------------
@router.delete("/clients/{account_id}/people/{person_id}", status_code=204)
def owner_delete_person(
    account_id: int,
    person_id: int,
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Remove a person from a client account by id."""
    acc = (
        db.query(ClientAccount)
        .filter(
            ClientAccount.id == account_id,
            ClientAccount.owner_user_id == user.sub,
            ClientAccount.deleted_at.is_(None),
        )
        .first()
    )
    if not acc:
        raise HTTPException(status_code=404, detail="Client account not found")

    p = (
        db.query(Person)
        .filter(Person.id == person_id, Person.account_id == acc.id)
        .first()
    )
    if not p:
        raise HTTPException(status_code=404, detail="Person not found")
    db.delete(p)
    db.commit()
    return Response(status_code=204)
