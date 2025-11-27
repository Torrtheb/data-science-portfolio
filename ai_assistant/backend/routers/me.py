from fastapi import APIRouter, Depends, HTTPException, Body, Response
from sqlalchemy.orm import Session, joinedload
from app.db import SessionLocal
from app.core.auth import TokenUser, get_current_user
from app.models import User, ClientAccount, Person, ClientEmail
from app.schemas import (
    ProfileOut,
    PersonCreate,
    PersonOut,
    ClientProfileOut,
    ClientProfileUpdate,
)
from sqlalchemy import func

router = APIRouter(prefix="/api/me", tags=["me"])


def get_db():
    """FastAPI dependency that yields a SQLAlchemy Session and closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/profile", response_model=ProfileOut)
def my_profile(
    db: Session = Depends(get_db), user: TokenUser = Depends(get_current_user)
) -> ProfileOut:
    """Return a minimal authenticated user profile with basic fields."""
    db_user = db.query(User).filter(User.id == user.sub).first()
    return ProfileOut(
        user_id=user.sub,
        email=db_user.email if db_user else None,
        name=db_user.name if db_user else None,
        people=[],
    )


def _find_account_by_user_email(db: Session, email: str | None):
    """Find a ClientAccount by a client email address.

    Searches the 'client_emails' table for a case-insensitive match and returns
    the most recent client account that contains it, ignoring soft-deleted rows.
    """
    if not email:
        return None
    email_lc = email.strip().lower()
    acc = (
        db.query(ClientAccount)
        .join(ClientEmail, ClientEmail.account_id == ClientAccount.id)
        .filter(
            ClientAccount.deleted_at.is_(None),
            func.lower(ClientEmail.email) == email_lc,
        )
        .order_by(ClientAccount.id.desc())
        .first()
    )
    return acc


@router.get("/client-profile", response_model=ClientProfileOut)
def get_client_profile(
    db: Session = Depends(get_db), user: TokenUser = Depends(get_current_user)
) -> ClientProfileOut:
    """Return the client account profile including emails and people.

    If an account with 'client_user_id' is not found, attempts to locate an
    account by the user’s email and link it to the user.
    """
    acc = (
        db.query(ClientAccount)
        .options(joinedload(ClientAccount.people), joinedload(ClientAccount.emails))
        .filter(
            ClientAccount.client_user_id == user.sub, ClientAccount.deleted_at.is_(None)
        )
        .first()
    )
    if not acc:
        acc = _find_account_by_user_email(db, user.email)
        if acc:
            acc.client_user_id = user.sub
            db.commit()
            db.refresh(acc)

    if not acc:
        return ClientProfileOut(
            account_id=0,
            name=None,
            phone=None,
            emergency_contact=None,
            emails=[],
            people=[],
        )

    emails = [
        dict(id=e.id, email=e.email, is_primary=bool(e.is_primary)) for e in acc.emails
    ]
    people = [
        PersonOut(id=p.id, full_name=p.full_name, email=p.email) for p in acc.people
    ]
    return ClientProfileOut(
        account_id=acc.id,
        name=acc.name,
        phone=acc.phone,
        emergency_contact=acc.emergency_contact,
        emails=emails,  # type: ignore
        people=people,
    )


@router.put("/client-profile", response_model=ClientProfileOut)
def update_client_profile(
    payload: ClientProfileUpdate = Body(...),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(get_current_user),
) -> ClientProfileOut:
    """Update client profile scalars and emails; returns updated profile.

    Rules:
      - 'name', 'phone', and 'emergency_contact' are sanitized; phone expects
        10 digits and is formatted as '###-###-####'.
      - Up to 2 emails are accepted; replaces existing emails when provided.
      - If no account is present but found by email, links the account.

    Raises:
      - HTTPException 400 when no account can be found for the user or on
        invalid phone/email constraints.
    """
    acc = (
        db.query(ClientAccount)
        .options(joinedload(ClientAccount.people), joinedload(ClientAccount.emails))
        .filter(
            ClientAccount.client_user_id == user.sub, ClientAccount.deleted_at.is_(None)
        )
        .first()
    )

    if not acc:
        acc = _find_account_by_user_email(db, user.email)
        if acc:
            acc.client_user_id = user.sub
            db.commit()
            db.refresh(acc)

    if not acc:
        raise HTTPException(
            status_code=400,
            detail="No client account found. Ask the owner to attach your account first.",
        )
    if payload.name is not None:
        acc.name = (payload.name or "").strip() or None
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
        acc.emergency_contact = (payload.emergency_contact or "").strip() or None
    if payload.emails is not None:
        if len(payload.emails) > 2:
            raise HTTPException(
                status_code=400, detail="At most two emails are allowed"
            )
        db.query(ClientEmail).filter(ClientEmail.account_id == acc.id).delete()
        for e in payload.emails:
            email_str = str(e.email).lower().strip()
            if not email_str:
                continue
            db.add(
                ClientEmail(
                    account_id=acc.id,
                    email=email_str,
                    is_primary=1 if e.is_primary else 0,
                )
            )

    db.commit()
    db.refresh(acc)
    return get_client_profile(db, user)


@router.get("/people", response_model=list[PersonOut])
def list_people(
    db: Session = Depends(get_db), user: TokenUser = Depends(get_current_user)
) -> list[PersonOut]:
    """List people (lesson attendees) under the caller’s client account."""
    account = (
        db.query(ClientAccount).filter(ClientAccount.client_user_id == user.sub).first()
    )
    if not account:
        return []
    people = db.query(Person).filter(Person.account_id == account.id).all()
    return [PersonOut(id=p.id, full_name=p.full_name, email=p.email) for p in people]


@router.post("/people", response_model=PersonOut)
def add_person(
    payload: PersonCreate = Body(...),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(get_current_user),
) -> PersonOut:
    """Add a person under the caller’s client account and return the record."""
    account = (
        db.query(ClientAccount).filter(ClientAccount.client_user_id == user.sub).first()
    )
    if not account:
        raise HTTPException(400, "No client account found for user")
    person = Person(
        account_id=account.id, full_name=payload.full_name, email=payload.email
    )
    db.add(person)
    db.commit()
    db.refresh(person)
    return PersonOut(id=person.id, full_name=person.full_name, email=person.email)


@router.delete("/people/{person_id}", status_code=204)
def delete_person(
    person_id: int,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(get_current_user),
) -> Response:
    """Delete a person under the caller’s client account by id."""
    account = (
        db.query(ClientAccount).filter(ClientAccount.client_user_id == user.sub).first()
    )
    if not account:
        raise HTTPException(400, "No client account found for user")
    p = (
        db.query(Person)
        .filter(Person.id == person_id, Person.account_id == account.id)
        .first()
    )
    if not p:
        raise HTTPException(404, "Person not found")
    db.delete(p)
    db.commit()
    return Response(status_code=204)
