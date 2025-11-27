from __future__ import annotations
from typing import Optional, List, Dict, Any
from sqlalchemy import or_, and_
from langchain_core.tools import tool, ToolException
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from app.db import SessionLocal
from sqlalchemy.orm import Session
from app.models import (
    User,
    ClientAccount,
    Person,
    ClientEmail,
)
from sqlalchemy import func


def _normalize_email(val: Optional[str]) -> Optional[str]:
    """Trim an email string and return None if empty.

    Args:
        val: Candidate email string.

    Returns:
        Normalized email or None when blank.
    """
    if not val:
        return None
    s = val.strip()
    return s or None


def _get_account_emails(db: Session, account_id: int) -> List[ClientEmail]:
    """Return account emails ordered primary-first then by id.

    Args:
        db: Database session.
        account_id: ClientAccount id.

    Returns:
        List of ClientEmail rows.
    """
    return (
        db.query(ClientEmail)
        .filter(ClientEmail.account_id == account_id)
        .order_by(ClientEmail.is_primary.desc(), ClientEmail.id.asc())
        .all()
    )


def _upsert_account_emails(
    db: Session,
    account_id: int,
    primary_email: Optional[str],
    secondary_email: Optional[str],
) -> None:
    """Upsert up to two ClientEmail rows, ensuring exactly one primary.

    Args:
        db: Database session.
        account_id: ClientAccount id.
        primary_email: Primary email to set or None.
        secondary_email: Secondary email to set or None.
    """
    primary_email = _normalize_email(primary_email)
    secondary_email = _normalize_email(secondary_email)

    rows = _get_account_emails(db, account_id)
    existing_by_email = {r.email.lower(): r for r in rows if r.email}

    desired = []
    if primary_email:
        desired.append(("primary", primary_email))
    if secondary_email and (
        not primary_email or secondary_email.lower() != primary_email.lower()
    ):
        desired.append(("secondary", secondary_email))
    seen = set()
    for kind, email in desired[:2]:
        key = email.lower()
        if key in existing_by_email:
            row = existing_by_email[key]
            row.is_primary = 1 if kind == "primary" else 0
            seen.add(row.id)
        else:
            row = ClientEmail(
                account_id=account_id,
                email=email,
                is_primary=1 if kind == "primary" else 0,
            )
            db.add(row)
            db.flush()
            seen.add(row.id)
    rows = _get_account_emails(db, account_id)
    primaries = [r for r in rows if int(r.is_primary) == 1]
    if not primaries and rows:
        rows[0].is_primary = 1
        primaries = [rows[0]]
    if len(primaries) > 1:
        for r in primaries[1:]:
            r.is_primary = 0
    rows = _get_account_emails(db, account_id)
    while len(rows) > 2:
        victim = rows[-1]
        db.delete(victim)
        db.flush()
        rows = _get_account_emails(db, account_id)


def _owner_id_from_config(config: RunnableConfig) -> str:
    """Extract owner id from tool config.

    Args:
        config: Runnable configuration with a 'configurable' sub-dict.

    Returns:
        Owner id as a string.

    Raises:
        ToolException: If no owner id is present.
    """
    cfg = (
        (config or {}).get("configurable", {})
        if isinstance(config, dict)
        else (getattr(config, "configurable", None) or {})
    )
    owner_id = cfg.get("user_id") or cfg.get("owner_id")
    if not owner_id:
        raise ToolException("Missing owner id in tool config")
    return str(owner_id)


def _get_cols(mdl: Any) -> set[str]:
    """Best-effort column name extraction for a mapped class.

    Returns an empty set for non-mapped inputs.
    """
    try:
        table = getattr(mdl, "__table__", None)
        if table is None:
            return set()
        return {c.name for c in table.columns}
    except Exception:
        return set()


def _get_first_email(client_profile: Any) -> Optional[str]:
    """Return the primary email from a profile or related emails if present."""
    if hasattr(client_profile, "email") and client_profile.email:
        return client_profile.email
    if hasattr(client_profile, "primary_email") and client_profile.primary_email:
        return client_profile.primary_email
    emails = getattr(client_profile, "emails", None)
    if emails:
        for e in emails:
            if getattr(e, "is_primary", False) and getattr(e, "email", None):
                return e.email
        if getattr(emails[0], "email", None):
            return emails[0].email
    return None


def _get_first_phone(client_profile: Any) -> Optional[str]:
    """Return a phone from a profile or related phones list if present."""
    if hasattr(client_profile, "phone") and client_profile.phone:
        return client_profile.phone
    phones = getattr(client_profile, "phones", None)
    if phones:
        for p in phones:
            if getattr(p, "is_primary", False) and getattr(p, "phone", None):
                return p.phone
        if getattr(phones[0], "phone", None):
            return phones[0].phone
    return None


def _profile_dict(client_profile: Any) -> Dict[str, Any]:
    """Normalize a variety of client profile shapes into a stable dict."""
    out: Dict[str, Any] = {}
    if hasattr(client_profile, "id"):
        out["id"] = str(client_profile.id)
    if hasattr(client_profile, "name") and client_profile.name:
        out["name"] = client_profile.name
    elif hasattr(client_profile, "full_name") and client_profile.full_name:
        out["name"] = client_profile.full_name
    out["primary_email"] = _get_first_email(client_profile)
    out["primary_phone"] = _get_first_phone(client_profile)
    for attr in ("notes", "owner_note", "client_note"):
        if hasattr(client_profile, attr) and getattr(client_profile, attr):
            out[attr] = getattr(client_profile, attr)
    return out


def _query_client_profiles(
    db: Session,
    owner_id: str,
    query: Optional[str] = None,
    include_people: bool = False,
) -> List[Dict[str, Any]]:
    """Return client ACCOUNT rows for this owner with a display dict.

    Shape:
      {
        id (account_id as str),
        user_id (auth.User.id),
        name (ACCOUNT/USER name only, never a Person),
        primary_email,
        primary_phone,
        people?: [{person_id, full_name, email}]
      }

    Search can match: account name, user name/email, or person full_name (but we still keep the account identity).
    """
    q = (
        db.query(ClientAccount, User)
        .join(User, ClientAccount.client_user_id == User.id)
        .filter(
            ClientAccount.owner_user_id == owner_id,
            ClientAccount.deleted_at.is_(None),
        )
    )

    if query:
        like = f"%{query}%"
        # allow searching by person full_name, but don't use it for display name
        q = (
            q.outerjoin(Person, Person.account_id == ClientAccount.id)
            .filter(
                or_(
                    User.name.ilike(like),
                    User.email.ilike(like),
                    ClientAccount.name.ilike(like),
                    Person.full_name.ilike(like),
                )
            )
            .order_by(ClientAccount.id.asc(), Person.id.asc())
        )

    rows = q.limit(500).all()

    # Preload people only once if requested
    people_by_account: Dict[int, List[Dict[str, Any]]] = {}
    if include_people:
        ppl_rows = (
            db.query(Person)
            .join(ClientAccount, ClientAccount.id == Person.account_id)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                ClientAccount.deleted_at.is_(None),
            )
            .order_by(Person.id.asc())
            .all()
        )
        for p in ppl_rows:
            people_by_account.setdefault(int(p.account_id), []).append(
                {"person_id": str(p.id), "full_name": p.full_name, "email": p.email}
            )

    seen: Dict[int, Dict[str, Any]] = {}
    for acct, user in rows:
        aid = int(acct.id)
        if aid in seen:
            continue
        display_name = acct.name or user.name
        emails = _get_account_emails(db, acct.id)
        primary_email = emails[0].email if emails else user.email

        item = {
            "id": str(acct.id),
            "user_id": str(acct.client_user_id),
            "name": display_name,
            "primary_email": primary_email,
            "primary_phone": acct.phone,
        }
        if include_people:
            item["people"] = people_by_account.get(aid, [])

        seen[aid] = item

    return list(seen.values())


def _account_profile_with_people(
    db: Session, owner_id: str, account_id: int
) -> Dict[str, Any]:
    """Return a single account profile including people list for the owner."""
    acct = (
        db.query(ClientAccount)
        .filter(
            and_(
                ClientAccount.owner_user_id == owner_id,
                ClientAccount.id == account_id,
                ClientAccount.deleted_at.is_(None),
            )
        )
        .first()
    )
    if not acct:
        return {}
    user = db.query(User).filter(User.id == acct.client_user_id).first()
    emails = _get_account_emails(db, acct.id)
    primary_email = emails[0].email if emails else (user.email if user else None)
    ppl = (
        db.query(Person)
        .filter(Person.account_id == acct.id)
        .order_by(Person.id.asc())
        .all()
    )
    people = [
        {"person_id": str(p.id), "full_name": p.full_name, "email": p.email}
        for p in ppl
    ]

    item = {
        "id": str(acct.id),
        "user_id": str(acct.client_user_id) if acct.client_user_id else None,
        "name": (acct.name or (user.name if user else None)),
        "primary_email": primary_email,
        "primary_phone": acct.phone,
        "people": people,
    }
    if len(people) == 1:
        item["person_id"] = people[0]["person_id"]
        item["person_name"] = people[0]["full_name"]
    return item


class ToolListClientsIn(BaseModel):
    limit: Optional[int] = Field(50, ge=1, le=200)
    query: Optional[str] = None
    include_people: bool = False


@tool("list_clients", args_schema=ToolListClientsIn, return_direct=False)
def list_clients_tool(
    limit: Optional[int] = 50,
    query: Optional[str] = None,
    include_people: bool = False,
    config: RunnableConfig = None,
) -> List[Dict[str, Any]]:
    """List up to 'limit' client accounts for this owner.

    Args:
        limit: Maximum number of accounts to return (default 50).
        query: Optional search string matching account/user/person fields.
        include_people: If True, include a 'people' array per account.
        config: Runnable configuration providing the owner id.

    Returns:
        List of account dicts with 'id', 'user_id', 'name',
        'primary_email', 'primary_phone', and optional 'people'.

    Raises:
        ToolException: If the owner id is missing from configuration.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        items = _query_client_profiles(
            db, owner_id, query=query, include_people=include_people
        )
        return items[: int(limit or 50)]


class ToolFindClientIn(BaseModel):
    selector: str = Field(..., description="Name or email")


@tool("find_client", args_schema=ToolFindClientIn, return_direct=False)
def find_client_tool(selector: str, config: RunnableConfig) -> Dict[str, Any]:
    """Find a single client ACCOUNT by name/email/id or by an associated PERSON.

    Behavior:
    - If a PERSON matches, returns the ACCOUNT profile and annotates
      'match_kind='person'' plus person details.
    - If the ACCOUNT matches, annotates 'match_kind='account''.
    - On ambiguity, returns '{"matches": [...]}' with ACCOUNT profiles.

    Args:
        selector: Name or email to search.
        config: Runnable configuration providing the owner id.

    Returns:
        Either a single profile dict or an object with 'matches'.

    Raises:
        ToolException: If the selector is empty.
    """
    owner_id = _owner_id_from_config(config)
    sel = (selector or "").strip()
    if not sel:
        raise ToolException("Empty selector")

    with SessionLocal() as db:
        s = sel.lower()
        if "@" in sel and "." in sel:
            person = (
                db.query(Person)
                .join(ClientAccount, ClientAccount.id == Person.account_id)
                .filter(
                    ClientAccount.owner_user_id == owner_id,
                    ClientAccount.deleted_at.is_(None),
                    func.lower(Person.email) == func.lower(sel),
                )
                .first()
            )
            if person:
                prof = _account_profile_with_people(
                    db, owner_id, int(person.account_id)
                )
                if prof:
                    prof["match_kind"] = "person"
                    prof["person_id"] = str(person.id)
                    prof["person_name"] = person.full_name
                    return prof

            acct = (
                db.query(ClientAccount)
                .join(ClientEmail, ClientEmail.account_id == ClientAccount.id)
                .filter(
                    ClientAccount.owner_user_id == owner_id,
                    ClientAccount.deleted_at.is_(None),
                    func.lower(ClientEmail.email) == func.lower(sel),
                )
                .first()
            )
            if acct:
                prof = _account_profile_with_people(db, owner_id, int(acct.id))
                prof["match_kind"] = "account"
                if prof.get("people") and len(prof["people"]) == 1:
                    prof["person_id"] = prof["people"][0]["person_id"]
                    prof["person_name"] = prof["people"][0]["full_name"]
                return prof
        accounts = _query_client_profiles(db, owner_id, query=sel, include_people=True)
        if not accounts:
            return {}
        for acc in accounts:
            if s == str(acc.get("id", "")).lower():
                acc["match_kind"] = "account"
                acc.setdefault("people", acc.get("people", []))
                if len(acc["people"]) == 1:
                    acc["person_id"] = acc["people"][0]["person_id"]
                    acc["person_name"] = acc["people"][0]["full_name"]
                return acc

        # exact primary email on account
        for acc in accounts:
            if (acc.get("primary_email") or "").lower() == s:
                acc["match_kind"] = "account"
                acc.setdefault("people", acc.get("people", []))
                if len(acc["people"]) == 1:
                    acc["person_id"] = acc["people"][0]["person_id"]
                    acc["person_name"] = acc["people"][0]["full_name"]
                return acc

        # exact account name
        for acc in accounts:
            if (acc.get("name") or "").strip().lower() == s:
                acc["match_kind"] = "account"
                acc.setdefault("people", acc.get("people", []))
                if len(acc["people"]) == 1:
                    acc["person_id"] = acc["people"][0]["person_id"]
                    acc["person_name"] = acc["people"][0]["full_name"]
                return acc

        # startswith account name/email
        starts = [
            acc
            for acc in accounts
            if (acc.get("name") or "").lower().startswith(s)
            or (acc.get("primary_email") or "").lower().startswith(s)
        ]
        if len(starts) == 1:
            starts[0]["match_kind"] = "account"
            return starts[0]

        # person hits (id/name/email contains)
        person_hits: List[Dict[str, Any]] = []
        for acc in accounts:
            for p in acc.get("people", []) or []:
                fields = [
                    str(p.get("person_id", "")),
                    p.get("full_name", ""),
                    p.get("email", ""),
                ]
                if any(s in (f or "").lower() for f in fields):
                    hit = dict(acc)
                    hit["match_kind"] = "person"
                    hit["person_id"] = p.get("person_id")
                    hit["person_name"] = p.get("full_name")
                    person_hits.append(hit)

        if len(person_hits) == 1:
            return person_hits[0]

        return {"matches": (starts or accounts)[:5]}


class ToolUpdateClientIn(BaseModel):
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    account_name: Optional[str] = None
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    primary_email: Optional[str] = None
    secondary_email: Optional[str] = None
    add_person_name: Optional[str] = None
    add_person_email: Optional[str] = None
    note: Optional[str] = None


@tool("update_client", args_schema=ToolUpdateClientIn, return_direct=False)
def update_client_tool(
    client_id: Optional[str] = None,
    user_id: Optional[str] = None,
    account_name: Optional[str] = None,
    phone: Optional[str] = None,
    emergency_contact: Optional[str] = None,
    name: Optional[str] = None,
    email: Optional[str] = None,
    primary_email: Optional[str] = None,
    secondary_email: Optional[str] = None,
    add_person_name: Optional[str] = None,
    add_person_email: Optional[str] = None,
    note: Optional[str] = None,
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """Update client details for an account or linked user.

    Target the account by 'client_id' (ClientAccount.id) or the linked
    'user_id' (auth.User.id). Updates account fields, user fields, primary
    and secondary emails (stored in ClientEmail), and can add a new Person.

    Args:
        client_id: ClientAccount.id (int in string form allowed).
        user_id: auth.User.id alternative to client_id.
        account_name: New account display name.
        phone: Phone number (validated to ###-###-####).
        emergency_contact: Optional emergency contact string.
        name: User display name.
        email: User email (backwards compat).
        primary_email: Primary account email address.
        secondary_email: Secondary account email address.
        add_person_name: Optional person to add under the account.
        add_person_email: Optional email for the new person.
        note: Free-form note (ignored if column missing).
        config: Runnable configuration providing the owner id.

    Returns:
        Dict with normalized account/user fields and optionally 'added_person'.

    Raises:
        ToolException: If the account/user cannot be resolved, validation fails,
        or ownership is violated.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        # Coerce IDs
        acct_id_int = None
        if client_id is not None:
            try:
                acct_id_int = int(str(client_id).strip())
            except Exception:
                acct_id_int = None
        user_id_val = str(user_id).strip() if user_id is not None else None

        # Resolve account
        acct = None
        if acct_id_int is not None:
            acct = (
                db.query(ClientAccount)
                .filter(
                    and_(
                        ClientAccount.owner_user_id == owner_id,
                        ClientAccount.id == acct_id_int,
                    )
                )
                .first()
            )
        elif user_id_val:
            acct = (
                db.query(ClientAccount)
                .filter(
                    and_(
                        ClientAccount.owner_user_id == owner_id,
                        ClientAccount.client_user_id == user_id_val,
                    )
                )
                .first()
            )
        if not acct:
            raise ToolException("Client not found for this owner")

        # Load auth user
        user = db.query(User).filter(User.id == acct.client_user_id).first()
        if not user:
            raise ToolException("Client user not found")

        # Normalize + validate phone to ###-###-#### (reject invalid)
        if phone is not None:
            raw_phone = str(phone).strip()
            digits = "".join(ch for ch in raw_phone if ch.isdigit())
            if len(digits) != 10:
                raise ToolException("INVALID_PHONE: expected ###-###-#### (10 digits)")
            phone = f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"

        # --- Apply updates ---
        if account_name is not None:
            acct.name = (account_name or "").strip() or None
        if phone is not None:
            acct.phone = phone
        if emergency_contact is not None:
            acct.emergency_contact = (emergency_contact or "").strip() or None

        # User-level
        if name is not None and hasattr(user, "name"):
            user.name = name.strip() or None
        if email is not None and hasattr(user, "email"):
            user.email = email.strip() or None

        # Emails table (primary/secondary)
        if primary_email is not None or secondary_email is not None:
            _upsert_account_emails(db, acct.id, primary_email, secondary_email)

        # People (simple add)
        added_person = None
        if add_person_name is not None:
            nm = (add_person_name or "").strip()
            if nm:
                p = Person(
                    account_id=acct.id,
                    full_name=nm,
                    email=(add_person_email.strip() if add_person_email else None),
                )
                db.add(p)
                db.flush()
                added_person = {
                    "person_id": str(p.id),
                    "full_name": p.full_name,
                    "email": p.email,
                }
        db.commit()
        db.refresh(acct)
        db.refresh(user)
        emails = _get_account_emails(db, acct.id)
        primary = emails[0].email if emails else user.email
        secondary = emails[1].email if len(emails) > 1 else None
        display_name = acct.name or user.name

        result = {
            "id": str(acct.id),
            "user_id": str(user.id),
            "name": display_name,
            "primary_email": primary,
            "secondary_email": secondary,
            "primary_phone": acct.phone,
            "emergency_contact": acct.emergency_contact,
            "action": "updated_client",
        }
        if added_person:
            result["added_person"] = added_person

        return result


class ToolListPeopleIn(BaseModel):
    client_id: str = Field(..., description="ClientAccount.id (int or string)")


@tool("list_people", args_schema=ToolListPeopleIn, return_direct=False)
def list_people_tool(
    client_id: str, config: RunnableConfig = None
) -> List[Dict[str, Any]]:
    """List people (children/household members) on a client account.

    Args:
        client_id: ClientAccount.id (int or string form).
        config: Runnable configuration providing the owner id.

    Returns:
        List of dicts with 'person_id', 'full_name', and 'email'.

    Raises:
        ToolException: If the account id is invalid or not owned.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        try:
            acct_id = int(str(client_id).strip())
        except Exception:
            raise ToolException("Invalid client_id")
        acct = (
            db.query(ClientAccount)
            .filter(
                and_(
                    ClientAccount.owner_user_id == owner_id,
                    ClientAccount.id == acct_id,
                    ClientAccount.deleted_at.is_(None),
                )
            )
            .first()
        )
        if not acct:
            raise ToolException("Account not found for this owner")

        rows = (
            db.query(Person)
            .filter(Person.account_id == acct_id)
            .order_by(Person.id.asc())
            .all()
        )
        return [
            {"person_id": str(p.id), "full_name": p.full_name, "email": p.email}
            for p in rows
        ]


class ToolUpdatePersonIn(BaseModel):
    person_id: str = Field(..., description="Person.id (int or string)")
    full_name: Optional[str] = None
    email: Optional[str] = None


@tool("update_person", args_schema=ToolUpdatePersonIn, return_direct=False)
def update_person_tool(
    person_id: str,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """Update a person's name and/or email with ownership enforcement.

    Args:
        person_id: Person.id (int or string form).
        full_name: New full name (optional).
        email: New email address (optional).
        config: Runnable configuration providing the owner id.

    Returns:
        Dict with 'person_id', 'account_id', 'full_name', and 'email'.

    Raises:
        ToolException: For invalid ids, missing person, or authorization errors.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        try:
            pid = int(str(person_id).strip())
        except Exception:
            raise ToolException("Invalid person_id")

        p = db.query(Person).filter(Person.id == pid).first()
        if not p:
            raise ToolException("Person not found")

        acct = (
            db.query(ClientAccount)
            .filter(
                and_(
                    ClientAccount.id == p.account_id,
                    ClientAccount.owner_user_id == owner_id,
                )
            )
            .first()
        )
        if not acct:
            raise ToolException("Not authorized for this person/account")
        if full_name is not None:
            p.full_name = (full_name or "").strip() or p.full_name
        if email is not None:
            e = (email or "").strip()
            p.email = e or None

        db.commit()
        db.refresh(p)
        return {
            "person_id": str(p.id),
            "account_id": str(p.account_id),
            "full_name": p.full_name,
            "email": p.email,
        }


class ToolDeletePersonIn(BaseModel):
    person_id: str = Field(..., description="Person.id (int or string)")


@tool("delete_person", args_schema=ToolDeletePersonIn, return_direct=False)
def delete_person_tool(person_id: str, config: RunnableConfig = None) -> Dict[str, Any]:
    """Delete a person from an account after ownership check.

    Args:
        person_id: Person.id (int or string form).
        config: Runnable configuration providing the owner id.

    Returns:
        Dict with 'deleted' and 'person_id'.

    Raises:
        ToolException: For invalid ids, missing person, or authorization errors.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        try:
            pid = int(str(person_id).strip())
        except Exception:
            raise ToolException("Invalid person_id")

        p = db.query(Person).filter(Person.id == pid).first()
        if not p:
            raise ToolException("Person not found")

        acct = (
            db.query(ClientAccount)
            .filter(
                and_(
                    ClientAccount.id == p.account_id,
                    ClientAccount.owner_user_id == owner_id,
                )
            )
            .first()
        )
        if not acct:
            raise ToolException("Not authorized for this person/account")

        db.delete(p)
        db.commit()
        return {"deleted": True, "person_id": str(pid)}
