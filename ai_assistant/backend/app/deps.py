import os
from fastapi import Depends, HTTPException, Request, status
from typing import Optional, Any, Iterator, TypedDict
from sqlalchemy import text
from .db import SessionLocal

OWNER_EMAIL = os.getenv("OWNER_EMAIL")
DEFAULT_TZ = "America/Toronto"


class UserContext(TypedDict):
    """
    Minimal user context returned by 'get_current_user'.

    Keys:
        sub: Stable user id (string).
        email: Email address, if known.
        role: "OWNER" | "STAFF" | "CLIENT" (uppercase).
        timezone: IANA timezone string (fallback to DEFAULT_TZ if missing).
    """

    sub: str
    email: Optional[str]
    role: str
    timezone: str


def get_db():
    """
    FastAPI dependency that yields a SQLAlchemy Session and ensures closure.

    Usage:
        def endpoint(db: Session = Depends(get_db)):
            ...

    Yields:
        A live 'Session' bound to the application's engine.

    Always closes the session after the endpoint returns, even on error.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _lookup_user_by_email(db, email: str):
    """
    Fetch a user row from Prisma's auth schema by email.

    Args:
        db: Active SQLAlchemy Session.
        email: Email address to search (case-sensitive as stored).

    Returns:
        A mapping with keys (id, email, timezone, role) or None if not found.
    """
    return (
        db.execute(
            text(
                'select id, email, timezone, role from auth."User" where email = :email'
            ),
            {"email": email},
        )
        .mappings()
        .first()
    )


def _lookup_user_by_id(db, user_id: str):
    """
    Fetch a user row from Prisma's auth schema by id.

    Args:
        db: Active SQLAlchemy Session.
        user_id: User id (UUID string) to search.

    Returns:
        A mapping with keys (id, email, timezone, role) or None if not found.
    """
    return (
        db.execute(
            text('select id, email, timezone, role from auth."User" where id = :id'),
            {"id": user_id},
        )
        .mappings()
        .first()
    )


def get_current_user(req: Request, db=Depends(get_db)):
    """
    Resolve the current user identity.

    Modes:
        1) **Forwarded headers (preferred when proxied)**:
            - Reads 'x-user-id' (required for this path) and optional 'x-user-role'.
            - Looks up the user by id in 'auth."User"'.
            - Role precedence: header role (if provided) else DB role (default "CLIENT").
        2) **Single-tenant fallback (dev/bootstrap)**:
            - Uses 'OWNER_EMAIL' to resolve the owner and returns them as role=OWNER.

    Args:
        req: The incoming FastAPI Request (for reading forwarded headers).
        db: SQLAlchemy Session (injected via Depends).

    Returns:
        UserContext: dict with 'sub, email, role, timezone'.

    Raises:
        HTTPException:
            - 401 if a forwarded user id does not resolve to a user.
            - 500 if OWNER_EMAIL is not configured (fallback path).
            - 401 if OWNER_EMAIL does not resolve to a user (fallback path).

    Security:
        This function trusts 'x-user-*' headers. Ensure they are set only by a
        trusted frontend or reverse proxy. For cryptographically verified auth,
        use the JWT-based dependencies in 'core/auth.py'.
    """
    uid = req.headers.get("x-user-id")
    role_hdr = (req.headers.get("x-user-role") or "").upper()

    if uid:
        row = _lookup_user_by_id(db, uid)
        if not row:
            raise HTTPException(status_code=401, detail="User not found")

        role = (role_hdr or (row["role"] or "CLIENT")).upper()
        return {
            "sub": row["id"],
            "email": row["email"],
            "role": role,
            "timezone": row["timezone"] or "America/Toronto",
        }

    if not OWNER_EMAIL:
        raise HTTPException(500, "OWNER_EMAIL not configured")
    row = _lookup_user_by_email(db, OWNER_EMAIL)
    if not row:
        raise HTTPException(401, "Owner user not found")
    return {
        "sub": row["id"],
        "email": row["email"],
        "role": "OWNER",
        "timezone": row["timezone"] or "America/Toronto",
    }


def require_owner(user=Depends(get_current_user)):
    """
    Authorization guard that permits only OWNER role.

    Args:
        user: The current user from 'get_current_user'.

    Returns:
        The same 'UserContext' if authorized.

    Raises:
        HTTPException: 403 if the user is not an OWNER.
    """

    if user["role"] != "OWNER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Owner role required"
        )
    return user


def require_client(user=Depends(get_current_user)):
    """
    Authorization guard that permits CLIENTs and OWNERs.

    This is useful for endpoints a client can access directly, while still
    allowing an owner to perform actions on behalf of clients.

    Args:
        user: The current user from 'get_current_user'.

    Returns:
        The same 'UserContext' if authorized.

    Raises:
        HTTPException: 403 if the role is neither CLIENT nor OWNER.
    """
    if user["role"] not in ("CLIENT", "OWNER"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Client role required"
        )
    return user


def get_single_owner_id(db):
    """
    Resolve the single-tenant owner's user id using 'OWNER_EMAIL'.

    Intended for dev/bootstrap flows where there is exactly one business owner.

    Args:
        db: Active SQLAlchemy Session.

    Returns:
        The owner's user id (UUID string).

    Raises:
        HTTPException:
            - 500 if OWNER_EMAIL is not configured.
            - 401 if no user exists with OWNER_EMAIL.
    """
    if not OWNER_EMAIL:
        raise HTTPException(500, "OWNER_EMAIL not configured")
    row = _lookup_user_by_email(db, OWNER_EMAIL)
    if not row:
        raise HTTPException(401, "Owner user not found")
    return row["id"]
