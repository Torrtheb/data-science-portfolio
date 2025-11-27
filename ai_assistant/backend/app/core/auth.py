from jose import jwt, JWTError
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import os
import logging

ALGORITHM = "HS256"
NEXTAUTH_SECRET = os.getenv("NEXTAUTH_SECRET")
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "0") == "1"

if not AUTH_DISABLED and not NEXTAUTH_SECRET:
    raise RuntimeError("NEXTAUTH_SECRET must be set when AUTH is enabled")

security = HTTPBearer(auto_error=not AUTH_DISABLED)


class TokenUser(BaseModel):
    """
    Minimal user identity extracted from a NextAuth JWT.

    Attributes:
        sub: The stable NextAuth user id (subject). Required for a valid identity.
        email: Optional email address associated with the account.
        role: Optional application role ("OWNER" | "STAFF" | "CLIENT").
        timezone: Optional IANA timezone string for user-local operations.
    """

    sub: str
    email: str | None = None
    role: str | None = None
    timezone: str | None = None


def _decode(token: str) -> TokenUser:
    """
    Decode and verify a NextAuth JWT to a 'TokenUser'.

    This function verifies the token using HS256 and the configured 'NEXTAUTH_SECRET'.
    On success, it maps known claims into a 'TokenUser' model. On any validation
    failure it raises an HTTP 401 error suitable for API responses.

    Args:
        token: Raw bearer token string (without the "Bearer " prefix).

    Returns:
        A populated 'TokenUser' instance.

    Raises:
        RuntimeError: If NEXTAUTH_SECRET is not configured while auth is enabled.
        HTTPException: 401 Unauthorized when the token is invalid or cannot be decoded.
    """
    if not NEXTAUTH_SECRET:
        raise RuntimeError("NEXTAUTH_SECRET not set")
    try:
        payload = jwt.decode(token, NEXTAUTH_SECRET, algorithms=[ALGORITHM])
        return TokenUser(
            sub=payload.get("sub"),
            email=payload.get("email"),
            role=payload.get("role"),
            timezone=payload.get("timezone"),
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> TokenUser:
    """
    Authenticate the current request and return a 'TokenUser'.

    Behavior:
    - **Dev bypass (AUTH_DISABLED=1)**:
        Requires 'DEV_FAKE_OWNER_ID'. Looks up the corresponding 'auth.User'
        record and returns a synthetic OWNER identity. This prevents silent
        elevation when developing without real tokens.
    - **Normal mode**:
        Requires a Bearer token. Verifies and decodes the token to 'TokenUser'.
        Ensures 'sub' (subject/user id) is present.

    Args:
        creds: Parsed bearer credentials from the 'Authorization' header. When
               'AUTH_DISABLED=1', this may be None due to 'auto_error=False'.

    Returns:
        The authenticated 'TokenUser' for this request.

    Raises:
        HTTPException:
            - 503 if dev bypass is enabled but 'DEV_FAKE_OWNER_ID' is missing or invalid.
            - 401 if the Authorization header is missing or malformed (normal mode).
            - 401 if the token decodes but is missing 'sub'.
    """
    try:
        if not AUTH_DISABLED:
            sec = NEXTAUTH_SECRET or ""
            if not sec or sec == "your_32+_base64_secret_key_here":
                logging.getLogger(__name__).warning(
                    "NEXTAUTH_SECRET is unset or default while AUTH is enabled"
                )
    except Exception:
        pass
    if AUTH_DISABLED:
        dev_owner_id = os.getenv("DEV_FAKE_OWNER_ID")
        if not dev_owner_id:
            raise HTTPException(
                status_code=503,
                detail="Authentication is disabled but DEV_FAKE_OWNER_ID is not set",
            )
        from app.db import SessionLocal
        from app.models import User

        with SessionLocal() as db:
            owner = db.query(User).filter(User.id == dev_owner_id).first()
            if not owner:
                raise HTTPException(
                    status_code=503,
                    detail="DEV_FAKE_OWNER_ID does not match an owner user",
                )
            return TokenUser(
                sub=owner.id, email=owner.email, role="OWNER", timezone=owner.timezone
            )
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    user = _decode(creds.credentials)
    if not user.sub:
        raise HTTPException(status_code=401, detail="Invalid token (no sub)")
    return user


def require_owner(user: TokenUser = Depends(get_current_user)) -> TokenUser:
    """
    Authorization guard that permits only OWNER role.

    In dev bypass mode ('AUTH_DISABLED=1'), this check is skipped and the
    current user (synthetic OWNER) is returned.

    Args:
        user: The authenticated user provided by 'get_current_user'.

    Returns:
        The same 'TokenUser' if authorized.

    Raises:
        HTTPException: 403 Forbidden if the user is authenticated but not an OWNER.
    """
    if AUTH_DISABLED:
        return user
    if user.role != "OWNER":
        raise HTTPException(status_code=403, detail="Forbidden (owner only)")
    return user


def require_client(user: TokenUser = Depends(get_current_user)) -> TokenUser:
    """
    Authorization guard that permits only CLIENT role.

    In dev bypass mode ('AUTH_DISABLED=1'), this check is skipped and the
    current user is returned.

    Args:
        user: The authenticated user provided by 'get_current_user'.

    Returns:
        The same 'TokenUser' if authorized.

    Raises:
        HTTPException: 403 Forbidden if the user is authenticated but not a CLIENT.
    """
    if AUTH_DISABLED:
        return user
    if user.role != "CLIENT":
        raise HTTPException(status_code=403, detail="Forbidden (client only)")
    return user
