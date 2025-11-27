from __future__ import annotations

import re
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .settings import settings


class SessionTokenManager:
    """
    Issue and verify signed session tokens.

    Tokens are opaque strings (URL-safe) that encode the session id ("sid") and
    expire after a configurable TTL. If no secret is configured, the manager is
    disabled and issuing/verifying will raise.
    """

    def __init__(self, secret: Optional[str], ttl_seconds: int = 7 * 24 * 3600):
        self.secret = secret.strip() if secret else None
        self.ttl_seconds = ttl_seconds
        self._serializer = (
            URLSafeTimedSerializer(self.secret, salt="session-token")
            if self.secret
            else None
        )

    @property
    def enabled(self) -> bool:
        """Whether session tokens are enabled (secret provided)."""
        return self._serializer is not None

    def issue(self, session_id: str) -> str:
        """
        Create a signed token carrying the given session id.

        Raises:
            RuntimeError: if tokens are disabled (no secret).
        """
        if not self._serializer:
            raise RuntimeError("Session tokens are disabled (no secret configured).")
        return self._serializer.dumps({"sid": session_id})

    def verify(self, token: str) -> str:
        """
        Verify a token and return the embedded session id.

        Raises:
            SignatureExpired: if token expired.
            BadSignature: if token is invalid.
            RuntimeError: if tokens are disabled.
        """
        if not self._serializer:
            raise RuntimeError("Session tokens are disabled (no secret configured).")
        data = self._serializer.loads(token, max_age=self.ttl_seconds)
        sid = (data or {}).get("sid")
        if not sid:
            raise BadSignature("Token missing session id.")
        return sid


def _parse_bearer(auth_header: Optional[str]) -> Optional[str]:
    """
    Extract the bearer token from an Authorization header.

    Args:
        auth_header: Raw Authorization header value.

    Returns:
        Token string if present, else None.
    """
    if not auth_header:
        return None
    m = re.match(r"^\s*Bearer\s+(.+)$", auth_header, re.IGNORECASE)
    return m.group(1).strip() if m else None


session_tokens = SessionTokenManager(
    settings.session_token_secret, settings.session_token_ttl_seconds
)

