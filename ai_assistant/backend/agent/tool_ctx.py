from __future__ import annotations
import contextvars
from typing import Final, Optional

owner_id_var: Final[contextvars.ContextVar[Optional[str]]] = contextvars.ContextVar(
    "owner_id", default=None
)
tz_var: Final[contextvars.ContextVar[Optional[str]]] = contextvars.ContextVar(
    "tz", default=None
)

__all__ = ["owner_id_var", "tz_var"]
