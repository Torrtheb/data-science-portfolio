from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.core.auth import require_owner, TokenUser
from .agent_chat import _cfg_graph  # reuse the same helper
from agent.tools_clients import (
    find_client_tool as find_client_tool,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clients", tags=["clients"])
"""Router for client lookup and email search endpoints."""


class ClientHit(BaseModel):
    """Single search hit for a client email.

    Fields:
        email: Email address.
        name: Display name associated with the email (account/person).
        source: Source of the hit, e.g. "account" or "person".
    """

    email: str
    name: Optional[str] = None
    source: Optional[str] = None


class SearchOut(BaseModel):
    """Response shape for client email search."""

    items: List[ClientHit]


@router.get("/search", response_model=SearchOut)
def search_clients(
    q: str = Query(..., min_length=1),
    user: TokenUser = Depends(require_owner),
) -> SearchOut:
    """Search client emails by delegating to the 'find_client' tool.

    Args:
        q: Free‑text selector forwarded to the tool.
        user: Authenticated owner context.

    Returns:
        'SearchOut' with a de‑duplicated list of 'ClientHit' results across
        account and person emails.
    """
    cfg = _cfg_graph(user.sub, "__client_picker__", user.sub)
    items: List[ClientHit] = []
    try:
        res = find_client_tool.invoke({"selector": q}, config=cfg)
        if isinstance(res, str):
            import json

            try:
                res = json.loads(res)
            except Exception:
                res = {}

        acc_email = (res or {}).get("primary_email")
        acc_name = (res or {}).get("name")
        if acc_email:
            items.append(
                ClientHit(email=acc_email, name=acc_name or q, source="account")
            )

        for p in (res or {}).get("people", []) or []:
            em = p.get("email")
            nm = p.get("full_name") or p.get("name")
            if em:
                items.append(ClientHit(email=em, name=nm, source="person"))

        seen = set()
        uniq: List[ClientHit] = []
        for it in items:
            if it.email not in seen:
                seen.add(it.email)
                uniq.append(it)
        items = uniq

    except Exception as e:
        log.warning("clients.search failed: %s", e)

    return {"items": [it.model_dump() for it in items]}
