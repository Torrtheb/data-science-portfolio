from __future__ import annotations
from typing import List, Optional, Literal
from fastapi import APIRouter, Depends, Body, Query
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session as SASession
from ..core.db import get_db, get_or_create_session, list_messages
from ..core.auth import session_tokens

router = APIRouter(tags=["sessions"])

# --------------------- Pydantic IO models ---------------------


class SessionListItem(BaseModel):
    """Optional summary for a session; currently unused (stub)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: Optional[str] = None


class SessionListResponse(BaseModel):
    """Response shape for GET /chat/sessions (kept for FE compatibility)."""

    model_config = ConfigDict(extra="ignore")

    sessions: List[SessionListItem] = Field(default_factory=list)


class SessionCreateBody(BaseModel):
    """Request body for POST /chat/sessions."""

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(
        default=None, description="Client-provided session id (optional)"
    )
    title: Optional[str] = Field(default="Chat", max_length=200)


class SessionCreateResponse(BaseModel):
    """Response for POST /chat/sessions."""

    model_config = ConfigDict(extra="ignore")

    sessionId: str
    token: Optional[str] = Field(
        default=None,
        description="Signed bearer token for this session (if token auth enabled).",
    )


class SessionMessage(BaseModel):
    """Wire format for a message in GET /chat/sessions/{session_id}."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    created_at: Optional[str]


class SessionGetResponse(BaseModel):
    """Response for GET /chat/sessions/{session_id}."""

    model_config = ConfigDict(extra="ignore")

    sessionId: str
    messages: List[SessionMessage] = Field(default_factory=list)


# ------------------------------- Routes -------------------------------------


@router.get("/chat/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    """
    List chat sessions (stub).
    Returns an empty list to preserve FE contract until a real listing is added.
    """
    return SessionListResponse(sessions=[])


@router.post("/chat/sessions", response_model=SessionCreateResponse, status_code=201)
def create_session(
    payload: SessionCreateBody = Body(...),
    db: SASession = Depends(get_db),
) -> SessionCreateResponse:
    """
    Create (or ensure) a session and return its id.
    - If 'id' is provided and doesn't exist, it's created.
    - If 'id' is omitted, a new one is generated.
    """
    session_id = get_or_create_session(db, payload.id, payload.title or "Chat")
    token = session_tokens.issue(session_id) if session_tokens.enabled else None
    return SessionCreateResponse(sessionId=session_id, token=token)


@router.get("/chat/sessions/{session_id}", response_model=SessionGetResponse)
def get_session(
    session_id: str,
    db: SASession = Depends(get_db),
    limit: int = Query(
        200, ge=1, le=2000, description="Max number of messages to return"
    ),
) -> SessionGetResponse:
    """
    Return messages for the given session id.
    Ensures the session exists; then returns up to 'limit' messages.
    """
    sid = get_or_create_session(db, session_id)
    rows = list_messages(db, sid, limit=limit)
    messages = [
        SessionMessage(
            role=r.role,
            content=r.content,
            created_at=(
                r.created_at.isoformat() if getattr(r, "created_at", None) else None
            ),
        )
        for r in rows
    ]
    return SessionGetResponse(sessionId=sid, messages=messages)
