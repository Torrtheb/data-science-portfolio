from __future__ import annotations
from typing import Optional, List, Literal, Any, Dict
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationInfo

AllowedRole = Literal["user", "assistant", "system", "tool"]


def utcnow() -> datetime:
    """Return timezone-aware UTC now (helper used in examples/tests)."""
    return datetime.now(timezone.utc)


class TokenUsage(BaseModel):
    """
    Metering information for an LLM call.

    - 'model' may be omitted if usage is aggregated across calls.
    - 'cost_usd' is optional; you can compute later from pricing tables.
    """

    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    cost_usd: Optional[float] = Field(default=None, ge=0)

    @property
    def total_tokens(self) -> Optional[int]:
        """Convenience property: input + output if both are present, else None."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class ToolCallRecord(BaseModel):
    """
    A single tool invocation record.

    'extra="allow"' lets tools attach heterogeneous fields without breaking validation
    (e.g., latency_ms, retries, vendor payloads).
    """

    model_config = ConfigDict(extra="allow")

    tool: Optional[str] = None
    name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    id: Optional[str] = None


class MessageIn(BaseModel):
    """
    Inbound message from the client.

    - Only 'user' messages must contain non-empty 'content'.
    - Assistant/system/tool messages may be blank (e.g., when carrying tool_calls only).
    """

    model_config = ConfigDict(extra="forbid")

    role: AllowedRole
    content: str = Field("", max_length=32000)

    tool_calls: Optional[List[ToolCallRecord]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @field_validator("role")
    @classmethod
    def _norm_role(cls, v: str) -> AllowedRole:
        v = (v or "").strip().lower()
        if v not in ("user", "assistant", "system", "tool"):
            raise ValueError("role must be one of: user, assistant, system, tool")
        return v

    @field_validator("content")
    @classmethod
    def _require_user_content(cls, v: str, info: ValidationInfo) -> str:
        role = (info.data or {}).get("role")
        v = v or ""
        if role == "user" and not v.strip():
            raise ValueError("user content cannot be empty")
        return v


class MessageOut(BaseModel):
    """
    Outbound message returned by the API.
    """

    model_config = ConfigDict(extra="ignore")

    id: int
    role: AllowedRole
    content: str
    created_at: datetime
    tool_calls: Optional[List[ToolCallRecord]] = None


class SessionCreate(BaseModel):
    """
    Request body to create a session.

    - 'id' is optional; if omitted, the server can generate one.
    - 'title' is optional and limited to 200 chars.
    """

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    title: Optional[str] = Field(default=None, max_length=200)


class SessionOut(BaseModel):
    """
    Session envelope returned by the API, with optional embedded messages.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: List[MessageOut] = Field(default_factory=list)


class ChatInput(BaseModel):
    """
    Top-level request body for the chat endpoint.

    - Supports both 'session_id' and 'sessionId' for backward compatibility.
    - 'messages' defaults to an empty list.
    """

    model_config = ConfigDict(extra="forbid")

    messages: List[MessageIn] = Field(default_factory=list)
    session_id: Optional[str] = None
    sessionId: Optional[str] = None
    title: Optional[str] = None
    debug: Optional[bool] = False

    @property
    def session_id_normalized(self) -> Optional[str]:
        """
        Unified accessor: prefer snake_case, else camelCase.
        Use this in your route handler to avoid branching.
        """
        return self.session_id or self.sessionId


class ToolSource(BaseModel):
    """
    Metadata describing where a ToolResult came from (for UI/source panels).
    """

    model_config = ConfigDict(extra="ignore")

    type: Literal["tool"] = "tool"
    name: str
    title: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """
    Standard wrapper for tool execution results.

    - 'ok=False' can be used to surface partial failures in a consistent shape.
    - 'data' contains tool-specific payload.
    - 'source' describes the origin and can be surfaced in the UI.
    """

    model_config = ConfigDict(extra="ignore")

    tool: str
    ok: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    source: ToolSource
