from __future__ import annotations

from typing import List, Optional
from datetime import datetime

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    Boolean,
    ForeignKey,
    Index,
    CheckConstraint,
    Numeric,
    JSON,
    func,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..core.db_base import Base


# --- Session (analytics view of chat sessions) --------------------------------
class Session(Base):
    """
    Chat session envelope. Each session groups many ChatTurns and Events.
    """

    __tablename__ = "sessions"
    __table_args__ = ({"sqlite_autoincrement": True},)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )
    turns: Mapped[List["ChatTurn"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    events: Mapped[List["Event"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    tool_invocations: Mapped[List["ToolInvocation"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# --- ChatTurn (a single message from user/assistant/system) -------------------
class ChatTurn(Base):
    """
    A single conversational turn with token, cost, latency, and tool trace metadata.
    Used to power analytics like token/cost over time, error rates, and tool usage.
    """

    __tablename__ = "chat_turns"
    __table_args__ = (
        Index("ix_chat_turns_session_time", "session_id", "created_at"),
        Index("ix_chat_turns_model_time", "model", "created_at"),
        CheckConstraint("tokens_in  >= 0", name="ck_turn_tokens_in_nonneg"),
        CheckConstraint("tokens_out >= 0", name="ck_turn_tokens_out_nonneg"),
        CheckConstraint("latency_ms >= 0", name="ck_turn_latency_nonneg"),
        CheckConstraint("cost_usd  >= 0", name="ck_turn_cost_nonneg"),
        CheckConstraint(
            "role IN ('user','assistant','system')", name="ck_turn_role_valid"
        ),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    session_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)

    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    had_rag: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    tools_used: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )

    session: Mapped["Session"] = relationship(back_populates="turns")
    tool_invocations: Mapped[List["ToolInvocation"]] = relationship(
        back_populates="turn",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# --- ToolInvocation (one tool call made during a turn) ------------------------
class ToolInvocation(Base):
    """
    Fine-grained tool telemetry: who called what, with which args, how long it took,
    and whether it succeeded. Powers the "Top Tools", latency, and failure charts.
    """

    __tablename__ = "tool_invocations"
    __table_args__ = (
        Index("ix_toolinv_session_time", "session_id", "created_at"),
        Index("ix_toolinv_tool_time", "tool_name", "created_at"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    session_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    turn_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("chat_turns.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    args: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )

    session: Mapped["Session"] = relationship(back_populates="tool_invocations")
    turn: Mapped[Optional["ChatTurn"]] = relationship(back_populates="tool_invocations")


# --- Event (generic analytics events) ----------------------------------------
class Event(Base):
    """
    Generic analytics event (e.g., UI click, calc submit). Keep flexible in props.
    """

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_session_time", "session_id", "created_at"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    session_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    props: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )
    session: Mapped[Optional["Session"]] = relationship(back_populates="events")
