from __future__ import annotations
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Generator

from sqlalchemy import (
    create_engine,
    event,
    select,
    Text,
    DateTime,
    ForeignKey,
    String,
    Integer,
    Index,
)
from sqlalchemy.orm import (
    sessionmaker,
    Mapped,
    mapped_column,
    relationship,
    Session as SASession,
)
from sqlalchemy.exc import IntegrityError

from .db_base import Base

try:
    from ..analytics.models_analytics import Base as AnalyticsBase
except Exception:
    AnalyticsBase = None

# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

if os.getenv("DATABASE_URL"):
    DATABASE_URL = os.getenv("DATABASE_URL")
else:
    if os.getenv("K_SERVICE"): 
        DEFAULT_SQLITE_PATH = "/tmp/finassist.db"
    else:
        DEFAULT_SQLITE_PATH = "./data/finassist.db"
    DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH}"


def _ensure_sqlite_dir(db_url: str) -> None:
    """
    Ensure the directory for a SQLite file-based database exists.

    For SQLite URLs, derives the file path and creates its parent directory
    if missing. No-ops for non-SQLite URLs or memory/special SQLite URIs.

    Args:
        db_url: SQLAlchemy database URL.
    """
    if not db_url.startswith("sqlite"):
        return
    if db_url.startswith("sqlite:////"):
        file_path = db_url.replace("sqlite:////", "/", 1)
    elif db_url.startswith("sqlite:///"):
        rel = db_url.replace("sqlite:///", "", 1)
        file_path = os.path.abspath(rel)
    else:
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

_ensure_sqlite_dir(DATABASE_URL)


engine = create_engine(
    DATABASE_URL,
    connect_args=(
        {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    ),
    pool_pre_ping=True,
    future=True,
)

if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_connection, connection_record):
        """
        Apply SQLite-specific pragmas on each new connection.

        - WAL journal for better concurrency
        - Enforce foreign keys
        - NORMAL sync for performance tradeoff
        """
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def utcnow() -> datetime:
    """Timezone-aware UTC 'now' for created_at/updated_at."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------


class SessionModel(Base):
    """
    Chat session metadata.

    Represents a conversation container. Messages reference a session via
    ''session_id''. Deleting a session cascades to its messages.
    """
    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    messages: Mapped[List["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="MessageModel.id",
        passive_deletes=True,
    )
    __table_args__ = (Index("ix_chat_sessions_updated_at", "updated_at"),)
    def __repr__(self) -> str:
        return f"<SessionModel id={self.id!r} title={self.title!r}>"


class MessageModel(Base):
    """
    A single chat message within a session.

    Stores role, content, creation time, and optional serialized tool-call data.
    """
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    tool_calls_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    session: Mapped[SessionModel] = relationship(
        "SessionModel", back_populates="messages"
    )

    __table_args__ = (Index("ix_chat_messages_session_id_id", "session_id", "id"),)
    def __repr__(self) -> str:
        return f"<MessageModel id={self.id} session_id={self.session_id!r} role={self.role!r}>"


# ---------------------------------------------------------------------------
# Schema creation & light dev migration
# ---------------------------------------------------------------------------


def init_db() -> None:
    """
    Create database schema (idempotent) and run a light SQLite-only migration.

    - Creates tables for primary app models using this module's ''Base''.
    - If analytics models are importable, creates their tables on the same engine.
    - For SQLite only: ensures the ''chat_messages.tool_calls_json'' column exists.
    """

    Base.metadata.create_all(bind=engine)
    if AnalyticsBase is not None and AnalyticsBase is not Base:
        AnalyticsBase.metadata.create_all(bind=engine)

    if str(engine.url).startswith("sqlite"):
        with engine.begin() as conn:
            cols = [
                row[1]
                for row in conn.exec_driver_sql("PRAGMA table_info(chat_messages)")
            ]
            if "tool_calls_json" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN tool_calls_json TEXT;"
                )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_db() -> Generator[SASession, None, None]:
    """
    FastAPI dependency that yields a database session and guarantees cleanup.

    Yields:
        A SQLAlchemy session bound to the configured engine.

    Ensures the session is closed after request handling completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------


def get_or_create_session(
    db: SASession,
    session_id: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Return an existing session id or create one if absent (race-safe).

    If ''session_id'' is provided and exists, it is returned unchanged.
    Otherwise, the function attempts to create the session. On a concurrent
    race where another request created it first, the function falls back to
    fetching the existing row and returning its id.

    Args:
        db: Active SQLAlchemy session.
        session_id: Optional explicit id; if None, a new UUID is generated.
        title: Optional human-readable title for the session.

    Returns:
        The canonical session id string.

    Raises:
        IntegrityError: If creation fails and the row cannot be retrieved afterwards.
    """
    if session_id:
        row = db.get(SessionModel, session_id)
        if row:
            return row.id
        new_id = session_id
    else:
        new_id = str(uuid.uuid4())

    now = utcnow()
    sess = SessionModel(id=new_id, title=title, created_at=now, updated_at=now)
    db.add(sess)
    try:
        db.commit()
        return new_id
    except IntegrityError:
        # Another request created it first; that's fine—use the existing one.
        db.rollback()
        existing = db.get(SessionModel, new_id)
        if existing:
            return existing.id
        # If we truly can't find it, surface the error.
        raise


def touch_session(db: SASession, session_id: str) -> None:
    """
    Update a session's ''updated_at'' timestamp.

    No-op if the session does not exist.

    Args:
        db: Active SQLAlchemy session.
        session_id: Identifier of the session to touch.
    """
    row = db.get(SessionModel, session_id)
    if row:
        row.updated_at = utcnow()
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise


def append_message(
    db: SASession,
    session_id: str,
    role: str,
    content: str,
    tool_calls_json: Optional[str],
) -> int:
    """
    Append a message to a session and update session ''updated_at'' atomically.

    Inserts the message, bumps the parent session's ''updated_at'', commits,
    and returns the new message id.

    Args:
        db: Active SQLAlchemy session.
        session_id: Owning session id (must exist).
        role: Author role (e.g. ''"user"'', ''"assistant"'', ''"system"'').
        content: Message text.
        tool_calls_json: Optional serialized tool-call metadata.

    Returns:
        The newly created message's integer primary key.

    Raises:
        Exception: Any commit error is propagated after rolling back.
    """
    now = utcnow()

    msg = MessageModel(
        session_id=session_id,
        role=role,
        content=content,
        created_at=now,
        tool_calls_json=tool_calls_json,
    )
    db.add(msg)

    sess = db.get(SessionModel, session_id)
    if sess:
        sess.updated_at = now

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(msg)
    return msg.id


def list_messages(
    db: SASession,
    session_id: str,
    limit: int = 200,
) -> List[MessageModel]:
    """
    Return messages for a session in ascending ''id'' order.

    Args:
        db: Active SQLAlchemy session.
        session_id: Session whose messages to list.
        limit: Maximum number of messages to return (default 200).

    Returns:
        A list of ''MessageModel'' rows ordered by ''id'' ascending.
    """
    q = (
        select(MessageModel)
        .where(MessageModel.session_id == session_id)
        .order_by(MessageModel.id.asc())
        .limit(limit)
    )
    return list(db.execute(q).scalars())


def delete_session(db: SASession, session_id: str) -> None:
    """
    Delete a session and (via cascade) its messages.
    No-op if the session does not exist.

    Args:
        db: Active SQLAlchemy session.
        session_id: Identifier of the session to delete.

    Raises:
        Exception: Any commit error is propagated after rolling back.
    """
    row = db.get(SessionModel, session_id)
    if row:
        db.delete(row)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
