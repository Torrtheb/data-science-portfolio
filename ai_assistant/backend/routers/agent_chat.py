from __future__ import annotations
from fastapi import APIRouter, Depends, Request, Query, HTTPException
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from app.core.auth import require_owner, TokenUser
from sqlalchemy.orm import Session
import sqlalchemy as sa
from app.db import get_db
from langchain_core.messages import SystemMessage
from app.models import User
from datetime import datetime
import os
from zoneinfo import ZoneInfo
import json
import re

from agent.graph import build_graph
from agent.memory import (
    cp_put,
    _norm_config,
)
from agent.memory_hardening import ensure_checkpoint_sane, prune_checkpoint_if_needed
import logging

from uuid import UUID
from app.models import OutboxEmail, OutboxEmailStatus
from .outbox import SendBody, send_outbox_email
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Tuple

log = logging.getLogger(__name__)

CHECKPOINT_NS = "v3"
_RATE_LIMIT_WINDOW_SECONDS = 60
FUN_AGENT_SPICE = float(os.getenv("FUN_AGENT_SPICE", "0") or "0")
router = APIRouter(prefix="/api/agent", tags=["agent"])
_graph = build_graph()
_METRICS: dict[str, int] = {}


DEV_TOKEN = os.getenv("DEV_DEBUG_TOKEN")
DEBUG_ENDPOINTS = os.getenv("DEBUG_ENDPOINTS", "0") == "1"
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.I),
    re.compile(r"forget\s+(?:what\s+)?you\s+were\s+told", re.I),
    re.compile(r"(?:override|replace|change)\s+the\s+system\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(?:root|admin|system)", re.I),
    re.compile(r"as\s+system[:,]", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bsudo\b", re.I),
    re.compile(r"'''?\s*system", re.I),
    re.compile(r"^\s*<\s*system\b", re.I),
    re.compile(r"do\s+not\s+use\s+tools|answer\s+from\s+memory\s+only", re.I),
]


def _ensure_debug_enabled() -> None:
    """
    Guard debug routes behind runtime flags and a configured dev token.

    Behavior:
        - If 'DEBUG_ENDPOINTS' is falsy, raise 404 to avoid revealing the existence
          of debug endpoints.
        - If 'DEV_TOKEN' is unset or still the placeholder value, raise 503 so
          operators know the token must be configured before enabling debug tools.

    Raises:
        HTTPException: 404 when debug endpoints are disabled.
        HTTPException: 503 when a non-placeholder dev token is not configured.
    """
    if not DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")
    if not DEV_TOKEN or DEV_TOKEN == "changeme-debug-token":
        raise HTTPException(
            status_code=503,
            detail="Debug endpoints disabled until DEV_DEBUG_TOKEN is configured",
        )


def _detect_prompt_injection(text: str) -> str | None:
    """
    Heuristically detect prompt-injection markers in free text.

    Scans the input against '_PROMPT_INJECTION_PATTERNS', a list of compiled
    regular expressions. If a pattern matches, returns the exact matched
    substring (useful for logging or UI highlighting); otherwise returns None.

    Args:
        text: Arbitrary input to scan (may be empty/None-like).

    Returns:
        The matched suspicious substring if any pattern matches; otherwise None.

    Notes:
        - This is a lightweight heuristic and should complement (not replace)
          stronger sandboxing/validation where applicable.
    """
    if not text:
        return None
    for pat in _PROMPT_INJECTION_PATTERNS:
        match = pat.search(text)
        if match:
            return match.group(0)
    return None


def _moderate_or_reject(text: str) -> None:
    """
    Call OpenAI moderation and raise HTTP 400 if flagged.

    Falls back silently (allow) on client errors so the app remains usable even
    if moderation is unavailable.
    """
    if not text:
        return
    try:
        from openai import OpenAI

        client = OpenAI()
        resp = client.moderations.create(model="omni-moderation-latest", input=text)
        result = resp.results[0] if resp and resp.results else None
        if result and getattr(result, "flagged", False):
            raise HTTPException(
                status_code=400,
                detail="Message blocked by safety filters. Please rephrase.",
            )
    except HTTPException:
        raise
    except Exception:
        return


@contextmanager
def _db_commit(db: Session) -> Iterator[None]:
    """
    Transaction helper that commits on success and rolls back on error.

    Usage:
        with _db_commit(db):
            db.add(obj)

    Behavior:
        - Yields control to the caller.
        - On normal exit, issues 'db.commit()'.
        - On exception, issues 'db.rollback()' and re-raises the original error.

    Args:
        db: SQLAlchemy session bound to the current request/context.

    Raises:
        Whatever exception occurred inside the context (after rollback).
    """
    try:
        yield
        db.commit()
    except Exception:
        db.rollback()
        raise


def _bump(metric: str, inc: int = 1) -> None:
    """Increment a named in‑memory metric counter.

    Args:
        metric: Counter key.
        inc: Increment amount (default 1).
    """
    try:
        _METRICS[metric] = int(_METRICS.get(metric, 0)) + int(inc)
    except Exception:
        pass


def _log_event(name: str, **fields) -> None:
    """Emit a one‑line structured log for easy grep/aggregation.

    Args:
        name: Event name.
        **fields: Arbitrary key=value pairs to include.
    """
    try:
        parts = [f"{k}={fields[k]}" for k in sorted(fields.keys())]
        log.info("event=%s %s", name, " ".join(parts))
    except Exception:
        try:
            log.info("event=%s", name)
        except Exception:
            pass


def _cfg_all(user_id: str, session: str, owner_id: str, tz: str | None = None) -> dict:
    """Build runtime + checkpoint config for the agent graph.

    Uses a minimal checkpoint key ('thread_id' + namespace) while passing
    runtime context for tools (user/owner/timezone) via config.

    Args:
        user_id: Auth user id associated with the owner.
        session: Conversation/thread id.
        owner_id: Owner id (multi‑tenant scope).
        tz: Optional IANA timezone name from the client.

    Returns:
        Normalized LangChain config mapping.
    """
    cfg = {
        "thread_id": f"owner-{user_id}:{session}",
        "checkpoint_ns": CHECKPOINT_NS,
        "user_id": user_id,
        "owner_id": owner_id,
    }
    if tz:
        cfg["tz"] = tz
    return _norm_config({"configurable": cfg})


def _coerce_text_block(content: object) -> str:
    """Normalize a heterogeneous message content block into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        t = content.get("text") or content.get("data") or ""
        return t if isinstance(t, str) else str(t or "")
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("text") or c.get("data") or ""
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(c, str):
                parts.append(c)
        return "".join(parts)
    return str(content or "")


def _msg_type(m: object) -> str:
    """Extract a best-effort message type from supported shapes."""
    if isinstance(m, dict):
        return str(m.get("type", "") or "")
    return str(getattr(m, "type", "") or "")


def _msg_content(m: object):
    """Extract content field across message shapes without raising."""
    if isinstance(m, dict):
        return m.get("content", "")
    return getattr(m, "content", "")


def _to_lc(m: object) -> BaseMessage | None:
    """Coerce a stored/history message into a LangChain 'BaseMessage'.

    Includes only user/assistant messages for history replays; tools/system are
    dropped.

    Args:
        m: Message object or dict with 'type'/'content'.

    Returns:
        'HumanMessage'/'AIMessage' or None when unsupported.
    """
    if isinstance(m, BaseMessage):
        return m
    if isinstance(m, dict):
        t = _msg_type(m)
        c = _coerce_text_block(_msg_content(m))
        if t == "human":
            return HumanMessage(c)
        if t == "ai":
            return AIMessage(c)
        return None
    t = _msg_type(m)
    c = _coerce_text_block(_msg_content(m))
    if t == "human":
        return HumanMessage(c)
    if t == "ai":
        return AIMessage(c)
    return None


def _cfg_cp(user_id: str, session: str) -> dict:
    """Return minimal checkpoint config for saver/loader.

    Args:
        user_id: Owner user id.
        session: Thread id.

    Returns:
        Normalized config containing only checkpoint keys.
    """
    return _norm_config(
        {
            "configurable": {
                "thread_id": f"owner-{user_id}:{session}",
                "checkpoint_ns": CHECKPOINT_NS,
            }
        }
    )


def _cfg_graph(user_id: str, session: str, owner_id: str) -> dict:
    """Return graph runtime config while preserving checkpoint keys.

    Args:
        user_id: Owner user id.
        session: Thread id.
        owner_id: Owner id.

    Returns:
        Normalized config with 'user_id'/'owner_id' added for tools.
    """
    base = _cfg_cp(user_id, session)
    c = dict(base["configurable"])
    c.update({"user_id": user_id, "owner_id": owner_id})
    return _norm_config({"configurable": c})


def _require_thread_uuid(raw: str) -> str:
    """
    Validate that 'raw' is a UUID string and return the normalized value.

    Raises HTTP 400 with a friendly message instead of allowing Postgres CAST
    errors to bubble up as 500s.
    """
    try:
        return str(UUID(str(raw)))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid session id; please start a new chat and try again.",
        )


def _rate_limited_mem(user_id: str, now_min: int, limit: int) -> tuple[bool, int]:
    """
    Best-effort, in-memory rate limiter keyed by user + minute.

    Note: This is per-process and non-persistent; prefer a shared store in
    production. Cleanup is eager to avoid unbounded growth.

    Returns (is_blocked, count_for_window).
    """
    global _CHAT_RL
    try:
        _CHAT_RL
    except NameError:
        _CHAT_RL = {}

    # Eager cleanup: drop all keys outside the current minute window.
    _CHAT_RL = {k: v for k, v in _CHAT_RL.items() if k[1] == now_min}

    key = (user_id, now_min)
    count = _CHAT_RL.get(key, 0) + 1
    _CHAT_RL[key] = count
    return (count > limit, count)


async def _rate_limited(
    user_id: str, now_min: int, limit: int
) -> Tuple[bool, int, str]:
    """
    Rate limiting with optional Redis backend.

    Returns:
        (is_blocked, count, backend) where backend is "redis" or "memory".
    """
    redis_url = os.getenv("RATE_LIMIT_REDIS_URL")
    if redis_url:
        try:
            import redis.asyncio as redis  # type: ignore

            global _REDIS_CLIENT
            try:
                _REDIS_CLIENT
            except NameError:
                _REDIS_CLIENT = None
            if _REDIS_CLIENT is None:
                _REDIS_CLIENT = redis.from_url(
                    redis_url, encoding="utf-8", decode_responses=True
                )

            key = f"chat:rate:{user_id}:{now_min}"
            count = await _REDIS_CLIENT.incr(key)
            # Keep a short TTL so keys self-clean.
            if count == 1:
                await _REDIS_CLIENT.expire(key, _RATE_LIMIT_WINDOW_SECONDS * 2)
            return (count > limit, int(count), "redis")
        except Exception:
            # Fall through to memory if Redis is unavailable/misconfigured
            pass

    blocked, count = _rate_limited_mem(user_id, now_min, limit)
    return blocked, count, "memory"


@router.get("/chat")
async def chat_stream(
    request: Request,
    q: str = Query(..., description="User message"),
    session: str = Query(..., description="Client-side session/thread id"),
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
) -> EventSourceResponse:
    """Stream assistant responses for a single user message via SSE.

    Behavior:
    - Enforces size limit and optional per-minute rate limiting via env.
    - Validates that the thread belongs to the current owner; creates one if missing.
    - Emits only assistant text tokens; tool events are recorded in tracing.
    - May emit an 'archived' event if the thread exceeds TTL and returns early.

    Args:
        request: FastAPI request (used for headers and request id).
        q: User message text.
        session: Client thread id (UUID string).
        user: Authenticated owner.
        db: SQLAlchemy session.

    Returns:
        'EventSourceResponse' yielding SSE events.

    Raises:
        HTTPException: 413 for oversized input, 429 on rate limit, 400 for
        prompt-injection patterns, or 401/404 where applicable.
    """
    import os
    import time

    rid = getattr(request.state, "request_id", None)
    MAX_CHAT_Q_CHARS = int(os.getenv("MAX_CHAT_Q_CHARS", "2000") or 2000)
    if len(q or "") > MAX_CHAT_Q_CHARS:
        raise HTTPException(
            status_code=413, detail=f"Query too large (>{MAX_CHAT_Q_CHARS} chars)"
        )

    # Moderation first (blocks flagged content when moderation is available)
    _moderate_or_reject(q or "")

    flagged = _detect_prompt_injection(q or "")
    if flagged:
        log.warning(
            "rid=%s agent_chat.chat_stream prompt injection blocked: %s", rid, flagged
        )
        _log_event(
            "chat_prompt_blocked",
            rid=rid,
            owner_id=str(user.sub),
            thread_id=session,
            reason=flagged,
        )
        raise HTTPException(
            status_code=400,
            detail="Message flagged for unsafe instructions. Please rephrase without system-level commands.",
        )

    default_rl = (
        "15"
        if (os.getenv("ENVIRONMENT", "").lower() in ("prod", "production"))
        else "0"
    )
    RATE_LIMIT = int(os.getenv("RATE_LIMIT_CHAT_PER_MIN") or default_rl)
    if RATE_LIMIT > 0:
        now_min = int(time.time() // _RATE_LIMIT_WINDOW_SECONDS)
        blocked, count, backend = await _rate_limited(
            str(user.sub), now_min, RATE_LIMIT
        )
        if blocked:
            _bump("chat_rate_limited")
            _log_event(
                "chat_rate_limited",
                rid=rid,
                owner_id=str(user.sub),
                thread_id=session,
                minute=now_min,
                count=count,
                limit=RATE_LIMIT,
                backend=backend,
            )
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded. Try again shortly."
            )
    thread_uuid = _require_thread_uuid(session)

    # Guard: make sure the thread belongs to this owner
    row = db.execute(
        sa.text(
            "select 1 from agent_threads where id = CAST(:tid AS uuid) and user_id = :u"
        ),
        {"tid": thread_uuid, "u": user.sub},
    ).first()

    if not row:
        # If the client accidentally sent a stale/foreign id, create one for them
        with _db_commit(db):
            new_row = db.execute(
                sa.text(
                    "insert into agent_threads (user_id, title) values (:u, :t) returning id"
                ),
                {"u": user.sub, "t": "New chat"},
            ).first()
        thread_uuid = str(new_row.id)

    # Soft-archive: block new sends on stale threads (history still readable)
    TTL_DAYS = int(os.getenv("CHAT_THREAD_TTL_DAYS", "0") or 0)
    if TTL_DAYS > 0:
        ttl_seconds = TTL_DAYS * 24 * 60 * 60
        last_row = db.execute(
            sa.text(
                """
                select coalesce(max(m.created_at), t.created_at) as last_at
                from agent_threads t
                left join agent_messages m on m.thread_id = t.id
                where t.id = CAST(:tid AS uuid) and t.user_id = :u
                group by t.created_at
                """
            ),
            {"tid": thread_uuid, "u": user.sub},
        ).first()
        last_at = getattr(last_row, "last_at", None)
        if last_at is not None:
            age = (datetime.utcnow() - last_at.replace(tzinfo=None)).total_seconds()
            if age > ttl_seconds:
                _bump("chat_stream_archived")

                async def gen_archived():
                    """SSE generator for archived threads.

                    Emits a single 'archived' event with a reason and TTL days
                    to instruct clients to start a new conversation.
                    """
                    evt = json.dumps(
                        {
                            "reason": "archived",
                            "message": f"This conversation is archived after {TTL_DAYS} days of inactivity. Start a new chat to continue.",
                            "ttl_days": TTL_DAYS,
                        }
                    )
                    yield {"event": "archived", "data": evt}

                return EventSourceResponse(
                    gen_archived(),
                    media_type="text/event-stream",
                    ping=15000,
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "X-Accel-Buffering": "no",
                    },
                )

    # Idempotency: avoid duplicate user rows on refresh
    last = db.execute(
        sa.text(
            """
            select role, content from agent_messages
            where thread_id = CAST(:tid AS uuid)
            order by created_at desc, id desc
            limit 1
        """
        ),
        {"tid": thread_uuid},
    ).first()

    if not last or not (
        last.role == "user" and (last.content or "").strip() == q.strip()
    ):
        with _db_commit(db):
            db.execute(
                sa.text(
                    """
                    insert into agent_messages (thread_id, role, content)
                    values (CAST(:tid AS uuid), 'user', :c)
                """
                ),
                {"tid": thread_uuid, "c": q},
            )
    from langchain_core.messages import (
        HumanMessage,
        AIMessage,
    )

    client_tz = request.headers.get("X-Client-TZ")
    try:
        if client_tz:
            _ = ZoneInfo(client_tz)
    except Exception:
        client_tz = None

    # Use the config that keeps checkpoint keys minimal and passes runtime context only to the graph
    # Build a plain runtime config (no helpers), then you can revert to _cfg_graph after it works
    cfg = {
        "configurable": {
            "thread_id": f"owner-{user.sub}:{thread_uuid}",
            "checkpoint_ns": CHECKPOINT_NS,
            "user_id": str(user.sub),
            "owner_id": str(user.sub),
        },
        # LangChain/LangSmith: propagate useful context for tracing and filtering
        # These keys are safely ignored if LangSmith is not enabled.
        "tags": [
            f"owner:{user.sub}",
            f"thread:{thread_uuid}",
            "route:/api/agent/chat",
        ],
        "metadata": {
            "route": "/api/agent/chat",
            "owner_id": str(user.sub),
            "thread_id": thread_uuid,
            "env": os.getenv("ENVIRONMENT", "dev"),
        },
        "run_name": "AgentChatStream",
    }
    if client_tz:
        cfg["configurable"]["tz"] = client_tz

    try:
        await ensure_checkpoint_sane(cfg)
    except Exception:
        pass
    try:
        _log_event(
            "chat_stream_start",
            rid=rid,
            owner_id=str(user.sub),
            thread_id=thread_uuid,
            q_len=len(q or ""),
        )
    except Exception:
        pass

    _bump("chat_stream_started")

    async def gen():
        """SSE generator that streams assistant output and UI events.

        Behavior:
        - Seeds the agent with an optional owner timezone clock hint + user text.
        - Iterates the graph stream and emits:
          - '{event: "message", data: <text delta>}' for assistant token deltas
          - '{event: "message", data: "UI:EMAIL_DRAFT:<json>"}' when a
            'create_email_draft' ToolMessage is observed (for UI hooks)
          - '{event: "done", data: "ok"}' on completion
        - Stops early if the client disconnects.
        - Persists the concatenated assistant text as a single DB row in 'finally'.
        """
        owner_rec = db.query(User).filter(User.id == user.sub).first()
        if owner_rec and owner_rec.timezone:
            now_owner = datetime.now(ZoneInfo(owner_rec.timezone))
            clock_hint = SystemMessage(
                content=f"[clock] Owner timezone: {owner_rec.timezone}. Today is {now_owner.strftime('%Y-%m-%d')}."
            )
            seed_msgs = [clock_hint, HumanMessage(q)]
        else:
            seed_msgs = [HumanMessage(q)]

        buffer = []
        last_len = 0
        prev_full_text = ""

        try:
            try:
                stream = _graph.stream(
                    {"messages": seed_msgs}, config=cfg, stream_mode="values"
                )
                for update in stream:
                    if await request.is_disconnected():
                        break

                    msg = update["messages"][-1]
                    if (
                        isinstance(msg, ToolMessage)
                        or getattr(msg, "type", "") == "tool"
                    ):
                        try:
                            name = getattr(msg, "name", None) or ""
                            if name == "create_email_draft":
                                raw = msg.content
                                payload = None
                                if isinstance(raw, dict):
                                    if raw.get("marker") == "email_draft":
                                        payload = raw.get("payload")
                                elif isinstance(raw, str):
                                    s = raw.strip()
                                    import json as _json
                                    import re as _re

                                    try:
                                        j = _json.loads(s)
                                        if (
                                            isinstance(j, dict)
                                            and j.get("marker") == "email_draft"
                                        ):
                                            payload = j.get("payload")
                                    except Exception:
                                        pass
                                    if (
                                        payload is None
                                        and '"marker"' in s
                                        and "email_draft" in s
                                    ):
                                        for m in _re.finditer(r"\{.*?\}", s, _re.S):
                                            try:
                                                j = _json.loads(m.group(0))
                                                if (
                                                    isinstance(j, dict)
                                                    and j.get("marker") == "email_draft"
                                                ):
                                                    payload = j.get("payload")
                                                    break
                                            except Exception:
                                                continue
                                if payload:
                                    try:
                                        ui_json = json.dumps(
                                            {
                                                "draft_id": payload.get("draft_id"),
                                                "to": payload.get("to"),
                                                "to_name": payload.get("to_name"),
                                                "subject": payload.get("subject"),
                                                "text": payload.get("text"),
                                            }
                                        )
                                        yield {
                                            "event": "message",
                                            "data": f"UI:EMAIL_DRAFT:{ui_json}",
                                        }
                                        log.info(
                                            "chat_stream: injected UI:EMAIL_DRAFT for draft_id=%s",
                                            payload.get("draft_id"),
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        continue

                    # ---------- existing: stream assistant text ----------
                    if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                        extra = getattr(msg, "additional_kwargs", {}) or {}
                        if extra.get("event") == "error":
                            err_text = _coerce_text_block(
                                getattr(msg, "content", "")
                            ) or (
                                "Sorry, something went wrong while processing your request."
                            )
                            buffer.append(err_text)
                            yield {"event": "error", "data": err_text}
                            yield {"event": "message", "data": err_text}
                            prev_full_text = err_text
                            last_len = len(err_text)
                            continue

                        tc = getattr(msg, "tool_calls", None) or getattr(
                            msg, "additional_kwargs", {}
                        ).get("tool_calls")
                        if tc:
                            continue

                        full = _coerce_text_block(getattr(msg, "content", ""))
                        try:
                            if prev_full_text and not str(full).startswith(
                                str(prev_full_text)
                            ):
                                last_len = 0
                                try:
                                    log.info(
                                        "rid=%s chat_stream: resetting delta (non-prefix). prev_len=%d new_len=%d",
                                        rid,
                                        len(prev_full_text or ""),
                                        len(full or ""),
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            last_len = 0
                        prev_full_text = full
                        if full and len(full) > last_len:
                            delta = full[last_len:]
                            has_ui = (
                                "UI:EMAIL_DRAFT:" in delta or "UI:EMAIL_DRAFT:" in full
                            )
                            has_pending = (
                                "PENDING_EMAIL_SEND:" in delta
                                or "PENDING_EMAIL_SEND:" in full
                            )
                            has_json_marker = (
                                '"marker"' in delta and '"email_draft"' in delta
                            )
                            try:
                                log.info(
                                    "rid=%s chat_stream: emit delta len=%d full_len=%d ui=%s pending=%s json_marker=%s",
                                    rid,
                                    len(delta),
                                    len(full),
                                    has_ui,
                                    has_pending,
                                    has_json_marker,
                                )
                            except Exception:
                                pass
                            last_len = len(full)
                            buffer.append(delta)
                            yield {"event": "message", "data": delta}
            except Exception:
                # Guard against unexpected errors in the graph/tool pipeline so
                # the SSE stream fails gracefully instead of crashing.
                try:
                    log.exception("rid=%s chat_stream: graph stream failed", rid)
                except Exception:
                    pass
                err_msg = (
                    "Sorry, something went wrong while processing your request. "
                    "I've logged the error and stopped this action."
                )
                buffer.append(err_msg)
                # Emit a dedicated error event for UIs that want to distinguish
                # failures, while preserving the existing friendly text message.
                yield {"event": "error", "data": err_msg}
                yield {"event": "message", "data": err_msg}
        finally:
            final_text = "".join(buffer).strip()
            if final_text:
                with _db_commit(db):
                    db.execute(
                        sa.text(
                            """
                            insert into agent_messages (thread_id, role, content)
                            values (CAST(:tid AS uuid), 'ai', :c)
                        """
                        ),
                        {"tid": thread_uuid, "c": final_text},
                    )
            try:
                has_ui = "UI:EMAIL_DRAFT:" in final_text
                has_pending = "PENDING_EMAIL_SEND:" in final_text
                has_json_marker = (
                    '"marker"' in final_text and '"email_draft"' in final_text
                )
                log.info(
                    "rid=%s chat_stream: final assistant chunk len=%d ui=%s pending=%s json_marker=%s",
                    rid,
                    len(final_text),
                    has_ui,
                    has_pending,
                    has_json_marker,
                )
            except Exception:
                pass
            try:
                _log_event(
                    "chat_stream_finish",
                    rid=rid,
                    owner_id=str(user.sub),
                    thread_id=thread_uuid,
                    bytes=len(final_text or ""),
                )
            except Exception:
                pass

        # Best-effort prune of checkpoint to keep memory small
        try:
            await prune_checkpoint_if_needed(cfg)
        except Exception:
            pass

        _bump("chat_stream_completed")
        yield {"event": "done", "data": "ok"}

    return EventSourceResponse(
        gen(),
        ping=15000,
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/metrics")
async def metrics(user: TokenUser = Depends(require_owner)) -> dict:
    """Return minimal in-memory agent chat metrics for visibility."""
    return {k: int(v) for k, v in _METRICS.items()}


@router.get("/history")
async def chat_history(
    session: str = Query(...),
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Return prior user/assistant messages for a conversation thread."""
    rows = db.execute(
        sa.text(
            """
            select m.role, m.content, m.created_at
            from agent_messages m
            join agent_threads t on t.id = m.thread_id
            where m.thread_id = CAST(:tid AS uuid)
            and t.user_id = :u
            order by m.created_at asc, m.id asc
        """
        ),
        {"tid": session, "u": user.sub},
    ).all()

    out = []
    for r in rows:
        role = "user" if r.role == "user" else ("ai" if r.role == "ai" else "tool")
        out.append({"role": role, "content": r.content})

    return {"messages": out}


@router.delete("/reset")
async def chat_reset(
    session: str = Query(...),
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Delete all messages for a thread owned by the current user."""
    with _db_commit(db):
        db.execute(
            sa.text(
                """
                delete from agent_messages
                where thread_id = CAST(:tid AS uuid)
                and exists (
                        select 1 from agent_threads t
                        where t.id = CAST(:tid AS uuid)
                            and t.user_id = :u
                )
            """
            ),
            {"tid": session, "u": user.sub},
        )
    try:
        cfg = _cfg_all(user.sub, session, user.sub)
        await cp_put(
            cfg, {"messages": []}, metadata={"step": -1}, new_versions={"messages": 1}
        )
        await ensure_checkpoint_sane(cfg)
    except Exception:
        pass
    return {"ok": True}


@router.get("/conversations")
async def list_conversations(
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> Dict[str, Any]:
    """List conversations for the owner with last-message preview."""
    rows = db.execute(
        sa.text(
            """
            select
              t.id,
              t.title,
              t.created_at,
              m_last.content as last_content,
              coalesce(m_last.created_at, t.created_at) as updated_at
            from public.agent_threads t
            left join lateral (
              select m.content, m.created_at
              from public.agent_messages m
              where m.thread_id = t.id and m.role in ('user','ai')
              order by m.created_at desc, m.id desc
              limit 1
            ) as m_last on true
            where t.user_id = :u
            order by updated_at desc
            """
        ),
        {"u": user.sub},
    ).all()

    items = []
    for r in rows:
        items.append(
            {
                "id": str(r.id),
                "title": r.title,
                "created_at": r.created_at.isoformat(),
                "updated_at": (
                    r.updated_at.isoformat()
                    if getattr(r, "updated_at", None)
                    else r.created_at.isoformat()
                ),
                "last_preview": r.last_content or "",
            }
        )
    return {"items": items}


@router.post("/conversations")
async def create_conversation(
    title: str | None = None,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> Dict[str, Any]:
    """Create a new conversation thread and seed its checkpoint."""
    with _db_commit(db):
        row = db.execute(
            sa.text(
                "insert into public.agent_threads (user_id, title) values (:u, :t) returning id"
            ),
            {"u": user.sub, "t": title},
        ).first()

    seed_cfg = _cfg_graph(user.sub, str(row.id), user.sub)
    await cp_put(
        seed_cfg, {"messages": []}, metadata={"step": -1}, new_versions={"messages": 1}
    )
    return {
        "id": str(row.id),
        "title": title,
        "created_at": datetime.utcnow().isoformat(),
    }


@router.patch("/conversations/{cid}")
async def rename_conversation(
    cid: str,
    title: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> Dict[str, Any]:
    """Rename a conversation owned by the current user."""
    with _db_commit(db):
        db.execute(
            sa.text(
                "update public.agent_threads set title=:t where id=:id and user_id=:u"
            ),
            {"t": title, "id": cid, "u": user.sub},
        )
    return {"ok": True}


@router.delete("/conversations/{cid}")
async def delete_conversation(
    cid: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> Dict[str, Any]:
    """Delete a conversation and its messages for the current user."""
    with _db_commit(db):
        db.execute(
            sa.text(
                "delete from public.agent_messages where thread_id = CAST(:id AS uuid)"
            ),
            {"id": cid},
        )
        db.execute(
            sa.text(
                "delete from public.agent_threads where id = CAST(:id AS uuid) and user_id = :u"
            ),
            {"id": cid, "u": user.sub},
        )
    try:
        cfg = _cfg_all(user.sub, cid, user.sub)
        await cp_put(
            cfg, {"messages": []}, metadata={"step": -1}, new_versions={"messages": 1}
        )
        await ensure_checkpoint_sane(cfg)
    except Exception:
        pass
    return {"ok": True}


@router.get("/debug_checkpoint")
async def debug_checkpoint(
    request: Request,
    session: str = Query(...),
    dev_token: str | None = Query(None),
    user: TokenUser = Depends(require_owner),
):
    """Diagnostics for the agent checkpoint state (DEV only)."""
    _ensure_debug_enabled()
    cfg = _cfg_all(user.sub, session, user.sub)
    chk = await ensure_checkpoint_sane(cfg)

    chv = chk.get("channel_values") or {}
    chver = chk.get("channel_versions") or {}
    seen = chk.get("versions_seen") or {}

    messages = chv.get("messages") or []
    summary_raw = chv.get("summary")
    summary_text = (
        summary_raw if isinstance(summary_raw, str) else str(summary_raw or "")
    )

    def _msg_type(m) -> str:
        if isinstance(m, dict):
            return str(m.get("type") or m.get("role") or "")
        return str(getattr(m, "type", "") or getattr(m, "role", "") or "")

    msg_counts = {
        "total": len(messages),
        "human": sum(1 for m in messages if _msg_type(m) == "human"),
        "ai": sum(1 for m in messages if _msg_type(m) == "ai"),
        "tool": sum(1 for m in messages if _msg_type(m) == "tool"),
    }

    version_delta = {}
    for k, v in chver.items():
        try:
            delta = int(v) - int(seen.get(k, 0))
        except Exception:
            delta = 0
        if delta:
            version_delta[k] = delta

    def _types(d):
        return {k: type(v).__name__ for k, v in (d or {}).items()}

    return {
        "thread_id": f"owner-{user.sub}:{session}",
        "channel_values_keys": list(chv.keys()),
        "channel_versions": chver,
        "channel_versions_types": _types(chver),
        "versions_seen": seen,
        "versions_seen_types": _types(seen),
        "message_counts": msg_counts,
        "summary_chars": len(summary_text or ""),
        "summary_preview": (summary_text or "")[:160],
        "channel_version_delta": version_delta,
        "growth_flags": {
            "messages_over_200": msg_counts["total"] > 200,
            "missing_summary": not summary_text and msg_counts["total"] > 40,
        },
    }


@router.post("/repair_session")
async def repair_session(
    session: str = Query(...),
    user: TokenUser = Depends(require_owner),
):
    """Reset the session checkpoint to an empty message list (DEV only)."""
    _ensure_debug_enabled()
    cfg = _cfg_all(user.sub, session, user.sub)
    await cp_put(
        cfg,
        {"messages": []},
        metadata={"step": -1},
        new_versions={"messages": 1},
    )
    await ensure_checkpoint_sane(cfg)
    return {"ok": True}


@router.get("/reset_dev")
async def reset_dev(
    session: str = Query(...), dev_token: str = Query(...)
) -> Dict[str, Any]:
    """DEV wrapper for chat_reset guarded by a dev token."""
    _ensure_debug_enabled()
    if dev_token != DEV_TOKEN:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = type("U", (), {"sub": "__dev__"})()
    return await chat_reset(session=session, user=user)


@router.get("/repair_session_dev")
async def repair_dev(
    session: str = Query(...), dev_token: str = Query(...)
) -> Dict[str, Any]:
    """DEV wrapper for repair_session guarded by a dev token."""
    _ensure_debug_enabled()
    if dev_token != DEV_TOKEN:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = type("U", (), {"sub": "__dev__"})()
    return await repair_session(session=session, user=user)


@router.post("/email/approve/{draft_id}")
def approve_email(
    draft_id: UUID,
    body: dict,
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Approve and send a pending email draft owned by the user."""
    payload = dict(body or {})
    payload.setdefault("approve", True)

    send_body = SendBody(**payload)
    draft = send_outbox_email(draft_id=draft_id, body=send_body, db=db, user=user)

    return {
        "ok": draft.status.lower() == OutboxEmailStatus.SENT.value,
        "status": draft.status,
        "draft_id": draft.id,
        "to": draft.to,
        "subject": draft.subject,
    }


@router.post("/email/reject/{draft_id}")
def reject_email(
    draft_id: UUID,
    body: dict,
    user: TokenUser = Depends(require_owner),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Reject a pending email draft with an optional reason."""
    reason = (body or {}).get("reason") or "Rejected by owner"
    ob = (
        db.query(OutboxEmail)
        .filter(
            OutboxEmail.id == draft_id,
            OutboxEmail.owner_user_id == user.sub,
        )
        .first()
    )
    if not ob:
        raise HTTPException(404, "Draft not found")
    with _db_commit(db):
        ob.status = OutboxEmailStatus.REJECTED.value
        ob.rejected_reason = reason
        db.add(ob)
    return {"ok": True, "status": ob.status, "draft_id": str(ob.id)}
