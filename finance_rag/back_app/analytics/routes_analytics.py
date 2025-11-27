from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from math import ceil
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SASession

from ..core.db import get_db, get_or_create_session
from ..core.settings import PRICING
from .models_analytics import ChatTurn, Event, ToolInvocation

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# --------------------------- Helpers & Types ---------------------------------


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime object."""
    return datetime.now(timezone.utc)


def _sanitize_days(days: int) -> int:
    """
    Clamp a lookback window (in days) to the safe range [0, 3650].

    Args:
        days: Requested number of days.

    Returns:
        An integer in the inclusive range [0, 3650].
        Falls back to 7 if input is invalid or cannot be cast to int.
    """    
    try:
        d = int(days)
    except Exception:
        d = 7
    return max(0, min(d, 3650))


def _percentiles(
    values: Iterable[int | float], pcts: Tuple[float, ...] = (0.95, 0.99)
) -> Dict[str, float]:
    """
    Compute percentile statistics from a sequence of numeric values.

    Args:
        values: Sequence of numbers (ints or floats).
        pcts: Percentiles to compute (0–1 range). Defaults to (0.95, 0.99).

    Returns:
        Dict mapping e.g. "p95" → value, "p99" → value.
        Returns 0.0 for all percentiles if input is empty.
    """
    arr = sorted([v for v in values if v is not None])
    n = len(arr)
    out: Dict[str, float] = {}
    if n == 0:
        for p in pcts:
            out[f"p{int(p * 100)}"] = 0.0
        return out
    for p in pcts:
        k = max(1, int(ceil(p * n)))
        out[f"p{int(p * 100)}"] = float(arr[k - 1])
    return out


def _compute_cost(
    model: Optional[str], tokens_in: Optional[int], tokens_out: Optional[int]
) -> float:
    """
    Compute USD cost of a model invocation based on token usage.

    Args:
        model: Model name (looked up in settings.PRICING).
        tokens_in: Number of input tokens consumed.
        tokens_out: Number of output tokens produced.

    Returns:
        Total cost in USD (float). Returns 0.0 if model not in PRICING.
    """
    ti = max(int(tokens_in or 0), 0) / 1000.0
    to = max(int(tokens_out or 0), 0) / 1000.0
    p = PRICING.get((model or "").strip())
    if not p:
        return 0.0

    in_price = getattr(p, "input_per_1k", None)
    out_price = getattr(p, "output_per_1k", None)
    if in_price is None or out_price is None:
        if isinstance(p, dict):
            in_price = p.get("in", 0.0)
            out_price = p.get("out", 0.0)
        else:
            in_price = 0.0
            out_price = 0.0

    return ti * float(in_price or 0.0) + to * float(out_price or 0.0)


def _ensure_session(db: SASession, session_id: Optional[str]) -> str:
    """
    Ensure that a session row exists in the database.

    Args:
        db: SQLAlchemy session.
        session_id: Requested session ID (string or None).

    Returns:
        The canonical session ID string, guaranteed to exist in DB.
    """
    raw = (session_id or "").strip() or "anonymous"
    sid = get_or_create_session(db, raw, title="Analytics")
    exists = db.execute(
        text("SELECT 1 FROM sessions WHERE id = :sid LIMIT 1"),
        {"sid": sid},
    ).fetchone()
    if not exists:
        db.execute(
            text(
                "INSERT OR IGNORE INTO sessions (id, title, created_at) "
                "VALUES (:sid, :title, CURRENT_TIMESTAMP)"
            ),
            {"sid": sid, "title": "Analytics"},
        )
        db.commit()

    return sid


def _insert_turn(db: SASession, payload: Dict[str, Any], sid: str) -> None:
    """
    Insert a 'ChatTurn' row from telemetry payload.

    Args:
        db: SQLAlchemy session.
        payload: Telemetry dict containing role, tokens, model, etc.
        sid: Canonical session ID.
    """
    tokens_in = int(payload.get("tokens_in") or 0)
    tokens_out = int(payload.get("tokens_out") or 0)
    model = (payload.get("model") or "").strip() or None

    turn = ChatTurn(
        session_id=sid,
        role=(payload.get("role") or "assistant"),
        content=payload.get("response_preview") or "",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=_compute_cost(model, tokens_in, tokens_out),
        model=model,
        latency_ms=int(payload.get("latency_ms") or 0),
        had_rag=bool(payload.get("had_rag") or False),
        error=payload.get("error"),
    )
    db.add(turn)


def _insert_tool(db: SASession, payload: Dict[str, Any], sid: str) -> None:
    """
    Insert a 'ToolInvocation' row from telemetry payload.

    Args:
        db: SQLAlchemy session.
        payload: Telemetry dict containing tool name, args, latency, etc.
        sid: Canonical session ID.
    """
    inv = ToolInvocation(
        session_id=sid,
        turn_id=(
            payload.get("turn_id") if payload.get("turn_id") is not None else None
        ),
        tool_name=str(payload.get("tool_name") or "tool"),
        args=(payload.get("args") if payload.get("args") is not None else None),
        latency_ms=(
            int(payload.get("latency_ms"))
            if payload.get("latency_ms") is not None
            else 0
        ),
        ok=(bool(payload.get("ok")) if payload.get("ok") is not None else True),
        error=(str(payload.get("error")) if payload.get("error") is not None else None),
    )
    db.add(inv)


def _insert_event(db: SASession, payload: Dict[str, Any], sid: str) -> None:
    """
    Insert an 'Event' row from telemetry payload.

    Args:
        db: SQLAlchemy session.
        payload: Telemetry dict containing event name and props.
        sid: Canonical session ID.
    """
    ev = Event(
        session_id=sid,
        name=str(payload.get("name") or "event"),
        props=payload.get("props") or {},
    )
    db.add(ev)


# ----------------------------- Ingest endpoint --------------------------------


class IngestOk(TypedDict):
    ok: Literal[True]


@router.post("/ingest", response_model=IngestOk)
async def ingest(payload: Dict[str, Any], db: SASession = Depends(get_db)) -> IngestOk:
    """
    REST endpoint: Ingest telemetry into analytics tables.

    Supports 3 types:
      - "turn": Conversation turns (user/assistant messages).
      - "tool": Tool invocations from the agent.
      - "event": Arbitrary events (e.g., UI clicks).

    Args:
        payload: JSON body with telemetry fields.
        db: Database session dependency.

    Returns:
        {"ok": True} on success.

    Raises:
        HTTPException(400) if payload type is invalid or DB integrity fails.
        HTTPException(500) for unexpected server errors.
    """
    t = payload.get("type")
    if t not in {"turn", "tool", "event"}:
        raise HTTPException(status_code=400, detail="Unknown type")

    sid = _ensure_session(db, payload.get("session_id"))
    payload["session_id"] = sid

    def _do_insert() -> None:
        if t == "turn":
            _insert_turn(db, payload, sid)
        elif t == "tool":
            _insert_tool(db, payload, sid)
        else:
            _insert_event(db, payload, sid)

    try:
        _do_insert()
        db.commit()
        return {"ok": True}
    except IntegrityError as e:
        db.rollback()
        msg = str(getattr(e, "orig", e))
        if "FOREIGN KEY constraint failed" in msg:
            _ = _ensure_session(db, sid)
            try:
                _do_insert()
                db.commit()
                return {"ok": True}
            except IntegrityError as e2:
                db.rollback()
                raise HTTPException(
                    status_code=400,
                    detail=f"ingest failed: IntegrityError: {getattr(e2, 'orig', e2)}",
                )
        raise HTTPException(
            status_code=400,
            detail=f"ingest failed: IntegrityError: {getattr(e, 'orig', e)}",
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"ingest failed: {type(e).__name__}: {e}"
        )


# -------------------------- Summary / Series / Top ----------------------------


class SummaryResponse(TypedDict):
    turns: int
    cost_usd: float
    tokens_in: int
    tokens_out: int
    avg_latency_ms: float
    rag_rate: float
    error_rate: float
    tool_error_rate: float


@router.get("/summary", response_model=SummaryResponse)
def summary(
    days: int = Query(7, description="Lookback window in days (0 = all time)"),
    db: SASession = Depends(get_db),
) -> SummaryResponse:
    """
    REST endpoint: Get high-level aggregates for dashboard header.

    Includes:
      - Total turns
      - Total tokens in/out
      - Average latency
      - Total cost (USD)
      - RAG rate, error rate, tool error rate

    Args:
        days: Lookback window in days (0 = all time).
        db: Database session.

    Returns:
        Dict with aggregate metrics.
    """
    days = _sanitize_days(days)

    turns_q = db.query(
        func.count().label("turns"),
        func.coalesce(func.sum(ChatTurn.tokens_in), 0).label("tokens_in"),
        func.coalesce(func.sum(ChatTurn.tokens_out), 0).label("tokens_out"),
        func.coalesce(func.avg(ChatTurn.latency_ms), 0).label("avg_latency_ms"),
        func.coalesce(
            func.sum(case((func.length(func.trim(ChatTurn.error)) > 0, 1), else_=0)), 0
        ).label("error_count"),
        func.coalesce(func.sum(case((ChatTurn.had_rag == True, 1), else_=0)), 0).label(
            "rag_count"
        ),
    )
    if days > 0:
        since = _utc_now() - timedelta(days=days)
        turns_q = turns_q.filter(ChatTurn.created_at >= since)

    trow = turns_q.one()
    turns = int(trow.turns or 0)
    tokens_in = int(trow.tokens_in or 0)
    tokens_out = int(trow.tokens_out or 0)
    avg_latency_ms = float(trow.avg_latency_ms or 0)
    rag_count = int(trow.rag_count or 0)
    error_count = int(trow.error_count or 0)

    cost_q = db.query(func.coalesce(func.sum(ChatTurn.cost_usd), 0.0))
    if days > 0:
        cost_q = cost_q.filter(ChatTurn.created_at >= since)
    cost_usd = float(cost_q.scalar() or 0.0)

    tools_q = db.query(
        func.count().label("inv"),
        func.coalesce(
            func.sum(case((ToolInvocation.ok == False, 1), else_=0)), 0
        ).label("fail"),
    )
    if days > 0:
        tools_q = tools_q.filter(ToolInvocation.created_at >= since)
    tool_row = tools_q.one()
    tool_inv = int(tool_row.inv or 0)
    tool_fail = int(tool_row.fail or 0)

    return {
        "turns": turns,
        "cost_usd": round(cost_usd, 6),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "avg_latency_ms": round(avg_latency_ms, 1),
        "rag_rate": round((rag_count / turns), 3) if turns else 0.0,
        "error_rate": round((error_count / turns), 3) if turns else 0.0,
        "tool_error_rate": round((tool_fail / tool_inv), 3) if tool_inv else 0.0,
    }


class SeriesPoint(TypedDict):
    t: str
    tokens: int


class SeriesResponse(TypedDict):
    points: List[SeriesPoint]


def _bucket_key(dt: datetime, days: int) -> datetime:
    """
    Normalize a timestamp into a bucketing key.

    Rules:
      - If lookback is 1–3 days → bucket to the nearest hour.
      - Otherwise → bucket to the nearest day.

    Args:
        dt: Datetime to normalize.
        days: Lookback window in days.

    Returns:
        A tz-aware datetime rounded to the bucket boundary.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if days in (1, 3):
        return dt.replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


@router.get("/series", response_model=SeriesResponse)
def series(
    days: int = Query(7, description="Lookback window in days (0 = all time)"),
    db: SASession = Depends(get_db),
) -> SeriesResponse:
    """
    REST endpoint: Get time-series of total token usage.

    Tokens (in + out) are aggregated per bucket:
      - Hourly if days=1 or 3
      - Daily otherwise

    Args:
        days: Lookback window in days (0 = all time).
        db: Database session.

    Returns:
        Dict with a list of {t: ISO timestamp, tokens: int}.
    """
    days = _sanitize_days(days)
    q = db.query(ChatTurn)
    if days > 0:
        since = _utc_now() - timedelta(days=days)
        q = q.filter(ChatTurn.created_at >= since)
    rows = q.order_by(ChatTurn.created_at.asc()).all()

    buckets: dict[datetime, dict[str, int]] = defaultdict(lambda: {"tokens": 0})
    for r in rows:
        k = _bucket_key(r.created_at, days)
        buckets[k]["tokens"] += int(r.tokens_in or 0) + int(r.tokens_out or 0)

    points: List[SeriesPoint] = [
        {"t": k.isoformat(), "tokens": v["tokens"]} for k, v in sorted(buckets.items())
    ]
    return {"points": points}


class TopTool(TypedDict):
    name: str
    count: int


class SlowTurn(TypedDict):
    id: int
    latency_ms: int
    model: Optional[str]
    created_at: str


class ErrorItem(TypedDict):
    id: int
    error: Optional[str]
    created_at: str


class TopResponse(TypedDict):
    tools: List[TopTool]
    slow_turns: List[SlowTurn]
    errors: List[ErrorItem]


@router.get("/top", response_model=TopResponse)
def top(
    days: int = Query(7, description="Lookback window in days (0 = all time)"),
    db: SASession = Depends(get_db),
) -> TopResponse:
    """
    REST endpoint: Get "top" analytics.

    Includes:
      - Most-used tools (top 10 by count)
      - Slowest conversation turns (top 10 by latency)
      - Most recent errors (last 10 turns with errors)

    Args:
        days: Lookback window in days (0 = all time).
        db: Database session.

    Returns:
        Dict with tools, slow_turns, and errors lists.
    """
    days = _sanitize_days(days)

    tools_q = db.query(ToolInvocation.tool_name)
    slow_q = db.query(ChatTurn)
    err_q = db.query(ChatTurn).filter(ChatTurn.error.isnot(None))

    if days > 0:
        since = _utc_now() - timedelta(days=days)
        tools_q = tools_q.filter(ToolInvocation.created_at >= since)
        slow_q = slow_q.filter(ChatTurn.created_at >= since)
        err_q = err_q.filter(ChatTurn.created_at >= since)

    tools = [t[0] for t in tools_q.all()]
    tool_counts = Counter(tools)

    slow_turns = slow_q.order_by(ChatTurn.latency_ms.desc()).limit(10).all()
    errors = err_q.order_by(ChatTurn.created_at.desc()).limit(10).all()

    def _iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    return {
        "tools": [{"name": k, "count": v} for k, v in tool_counts.most_common(10)],
        "slow_turns": [
            {
                "id": t.id,
                "latency_ms": int(t.latency_ms or 0),
                "model": t.model,
                "created_at": _iso(t.created_at),
            }
            for t in slow_turns
        ],
        "errors": [
            {"id": e.id, "error": e.error, "created_at": _iso(e.created_at)}
            for e in errors
        ],
    }


# ------------------------------- Breakdown ------------------------------------


class ByModelItem(TypedDict):
    model: str
    turns: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


class LatencyStats(TypedDict):
    avg_ms: float
    median_ms: int
    p95_ms: int
    p99_ms: int


class Totals(TypedDict):
    turns: int
    cost_usd: float
    tokens_in: int
    tokens_out: int


class BreakdownResponse(TypedDict):
    by_model: List[ByModelItem]
    latency: LatencyStats
    totals: Totals


@router.get("/breakdown", response_model=BreakdownResponse)
def breakdown(
    days: int = Query(7, description="Lookback window in days (0 = all time)"),
    db: SASession = Depends(get_db),
) -> BreakdownResponse:
    """
    REST endpoint: Get detailed breakdown of analytics.

    Includes:
      - Totals (turns, tokens, cost)
      - Latency stats (avg, median, p95, p99)
      - By-model aggregates (sorted by cost)

    Args:
        days: Lookback window in days (0 = all time).
        db: Database session.

    Returns:
        Dict with totals, latency stats, and by-model list.
    """
    days = _sanitize_days(days)

    q = db.query(ChatTurn)
    if days > 0:
        since = _utc_now() - timedelta(days=days)
        q = q.filter(ChatTurn.created_at >= since)

    rows: List[ChatTurn] = q.all()

    total_turns = len(rows)
    total_cost = sum(float(r.cost_usd or 0.0) for r in rows)
    total_in = sum(int(r.tokens_in or 0) for r in rows)
    total_out = sum(int(r.tokens_out or 0) for r in rows)

    latencies = [int(r.latency_ms or 0) for r in rows]
    avg_ms = (sum(latencies) / total_turns) if total_turns else 0.0
    med_ms = int(median(latencies)) if total_turns else 0
    pct = _percentiles(latencies, (0.95, 0.99))
    p95_ms = int(pct["p95"])
    p99_ms = int(pct["p99"])

    agg: Dict[str, ByModelItem] = {}
    for r in rows:
        m = (r.model or "unknown").strip() or "unknown"
        a = agg.setdefault(
            m,
            {"model": m, "turns": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0},
        )
        a["turns"] += 1
        a["tokens_in"] += int(r.tokens_in or 0)
        a["tokens_out"] += int(r.tokens_out or 0)
        a["cost_usd"] += float(r.cost_usd or 0.0)

    by_model = sorted(agg.values(), key=lambda x: (-x["cost_usd"], -x["turns"]))

    return {
        "by_model": by_model,
        "latency": {
            "avg_ms": round(avg_ms, 1),
            "median_ms": med_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
        },
        "totals": {
            "turns": total_turns,
            "cost_usd": round(total_cost, 6),
            "tokens_in": total_in,
            "tokens_out": total_out,
        },
    }
