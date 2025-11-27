from __future__ import annotations

import os
import time
from typing import Any, Mapping, Optional, Union

from loguru import logger
from pydantic import BaseModel
from fastapi import HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from ..core.settings import settings

# -------------------------------------------------------------------
# Chat history → LangChain messages
# -------------------------------------------------------------------


def to_lc_messages(history: list[dict]) -> list[BaseModel]:
    """
    Convert persisted chat rows (role/content dicts) into LangChain message objects.

    Accepts items like {"role": "user"|"assistant"|"system"|"tool", "content": "...", ...}
    and returns the corresponding message classes.

    Notes:
        - Unknown roles are ignored.
        - For "tool" messages, 'tool_call_id' (if present) is passed through.

    Args:
        history: List of message dicts.

    Returns:
        List of LangChain message objects.
    """
    out: list[BaseModel] = []
    for m in history or []:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id")))
    return out


# -------------------------------------------------------------------
# Rate limiting: simple token bucket
# -------------------------------------------------------------------


class _TokenBucket:
    """
    Simple in-memory token-bucket rate limiter.

    Concept
    -------
    - Each 'key' (e.g., client IP) has a bucket with:
        * capacity = 'burst' tokens,
        * refill rate = 'rate_per_sec' tokens per second,
        * current balance tracked as a float.
    - A request "costs" 'tokens' (default 1.0). If the balance covers it,
      the request is allowed and the balance is reduced; otherwise it's denied.
    - Balances refill over time and are capped at 'burst'.

    Notes
    -----
    - Uses 'time.monotonic()' for stable elapsed-time calculations.
    - In-memory only: state resets on process restart and is **not**
      shared across workers/instances. Use Redis (or similar) for
      distributed deployments.
    - Not thread-safe for heavy multi-threaded mutation; typical async
      FastAPI single-threaded event loop usage is fine.

    Parameters
    ----------
    rate_per_sec : float
        Refill rate in tokens per second. Clamped to at least 0.1.
    burst : float
        Maximum bucket size (and initial balance). Clamped to at least 1.0.
    """

    def __init__(self, rate_per_sec: float, burst: float) -> None:
        self.rate: float = max(rate_per_sec, 0.1)
        self.burst: float = max(burst, 1.0)
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, tokens: float = 1.0) -> bool:
        """
        Try to consume 'tokens' for the given 'key'.

        Recomputes the available balance by refilling from the last timestamp
        at 'self.rate', caps it at 'self.burst', then either:
          - denies and saves the refilled balance if insufficient, or
          - deducts 'tokens', saves, and allows.

        Args:
            key: Identifier for the caller (e.g., "ip:1.2.3.4" or a user id).
            tokens: Cost of this operation in tokens (may be fractional).

        Returns:
            True if allowed (tokens deducted), False otherwise.
        """
        now = time.monotonic()
        last, avail = self._buckets.get(key, (now, self.burst))
        avail = min(self.burst, avail + (now - last) * self.rate)

        if avail < tokens:
            self._buckets[key] = (now, avail)
            return False

        self._buckets[key] = (now, avail - tokens)
        return True


class _RedisTokenBucket:
    """
    Distributed token bucket backed by Redis.

    Stores bucket state per key in Redis as:
      HSET key tokens <float> ts <float>
    Uses a Lua script to refill and deduct atomically.

    This is a minimal drop-in for production use when multiple workers
    or replicas need to share rate-limit state.
    """

    _SCRIPT = """
    local key       = KEYS[1]
    local cost      = tonumber(ARGV[1])
    local rate      = tonumber(ARGV[2])
    local burst     = tonumber(ARGV[3])
    local now       = tonumber(ARGV[4])
    local ttl_sec   = tonumber(ARGV[5])

    local data = redis.call('HMGET', key, 'tokens', 'ts')
    local tokens = tonumber(data[1]) or burst
    local ts     = tonumber(data[2]) or now

    local delta = now - ts
    if delta < 0 then delta = 0 end
    tokens = math.min(burst, tokens + delta * rate)

    if tokens < cost then
      redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
      redis.call('EXPIRE', key, ttl_sec)
      return 0
    end

    tokens = tokens - cost
    redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
    redis.call('EXPIRE', key, ttl_sec)
    return 1
    """

    def __init__(
        self, url: str, rate_per_sec: float, burst: float, namespace: str = "rate"
    ) -> None:
        try:
            import redis  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Redis is not installed; install 'redis' package to use _RedisTokenBucket."
            ) from e

        self.rate: float = max(rate_per_sec, 0.1)
        self.burst: float = max(burst, 1.0)
        self.namespace = (namespace or "rate").strip(":")
        self.client = redis.Redis.from_url(url)
        try:
            self._lua = self.client.register_script(self._SCRIPT)
        except Exception as e:
            raise RuntimeError(f"Failed to register rate-limit Lua script: {e}") from e

    def allow(self, key: str, tokens: float = 1.0) -> bool:
        now = time.time()
        ttl = max(1, int((self.burst / self.rate) * 2))
        namespaced = f"{self.namespace}:{key}"
        try:
            res = self._lua(
                keys=[namespaced],
                args=[tokens, self.rate, self.burst, now, ttl],
            )
            return bool(int(res))
        except Exception as e:
            logger.warning(
                "Redis rate-limit error (allowing request by default): %s", e
            )
            return True


# -------------------------------------------------------------------
# Percent parsing helper
# -------------------------------------------------------------------

PercentLike = Union[float, str, None]


def parse_percent(
    x: PercentLike, *, empty_returns: Optional[float] = None
) -> Optional[float]:
    """
    Parse a percent-like input into *percentage points* (e.g., 7.0 means 7%).

    Accepted inputs:
      - "7%", " 0.5% "   -> numeric part in percentage points
      - "0.07", "7"      -> parsed to float and normalized (see rules)
      - 0.07, 7          -> normalized (see rules)
      - None / "" / " "  -> returns 'empty_returns' (default: None)

    Normalization rules:
      - If the non-% value v is in [0, 1], treat as a fraction and scale:
            0.07 → 7.0
            1.0  → 100.0
      - Otherwise treat as already in percentage points:
            7 → 7.0, 125 → 125.0, -5 → -5.0

    Args:
        x: Percent-like input.
        empty_returns: Value to return for empty/None inputs.

    Returns:
        Percentage points as float (e.g., 7.0), or 'empty_returns'.
    """
    if x is None:
        return empty_returns
    s = str(x).strip()
    if s == "":
        return empty_returns

    if s.endswith("%"):
        try:
            return float(s[:-1].strip())
        except ValueError:
            return empty_returns

    try:
        v = float(s)
    except ValueError:
        return empty_returns

    return v * 100.0 if 0.0 <= v <= 1.0 else v


# -------------------------------------------------------------------
# Analytics emitter (best-effort)
# -------------------------------------------------------------------


async def _emit_analytics(meta: Mapping[str, Any]) -> None:
    """
    Analytics disabled: no-op to avoid network overhead.
    """
    return None


# -------------------------------------------------------------------
# Admin gate
# -------------------------------------------------------------------


def _require_admin(request: Request) -> None:
    """
    Enforce an admin header when ADMIN_KEY is set.

    - If env var 'ADMIN_KEY' is **unset**, this is a no-op (dev friendly).
    - If set, requires header 'X-Admin-Key: <ADMIN_KEY>' or raises 403.

    Args:
        request: FastAPI 'Request'.

    Raises:
        HTTPException(403): when the header does not match the configured key.
    """
    key = os.getenv("ADMIN_KEY")
    if not key:
        return
    if request.headers.get("X-Admin-Key") != key:
        raise HTTPException(status_code=403, detail="Forbidden")
