from __future__ import annotations

import json
import os
import urllib.request
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, Optional, Iterator, TypedDict
from loguru import logger

from ..core.settings import PRICING, TRACK_TOKENS, EMBEDDING_PRICING

ANALYTICS_BASE_URL = os.getenv("ANALYTICS_BASE_URL")

# ---------------------------- pricing helpers ----------------------------


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """
    Compute the USD cost for a chat/completions request.

    Looks up the model in ''PRICING'' (supports both object-style attrs
    ''input_per_1k'' / ''output_per_1k'' and dict-style ''{"in": ..., "out": ...}'').

    Args:
        model: Model name key into ''PRICING''.
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of completion/output tokens.

    Returns:
        The total cost in USD, or ''None'' if the model is not present
        or lacks pricing information.
    """
    mp = PRICING.get(model)
    if not mp:
        return None
    in_price = getattr(mp, "input_per_1k", None)
    out_price = getattr(mp, "output_per_1k", None)
    if in_price is None or out_price is None and isinstance(mp, dict):
        in_price = mp.get("in")
        out_price = mp.get("out")
    if in_price is None or out_price is None:
        return None
    return (input_tokens / 1000.0) * float(in_price) + (output_tokens / 1000.0) * float(
        out_price
    )


def _embedding_cost_usd(model: str, tokens: int) -> Optional[float]:
    """
    Compute the USD cost for an embeddings request.

    Looks up the model in ''EMBEDDING_PRICING'' (supports object-style
    ''input_per_1k'' or dict-style ''{"in": ...}'').

    Args:
        model: Embedding model name key into ''EMBEDDING_PRICING''.
        tokens: Number of input tokens to embed.

    Returns:
        The total cost in USD, or ''None'' if the model is not present
        or lacks pricing information.
    """
    ep = EMBEDDING_PRICING.get(model)
    if not ep:
        return None
    price = getattr(ep, "input_per_1k", None)
    if price is None and isinstance(ep, dict):
        price = ep.get("in")
    return (tokens / 1000.0) * float(price or 0.0)


def log_llm_usage(
    tag: str, model: str, usage: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a structured token-usage line for chat/completions.

    Respects ''TRACK_TOKENS''; if disabled, this is a no-op. Accepts usage payloads
    from various SDKs (keys like ''prompt_tokens''/''input_tokens'', etc.) and
    computes cost via ''_cost_usd''.

    Args:
        tag: Logical tag for the operation (e.g., ''"chat"'', ''"agent"'').
        model: Model name used for pricing lookup.
        usage: Dict containing token counts (prompt/input, completion/output, total).
        extra: Optional extra key/values to merge into the logged payload.

    Returns:
        None. Writes a structured line via ''logger.info''.
    """
    if not TRACK_TOKENS:
        return
    it = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    ot = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    tt = int(usage.get("total_tokens", it + ot))
    cost = _cost_usd(model, it, ot)
    payload = {
        "tag": tag,
        "model": model,
        "input_tokens": it,
        "output_tokens": ot,
        "total_tokens": tt,
        "cost_usd": round(cost, 6) if cost is not None else None,
    }
    if extra:
        payload.update(extra)
    logger.info(f"[TOKENS] {payload}")


def log_embedding_usage(
    model: str, tokens: int, extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a structured token-usage line for embeddings calls.

    Respects ''TRACK_TOKENS''; if disabled, this is a no-op.

    Args:
        model: Embedding model name used for pricing lookup.
        tokens: Number of input tokens embedded.
        extra: Optional extra key/values to merge into the logged payload.

    Returns:
        None. Writes a structured line via ''logger.info''.
    """
    if not TRACK_TOKENS:
        return
    cost = _embedding_cost_usd(model, tokens)
    payload = {
        "tag": "embeddings",
        "model": model,
        "input_tokens": tokens,
        "cost_usd": round(cost, 6) if cost is not None else None,
    }
    if extra:
        payload.update(extra)
    logger.info(f"[TOKENS] {payload}")


# ---- LangChain callbacks --------------------------------------------------

try:
    from langchain_community.callbacks.openai_info import (
        OpenAICallbackHandler as _LCOpenAICB,
    )
except Exception:
    try:
        from langchain.callbacks import OpenAICallbackHandler as _LCOpenAICB
    except Exception:
        _LCOpenAICB = None

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class _TokenTally(BaseCallbackHandler):
    """
    Fallback token counter when the official OpenAI callback handler is unavailable.

    This handler aggregates usage information from multiple locations on
    ''LLMResult'' to provide robustness across LangChain/OpenAI SDK versions.
    Accumulates:
      - ''prompt_tokens'' (a.k.a. ''input_tokens'' / ''prompt_token_count'')
      - ''completion_tokens'' (a.k.a. ''output_tokens'' / ''output_token_count'')
      - ''total_tokens'' (falls back to ''prompt + completion'')
    """


    def __init__(self) -> None:
        """Initialize counters and seen-chain tracking."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._seen_chains = set()

    def _merge_usage(self, u: dict) -> None:
        """
        Merge a usage-like dict into the running tallies.

        Args:
            u: Dict that may contain keys like ''prompt_tokens'', ''input_tokens'',
               ''prompt_token_count'', ''completion_tokens'', ''output_tokens'',
               ''output_token_count'', and ''total_tokens''.
        """
        if not isinstance(u, dict):
            return
        pt = int(
            u.get("prompt_tokens")
            or u.get("input_tokens")
            or u.get("prompt_token_count")
            or 0
        )
        ct = int(
            u.get("completion_tokens")
            or u.get("output_tokens")
            or u.get("output_token_count")
            or 0
        )
        tt = int(u.get("total_tokens") or (pt + ct))
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt if tt else (pt + ct)

    def on_llm_end(self, result: LLMResult, **kwargs) -> None:
        """
        LangChain callback invoked at the end of an LLM run.

        Attempts to extract usage info from ''llm_output'', per-generation
        ''response_metadata'' and ''generation_info''.
        """
        md = getattr(result, "llm_output", None) or {}
        self._merge_usage(md.get("token_usage") or md.get("usage") or {})
        try:
            gens = getattr(result, "generations", None) or []
            for glist in gens:
                for g in glist or []:
                    msg = getattr(g, "message", None)
                    if msg and hasattr(msg, "response_metadata"):
                        rm = getattr(msg, "response_metadata", {}) or {}
                        self._merge_usage(rm.get("token_usage", rm))
                    gi = getattr(g, "generation_info", None) or {}
                    self._merge_usage(gi.get("token_usage", gi))
        except Exception:
            pass

    def on_chain_end(
        self, outputs: dict, *, run_id, parent_run_id=None, **kwargs
    ) -> None:
        """
        LangChain callback invoked at the end of a chain run.

        Merges usage from ''outputs.get("usage")'' if present and ensures
        the same ''run_id'' is not processed more than once.
        """
        if run_id in self._seen_chains:
            return
        self._seen_chains.add(run_id)
        try:
            self._merge_usage(outputs.get("usage") or {})
        except Exception:
            pass


class _UsageBag(TypedDict, total=False):
    """
    Structured container returned by ''with_token_log'' to pass callback(s)
    into LangChain calls and to read back token/cost totals after the block.

    Keys (optional unless stated otherwise):
        model (str): Model name used for pricing (required in construction).
        callbacks (list): Callback handlers to attach to LangChain calls.
        prompt_tokens (int): Accumulated prompt/input tokens.
        completion_tokens (int): Accumulated completion/output tokens.
        total_tokens (int): Accumulated total tokens.
        cost_usd (float): Computed USD cost for the operation.
        tag (str): Logical tag for the operation (e.g., ''"chat"'').
        session_id (str): Optional session identifier to forward to analytics.
        turn_id (int): Optional turn identifier for tool telemetry correlation.
    """
    model: str
    callbacks: list
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    tag: str
    session_id: str
    turn_id: int


@contextmanager
def with_token_log(
    model: str, *, tag: str = "chat", session_id: Optional[str] = None
) -> Iterator[_UsageBag]:
    """
    Context-manage one logical LLM operation (agent/chain/call) to collect usage.

    Attaches either LangChain's ''OpenAICallbackHandler'' (if available) or a
    lightweight fallback (''_TokenTally''). After the block exits, updates the
    returned bag with token counts and computed cost, and logs it via
    ''log_llm_usage''.

    Usage:
        >>> with with_token_log("gpt-4o-mini", tag="chat") as usage:
        ...     chain.invoke(inputs, config={"callbacks": usage["callbacks"]})

    Args:
        model: Model name used for pricing lookup.
        tag: Logical tag for grouping operations (default: ''"chat"'').
        session_id: Optional session id to forward to analytics logs.

    Yields:
        A mutable ''_UsageBag'' dict. Use ''bag["callbacks"]'' when invoking
        LangChain components so usage is captured.

    Notes:
        The context manager always logs a usage line on exit (respecting
        ''TRACK_TOKENS''). Costs use pricing from ''PRICING''.
    """
    bag: _UsageBag = {
        "model": model,
        "tag": tag,
        "callbacks": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if session_id:
        bag["session_id"] = session_id

    if _LCOpenAICB:
        oaicb = _LCOpenAICB()
        bag["callbacks"] = [oaicb]
        try:
            yield bag
        finally:
            pt = int(getattr(oaicb, "prompt_tokens", 0))
            ct = int(getattr(oaicb, "completion_tokens", 0))
            tt = int(getattr(oaicb, "total_tokens", pt + ct))
            bag.update(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                cost_usd=_cost_usd(model, pt, ct) or 0.0,
            )
            log_llm_usage(
                tag,
                model,
                bag,
                extra={"session_id": session_id} if session_id else None,
            )
    else:
        tally = _TokenTally()
        bag["callbacks"] = [tally]
        try:
            yield bag
        finally:
            pt = int(tally.prompt_tokens)
            ct = int(tally.completion_tokens)
            tt = int(tally.total_tokens if tally.total_tokens else (pt + ct))
            bag.update(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                cost_usd=_cost_usd(model, pt, ct) or 0.0,
            )
            log_llm_usage(
                tag,
                model,
                bag,
                extra={"session_id": session_id} if session_id else None,
            )


# ---- Analytics ingestion (endpoint-first) --------------------------------


def _post_analytics(path: str, payload: dict) -> None:
    """
    Analytics disabled: no-op to avoid extra network calls.
    """
    return None


def on_llm_end(meta: dict) -> None:
    """
    Send a ''type="turn"'' ingest (assistant turn) to analytics if configured.

    Expected keys in ''meta'':
        - ''session_id'' (str)
        - ''response_preview'' (str, truncated to 4000 chars)
        - ''tokens_in'' (int)
        - ''tokens_out'' (int)
        - ''model'' (str)
        - ''latency_ms'' (int)
        - ''had_rag'' (bool)
        - ''error'' (str|None)

    Args:
        meta: Dictionary with the fields listed above.

    Returns:
        None. Posts to ''/api/analytics/ingest'' when base URL is set.
    """
    payload = {
        "type": "turn",
        "session_id": (meta.get("session_id") or "").strip() or "anonymous",
        "role": "assistant",
        "response_preview": (meta.get("response_preview") or "")[:4000],
        "tokens_in": int(meta.get("tokens_in", 0) or 0),
        "tokens_out": int(meta.get("tokens_out", 0) or 0),
        "model": meta.get("model"),
        "latency_ms": int(meta.get("latency_ms", 0) or 0),
        "had_rag": bool(meta.get("had_rag", False)),
        "error": meta.get("error"),
    }
    _post_analytics("/api/analytics/ingest", payload)


def log_tool(
    session_id: str,
    turn_id: int | None,
    name: str,
    args: dict,
    ms: int,
    ok: bool,
    error: str | None,
) -> None:
    """
    Send a ''type="tool"'' ingest to analytics if configured.

    Args:
        session_id: Session identifier (falls back to ''"anonymous"'' if blank).
        turn_id: Optional associated turn id.
        name: Tool name (e.g., ''"quote_lookup"'').
        args: Tool arguments payload (JSON-serializable).
        ms: Tool latency in milliseconds.
        ok: Whether the tool call succeeded.
        error: Optional error string if the tool failed.

    Returns:
        None. Posts to ''/api/analytics/ingest'' when base URL is set.
    """
    payload = {
        "type": "tool",
        "session_id": (session_id or "").strip() or "anonymous",
        "turn_id": turn_id,
        "tool_name": name or "tool",
        "args": args or {},
        "latency_ms": int(ms or 0),
        "ok": bool(ok),
        "error": error,
    }
    _post_analytics("/api/analytics/ingest", payload)


@contextmanager
def track_tool(
    session_id: str, turn_id: int | None, name: str, args: dict | None = None
) -> Iterator[None]:
    """
    Context manager to time and log analytics for a tool call.

    Args:
        session_id: Session identifier (falls back to ''"anonymous"'' if blank).
        turn_id: Optional associated turn id.
        name: Tool name (e.g., ''"quote_lookup"'').
        args: Optional dictionary of tool arguments to record.

    Yields:
        None. Use within a ''with'' block to automatically measure runtime.

    Raises:
        Re-raises any exception thrown inside the context after logging
        a failed tool ingest.
    """
    start = perf_counter()
    try:
        yield
        ms = int((perf_counter() - start) * 1000)
        log_tool(session_id, turn_id, name, args or {}, ms, ok=True, error=None)
    except Exception as e:
        ms = int((perf_counter() - start) * 1000)
        log_tool(session_id, turn_id, name, args or {}, ms, ok=False, error=str(e))
        raise
