from __future__ import annotations
import asyncio
import uuid
import logging
from typing import Any
from agent.memory import get_checkpointer, _norm_config, cp_put

log = logging.getLogger(__name__)


def _as_int(v: Any) -> int:
    """Coerce a value to 'int', defaulting to '1' on failure.

    Args:
        v: Any value.

    Returns:
        Integer representation or '1' if conversion fails.
    """
    try:
        return int(v)
    except Exception:
        return 1


def _int_map(d: Any) -> dict[str, int]:
    """Coerce a mapping's values to 'int'.

    Args:
        d: Source mapping or other value.

    Returns:
        'dict[str, int]' with integer values. Returns an empty dict for
        non‑mapping inputs.
    """
    out: dict[str, int] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out[str(k)] = _as_int(v)
    return out


def _has_non_ints(d: Any) -> bool:
    """Return True if any non‑integer exists in a (nested) mapping."""
    if not isinstance(d, dict):
        return False
    for v in d.values():
        if isinstance(v, dict):
            if _has_non_ints(v):
                return True
        elif not isinstance(v, int):
            return True
    return False


async def ensure_checkpoint_sane(config: dict) -> dict:
    """Normalize and repair a checkpoint in place if needed.

    Ensures the underlying saver exposes a checkpoint with the fields LangGraph
    expects, coercing version maps to integers and setting defaults when
    missing. Updates both inner and store‑specific outer version maps.

    Args:
        config: Runnable configuration used to scope the checkpoint.

    Returns:
        The repaired checkpoint dictionary as persisted by the saver.
    """
    inner = get_checkpointer()
    cfg = _norm_config(config)
    saved: Any = None
    try:
        saved = inner.get_tuple(cfg) if hasattr(inner, "get_tuple") else inner.get(cfg)
    except Exception:
        saved = None

    chk: dict = {}
    md: dict = {}
    outer_versions_raw = None
    if saved is None:
        pass
    elif isinstance(saved, dict):
        chk = saved.get("checkpoint") or {}
        md = saved.get("metadata") or {}
        outer_versions_raw = saved.get("versions")
    else:
        chk = getattr(saved, "checkpoint", {}) or {}
        md = getattr(saved, "metadata", {}) or {}
        outer_versions_raw = getattr(saved, "versions", None)
    ch_vals = chk.get("channel_values")
    if not isinstance(ch_vals, dict) or not ch_vals:
        ch_vals = {"messages": []}

    # --- channel_versions (dict[str, int]) one counter per channel
    ch_vers = _int_map(chk.get("channel_versions"))
    if not ch_vers:
        ch_vers = {k: 1 for k in ch_vals.keys()} or {"messages": 1}

    # inner versions is optional; if present, must be dict[str, int]
    inner_versions = _int_map(chk.get("versions"))
    if not inner_versions or _has_non_ints(inner_versions):
        inner_versions = dict(ch_vers)

    # --- top-level previous_versions (what LG compares against)
    outer_versions_seen = _int_map(outer_versions_raw)
    effective_outer_versions = (
        dict(outer_versions_seen) if outer_versions_seen else dict(inner_versions)
    )
    # if store's value is missing or invalid, fallback to inner versions
    if not effective_outer_versions or _has_non_ints(effective_outer_versions):
        effective_outer_versions = dict(ch_vers)

    # --- versions_seen (flatten to {} so LG refills with nested structure)
    seen = {}

    # --- v: checkpoint schema version, default to 4 if missing/invalid
    try:
        vint = int(chk.get("v", 4))
    except Exception:
        vint = 4

    # ---- Fixup and decide whether to write back
    fixed = {
        "id": chk.get("id") or str(uuid.uuid4()),
        "parent_id": chk.get("parent_id"),
        "v": vint,
        "channel_values": ch_vals,
        "channel_versions": ch_vers,
        "versions": inner_versions,
        "versions_seen": seen,
    }
    if not isinstance(md, dict):
        md = {}
    try:
        md["step"] = int(md.get("step", -1))
    except Exception:
        md["step"] = -1
    dirty = (
        fixed != chk
        or _has_non_ints(chk.get("channel_versions"))
        or _has_non_ints(chk.get("versions"))
        or not isinstance(chk.get("versions_seen"), dict)
        or _has_non_ints(chk.get("versions_seen"))
        or _has_non_ints(outer_versions_raw)
        or _int_map(outer_versions_raw) != effective_outer_versions
    )

    if dirty:
        try:
            await cp_put(cfg, fixed, metadata=md, new_versions=effective_outer_versions)
        except Exception:
            try:
                if hasattr(inner, "aput"):
                    await inner.aput(
                        cfg, fixed, md, new_versions=effective_outer_versions
                    )
                else:
                    loop = asyncio.get_running_loop()
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda: inner.put(cfg, fixed, md, effective_outer_versions),
                        )
                    except TypeError:
                        await loop.run_in_executor(None, lambda: inner.put(cfg, fixed))  # type: ignore
            except Exception:
                pass

        # Re-read to reflect storage
        try:
            reread = (
                inner.get_tuple(cfg) if hasattr(inner, "get_tuple") else inner.get(cfg)
            )
            if isinstance(reread, dict):
                fixed = reread.get("checkpoint") or fixed
            else:
                fixed = getattr(reread, "checkpoint", None) or fixed
        except Exception:
            pass
    try:

        def _types(d):
            return (
                {k: type(v).__name__ for k, v in (d or {}).items()}
                if isinstance(d, dict)
                else str(type(d))
            )

        reread_outer = None
        try:
            reread_saved = (
                inner.get_tuple(cfg) if hasattr(inner, "get_tuple") else inner.get(cfg)
            )
            if isinstance(reread_saved, dict):
                reread_outer = reread_saved.get("versions")
            else:
                reread_outer = getattr(reread_saved, "versions", None)
        except Exception:
            pass
        log.warning(
            "Checkpoint sanity for %s: ch_versions=%s (types=%s), inner.versions=%s (types=%s), OUTER versions=%s (types=%s), versions_seen=%s",
            cfg,
            fixed.get("channel_versions"),
            _types(fixed.get("channel_versions")),
            fixed.get("versions"),
            _types(fixed.get("versions")),
            reread_outer,
            (
                _types(reread_outer)
                if isinstance(reread_outer, dict)
                else str(type(reread_outer))
            ),
            fixed.get("versions_seen"),
        )
    except Exception:
        pass

    return fixed


def _msg_type(m: Any) -> str:
    """Extract a message "type" from LangChain messages or dicts.

    Args:
        m: Message object or mapping.

    Returns:
        The type string (e.g., '"human"', '"ai"', '"tool"') or empty
        string if unavailable.
    """
    try:
        t = getattr(m, "type", None)
        if t:
            return str(t)
    except Exception:
        pass
    if isinstance(m, dict):
        return str(m.get("type") or m.get("role") or "")
    return ""


def _has_tool_calls(m: Any) -> bool:
    """Return True if the message carries tool call metadata."""
    try:
        tc = getattr(m, "tool_calls", None)
        if tc:
            return True
    except Exception:
        pass
    if isinstance(m, dict):
        if m.get("tool_calls"):
            return True
        ak = m.get("additional_kwargs") or {}
        if isinstance(ak, dict) and ak.get("tool_calls"):
            return True
    return False


def _safe_tail(messages: list[Any], limit: int) -> list[Any]:
    """Return a tail slice that keeps tool call pairs intact.

    Keeps the last 'limit' human/ai messages without splitting an assistant
    tool call from its following tool result.

    Args:
        messages: Full message history list.
        limit: Number of human/ai messages to preserve.

    Returns:
        The pruned list containing only the tail segment.
    """
    if not isinstance(messages, list) or not messages:
        return messages or []

    human_ai = 0
    cut = 0
    for i in range(len(messages) - 1, -1, -1):
        mt = _msg_type(messages[i])
        if mt in ("human", "ai"):
            human_ai += 1
        if human_ai > limit:
            cut = i + 1
            break

    if human_ai <= limit:
        return messages
    while cut < len(messages) and _msg_type(messages[cut]) == "tool":
        if (
            cut - 1 >= 0
            and _msg_type(messages[cut - 1]) == "ai"
            and _has_tool_calls(messages[cut - 1])
        ):
            cut -= 1
            break
        cut += 1
    if (
        0 < cut < len(messages)
        and _msg_type(messages[cut - 1]) == "ai"
        and _has_tool_calls(messages[cut - 1])
    ):
        cut -= 1

    return messages[cut:]


async def prune_checkpoint_if_needed(
    config: dict, human_ai_limit: int | None = None
) -> dict | None:
    """Prune the messages channel to the last N human/ai turns.

    Preserves assistant tool call and immediate tool message pairs. If pruning
    occurs, writes an updated checkpoint with a bumped 'messages' channel
    version and returns a minimal echo containing updated values.

    Args:
        config: Runnable configuration used to scope the checkpoint.
        human_ai_limit: Optional override for the number of human/ai turns to
            keep; defaults to 'PRUNE_HUMAN_AI_LIMIT' env var or 200.

    Returns:
        A dict with 'channel_values' and 'channel_versions' if pruning was
        performed, otherwise 'None'.
    """
    try:
        import os

        limit = human_ai_limit
        if limit is None:
            try:
                limit = int(os.getenv("PRUNE_HUMAN_AI_LIMIT", "200") or 200)
            except Exception:
                limit = 200

        inner = get_checkpointer()
        cfg = _norm_config(config)
        try:
            saved = (
                inner.get_tuple(cfg) if hasattr(inner, "get_tuple") else inner.get(cfg)
            )
        except Exception:
            saved = None

        chk = {}
        if saved is None:
            return None
        if isinstance(saved, dict):
            chk = saved.get("checkpoint") or {}
        else:
            chk = getattr(saved, "checkpoint", None) or {}

        ch_vals = chk.get("channel_values") or {}
        if not isinstance(ch_vals, dict):
            return None

        msgs = ch_vals.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return None
        ha = sum(1 for m in msgs if _msg_type(m) in ("human", "ai"))
        if ha <= limit:
            return None

        new_msgs = _safe_tail(msgs, limit)
        if new_msgs == msgs:
            return None
        new_state = dict(ch_vals)
        new_state["messages"] = new_msgs
        ch_vers = chk.get("channel_versions") or {}
        try:
            current = int(ch_vers.get("messages", 1))
        except Exception:
            current = 1
        desired_nv = {
            k: (int(v) if isinstance(v, int) or str(v).isdigit() else 1)
            for k, v in ch_vers.items()
        }
        desired_nv["messages"] = current + 1

        await cp_put(cfg, new_state, metadata={"step": -1}, new_versions=desired_nv)
        return {"channel_values": new_state, "channel_versions": desired_nv}
    except Exception:
        return None
