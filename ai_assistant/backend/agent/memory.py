from __future__ import annotations
import atexit
import asyncio
import logging
import os
import uuid
from typing import Any, Callable


def _norm_config(config: dict | None) -> dict[str, Any]:
    """Normalize runnable config ensuring a checkpoint namespace is present.

    Ensures there is a 'configurable' sub-dict and that
    'configurable['checkpoint_ns']' exists (default empty string for
    single-tenant).

    Args:
        config: Optional runnable configuration mapping.

    Returns:
        A copy of the config with 'configurable.checkpoint_ns' ensured.
    """
    out = dict(config or {})
    conf = dict(out.get("configurable") or {})
    if "checkpoint_ns" not in conf:
        conf["checkpoint_ns"] = ""
    out["configurable"] = conf
    return out


# ---------- checkpoint builders / fixers ----------
def _to_checkpoint(state: dict | None) -> dict[str, Any]:
    """Build a versioned checkpoint mapping from a partial state.

    Produces a structure compatible with LangGraph and langchain‑postgres
    savers. If 'state' is None or empty, returns a minimal checkpoint with an
    empty 'channel_values' mapping.

    Args:
        state: Optional state object (e.g., '{"messages": [...]}').

    Returns:
        A canonical checkpoint dict including channel versions and metadata.
    """
    if not state:
        ch = {}
    elif "channel_values" in state and isinstance(state["channel_values"], dict):
        ch = state["channel_values"]
    else:
        ch = dict(state)

    def _int(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 1

    ch_versions = {k: 1 for k in ch.keys()}

    return {
        "id": str(uuid.uuid4()),
        "parent_id": None,
        "v": 4,
        "channel_values": ch,
        "channel_versions": {k: _int(v) for k, v in ch_versions.items()},
        "versions_seen": {k: 1 for k in ch_versions.keys()},
        "versions": {},
    }


def _coerce_checkpoint_types(chk: dict) -> dict[str, Any]:
    """Coerce legacy or malformed checkpoints into the expected shape.

    Accepts arbitrary input and returns a well‑formed checkpoint dictionary as
    expected by LangGraph savers, normalizing integer fields and defaulting
    missing maps from present channels.

    Args:
        chk: Candidate checkpoint mapping.

    Returns:
        A normalized checkpoint dictionary safe to persist.
    """
    if not isinstance(chk, dict):
        chk = {}

    v = chk.get("v", 4)
    try:
        v = int(v)
    except Exception:
        v = 4

    ch_vals = chk.get("channel_values")
    if not isinstance(ch_vals, dict):
        ch_vals = dict(chk) if isinstance(chk, dict) else {}
        for k in (
            "id",
            "parent_id",
            "v",
            "channel_values",
            "channel_versions",
            "versions_seen",
            "versions",
        ):
            ch_vals.pop(k, None)

    def _as_int_map(d: Any) -> dict[str, int]:
        out: dict[str, int] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    out[str(k)] = int(v)
                except Exception:
                    out[str(k)] = 1
        return out

    ch_vers = _as_int_map(chk.get("channel_versions"))
    if not ch_vers:
        ch_vers = {k: 1 for k in ch_vals.keys()}

    seen = _as_int_map(chk.get("versions_seen"))
    if not seen:
        seen = dict(ch_vers)

    fixed = {
        "id": chk.get("id") or str(uuid.uuid4()),
        "parent_id": chk.get("parent_id"),
        "v": v,
        "channel_values": ch_vals,
        "channel_versions": ch_vers,
        "versions_seen": seen,
        "versions": chk.get("versions") or {},
    }
    return fixed


def _dsn() -> str:
    """Resolve and normalize the Postgres DSN from environment variables.

    Reads, in order: 'PG_DSN', 'DATABASE_URL', 'BACKEND_DATABASE_URL'.
    If the URL is SQLAlchemy‑style (e.g., 'postgresql+psycopg://'), it is
    normalized to 'postgresql://' for psycopg compatibility.

    Returns:
        A psycopg‑compatible Postgres DSN string.

    Raises:
        RuntimeError: If no DSN/URL environment variable is set.
    """
    url = (
        os.environ.get("PG_DSN")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("BACKEND_DATABASE_URL")
    )
    if not url:
        raise RuntimeError("Set PG_DSN or DATABASE_URL or BACKEND_DATABASE_URL")
    return url.replace("postgresql+psycopg://", "postgresql://")


_CP_INNER = None
_CP_CTXMAN = None
_CP_KIND = None
_CP_RAW = None


try:
    from psycopg import OperationalError as _PsyOpErr
except Exception:
    try:
        from psycopg.errors import OperationalError as _PsyOpErr
    except Exception:
        _PsyOpErr = Exception


class _ResilientSaver:
    """Proxy around a saver that auto‑recovers from connection errors.

    The proxy calls a getter to fetch the current raw saver instance. If a
    psycopg 'OperationalError' occurs, it triggers a reset callback and retries
    the operation once.
    """

    def __init__(self, getter: Callable[[], Any], resetter: Callable[[], None]):
        self._get = getter
        self._reset = resetter

    # --- sync helpers ---
    def _call(self, name: str, *args: Any, **kwargs: Any):
        inner = self._get()
        fn = getattr(inner, name)
        try:
            return fn(*args, **kwargs)
        except _PsyOpErr:
            self._reset()
            inner = self._get()
            fn = getattr(inner, name)
            return fn(*args, **kwargs)

    # --- async helpers ---
    async def _acall(self, name: str, *args: Any, **kwargs: Any):
        inner = self._get()
        fn = getattr(inner, name)
        try:
            return await fn(*args, **kwargs)
        except _PsyOpErr:
            self._reset()
            inner = self._get()
            fn = getattr(inner, name)
            return await fn(*args, **kwargs)

    # Common saver APIs used by LangGraph variants
    def get(self, cfg: dict) -> Any:
        return self._call("get", cfg)

    def put(self, cfg: dict, chk: dict, md: dict, *nv: Any) -> Any:
        return self._call("put", cfg, chk, md, *nv)

    def get_tuple(self, cfg: dict) -> Any:
        return self._call("get_tuple", cfg)

    def create_tables(self) -> Any:
        return self._call("create_tables")

    def migrate(self) -> Any:
        return self._call("migrate")

    async def aget(self, cfg: dict) -> Any:
        return await self._acall("aget", cfg)

    async def aput(self, cfg: dict, chk: dict, md: dict, *nv: Any) -> Any:
        return await self._acall("aput", cfg, chk, md, *nv)

    def get_next_version(self, cfg, keys=None):
        inner = self._get()
        if hasattr(inner, "get_next_version"):
            try:
                return getattr(inner, "get_next_version")(cfg, keys)
            except _PsyOpErr:
                self._reset()
                inner = self._get()
                return getattr(inner, "get_next_version")(cfg, keys)

        def _extract(saved_obj):
            if saved_obj is None:
                return {}
            if isinstance(saved_obj, dict):
                chk = saved_obj.get("checkpoint") or {}
            else:
                chk = getattr(saved_obj, "checkpoint", None) or {}
            vers = chk.get("channel_versions") or {}
            out = {}
            for k, v in vers.items() if isinstance(vers, dict) else []:
                try:
                    out[str(k)] = int(v)
                except Exception:
                    out[str(k)] = 0
            return out

        try:
            saved = (
                self.get_tuple(cfg) if hasattr(inner, "get_tuple") else self.get(cfg)
            )
        except Exception:
            saved = None
        current = _extract(saved)

        # Build result: increment each requested key; if none specified, increment all known
        if keys:
            res = {}
            for k in keys:
                v = 0
                try:
                    v = int(current.get(k, 0))
                except Exception:
                    v = 0
                res[str(k)] = v + 1
            return res
        else:
            if not current:
                return {"messages": 1}
            return {k: int(v) + 1 for k, v in current.items()}


def _requested_kind() -> str:
    """Return the configured checkpointer kind ('postgres' or 'memory')."""
    raw = (os.getenv("AGENT_CHECKPOINTER") or "").strip().lower()
    if not raw:
        return "postgres"
    if raw in {"postgres", "pg", "postgresql"}:
        return "postgres"
    if raw in {"memory", "mem", "inmemory"}:
        return "memory"
    logging.getLogger(__name__).warning(
        "Unknown AGENT_CHECKPOINTER=%s; falling back to postgres", raw
    )
    return "postgres"


def _reset_cp() -> None:
    """Tear down and reinitialize the Postgres saver in place.

    Keeps the exposed '_CP_INNER' proxy stable while refreshing the underlying
    connection and applying migrations if available.
    """
    global _CP_CTXMAN, _CP_RAW, _CP_KIND
    try:
        if _CP_CTXMAN is not None:
            _CP_CTXMAN.__exit__(None, None, None)
    except Exception:
        pass

    dsn = _dsn()
    from langgraph.checkpoint.postgres import PostgresSaver as LGPostgresSaver  # type: ignore

    _CP_CTXMAN = LGPostgresSaver.from_conn_string(dsn)
    _CP_RAW = _CP_CTXMAN.__enter__()
    _CP_KIND = "postgres"
    try:
        if hasattr(_CP_RAW, "migrate"):
            _CP_RAW.migrate()
        elif hasattr(_CP_RAW, "create_tables"):
            _CP_RAW.create_tables()
        # langgraph checkpoint API changed; older ResilientSaver may miss 'put_writes'
        # which SyncPregelLoop expects. Patch a compatible method when absent.
        if not hasattr(_CP_RAW, "put_writes") and hasattr(_CP_RAW, "put"):
            _CP_RAW.put_writes = _CP_RAW.put  # type: ignore[attr-defined]
    except Exception:
        pass


def _init_cp() -> None:
    """Initialize the global checkpointer based on environment configuration."""
    global _CP_INNER, _CP_CTXMAN, _CP_KIND, _CP_RAW
    if _CP_INNER is not None:
        return

    kind = _requested_kind()

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        _CP_INNER = MemorySaver()
        _CP_CTXMAN = None
        _CP_KIND = "memory"
        return

    dsn = _dsn()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver as LGPostgresSaver  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing langgraph Postgres saver. Install: "
            "pip install 'langgraph>=0.2' 'langgraph-checkpoint-postgres>=0.1.8' 'psycopg[binary]'"
        ) from e

    _CP_CTXMAN = LGPostgresSaver.from_conn_string(dsn)
    _CP_RAW = _CP_CTXMAN.__enter__()
    _CP_KIND = "postgres"
    try:
        if hasattr(_CP_RAW, "migrate"):
            _CP_RAW.migrate()
        elif hasattr(_CP_RAW, "create_tables"):
            _CP_RAW.create_tables()
        # Patch compatibility: some ResilientSaver impls lack 'put_writes'; SyncPregelLoop expects it.
        if not hasattr(_CP_RAW, "put_writes") and hasattr(_CP_RAW, "put"):
            _CP_RAW.put_writes = _CP_RAW.put  # type: ignore[attr-defined]
    except Exception:
        pass

    # Expose a resilient proxy so callers can recover from closed connections
    _CP_INNER = _ResilientSaver(lambda: _CP_RAW, _reset_cp)

    atexit.register(lambda: _CP_CTXMAN and _CP_CTXMAN.__exit__(None, None, None))


def get_checkpointer() -> Any:
    """Return the initialized checkpointer (Postgres or in‑memory)."""
    _init_cp()
    return _CP_INNER


def get_checkpointer_kind() -> str:
    """Return a short string indicating the active checkpointer kind."""
    _init_cp()
    return _CP_KIND or "unknown"


# ---------- public get/put ----------
async def cp_get(config: dict) -> Any:
    """Fetch the current checkpoint for the given config.

    Prefers the async 'aget' API when available, otherwise runs the sync
    'get' on a thread. The return value may be store‑specific.

    Args:
        config: Runnable configuration used to scope the checkpoint.

    Returns:
        Either a raw store‑specific object, or a dict containing
        '{"checkpoint": ..., "metadata": ...}'.
    """
    inner = get_checkpointer()
    cfg = _norm_config(config)
    try:
        if hasattr(inner, "aget"):
            try:
                saved = await inner.aget(cfg)
            except NotImplementedError:
                saved = None
        else:
            loop = asyncio.get_running_loop()
            saved = await loop.run_in_executor(None, lambda: inner.get(cfg))
        if isinstance(saved, dict) and "checkpoint" in saved:
            return saved
        return saved
    except Exception:
        return {}


async def cp_put(
    config: dict,
    state: dict,
    *,
    metadata: dict | None = None,
    new_versions: dict | None = None,
) -> None:
    """Persist a checkpoint for the given config and state.

    Normalizes config and state to the expected shapes and handles saver API
    variations across versions. If 'new_versions' is omitted, all channels are
    bumped to version 1.

    Args:
        config: Runnable configuration used to scope the checkpoint.
        state: Partial state to persist (e.g., messages); converted to a
            canonical checkpoint.
        metadata: Optional metadata mapping; 'step' defaults to '-1' to
            allow LangGraph to increment.
        new_versions: Optional explicit channel version map.
    """
    inner = get_checkpointer()
    cfg = _norm_config(config)
    chk = _to_checkpoint(state)

    md: dict[str, Any] = {} if metadata is None else dict(metadata)
    if "step" not in md:
        md["step"] = -1

    # Compute the version map we want to end up with
    desired_nv: dict[str, int] = {}
    if new_versions is not None and isinstance(new_versions, dict):
        for k, v in new_versions.items():
            try:
                desired_nv[str(k)] = int(v)
            except Exception:
                desired_nv[str(k)] = 1
    else:
        desired_nv = {k: 1 for k in (chk.get("channel_values") or {}).keys()}
    chk["channel_versions"] = dict(desired_nv)

    if hasattr(inner, "aput"):
        try:
            await inner.aput(cfg, chk, md, desired_nv)
            return
        except TypeError:
            try:
                await inner.aput(cfg, chk, md)
                return
            except (TypeError, NotImplementedError):
                pass
        except NotImplementedError:
            pass

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, lambda: inner.put(cfg, chk, md, desired_nv))
        return
    except TypeError:
        await loop.run_in_executor(None, lambda: inner.put(cfg, chk, md))


def _extract_tuple(saved_obj: Any) -> tuple[dict, dict]:
    """Extract '(checkpoint, metadata)' from a saver 'get_tuple' result.

    Accepts dict‑like or dataclass‑like results.

    Args:
        saved_obj: Value returned by a saver.

    Returns:
        Tuple of checkpoint and metadata dictionaries (empty if unavailable).
    """
    if saved_obj is None:
        return {}, {}
    if isinstance(saved_obj, dict):
        return (saved_obj.get("checkpoint") or {}), (saved_obj.get("metadata") or {})
    chk = getattr(saved_obj, "checkpoint", None) or {}
    md = getattr(saved_obj, "metadata", None) or {}
    return chk, md


def _normalize_versions_map(d: Any) -> dict[str, int]:
    """Coerce an arbitrary mapping to 'dict[str, int]' with defaults of 1."""
    out: dict[str, int] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                out[str(k)] = 1
    return out


async def cp_repair_if_needed(config: dict) -> None:
    """Repair a malformed checkpoint in place if detected.

    Ensures required fields exist and that version maps contain integers,
    preserving metadata where available.

    Args:
        config: Runnable configuration used to scope the checkpoint.
    """
    inner = get_checkpointer()
    cfg = _norm_config(config)
    try:
        saved = inner.get_tuple(cfg) if hasattr(inner, "get_tuple") else None
    except Exception:
        saved = None

    def extract(saved_obj):
        if saved_obj is None:
            return {}, {}
        if isinstance(saved_obj, dict):
            return saved_obj.get("checkpoint") or {}, saved_obj.get("metadata") or {}
        return (
            getattr(saved_obj, "checkpoint", {}) or {},
            getattr(saved_obj, "metadata", {}) or {},
        )

    chk, md = extract(saved)
    ch_vals = chk.get("channel_values")
    if not isinstance(ch_vals, dict):
        ch_vals = {}
    if not ch_vals:
        ch_vals = {"messages": []}

    ch_vers = _normalize_versions_map(chk.get("channel_versions"))
    if not ch_vers:
        ch_vers = {k: 1 for k in ch_vals.keys()}

    seen = _normalize_versions_map(chk.get("versions_seen"))
    if not seen:
        seen = dict(ch_vers)
    versions = chk.get("versions")
    if isinstance(versions, dict):
        versions = _normalize_versions_map(versions)
    else:
        versions = {}
    step_raw = (md or {}).get("step", -1)
    try:
        step = int(step_raw)
    except Exception:
        step = -1

    fixed = {
        "id": chk.get("id") or str(uuid.uuid4()),
        "parent_id": chk.get("parent_id"),
        "v": int(chk.get("v", 4)) if str(chk.get("v", 4)).isdigit() else 4,
        "channel_values": ch_vals,
        "channel_versions": ch_vers,
        "versions_seen": seen,
        "versions": versions,
    }
    needs_fix = fixed != chk

    if needs_fix or step_raw != step:
        md = {**(md or {}), "step": step}
        if hasattr(inner, "aput"):
            try:
                await inner.aput(cfg, fixed, md, ch_vers)
                return
            except Exception:
                pass
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: inner.put(cfg, fixed, md, ch_vers))
        except TypeError:
            await loop.run_in_executor(None, lambda: inner.put(cfg, fixed))
