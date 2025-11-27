from __future__ import annotations
import os
import io
import re
import json
import time
import inspect
import asyncio
import csv
from dataclasses import dataclass
from contextlib import asynccontextmanager, suppress
from typing import Any, Optional, Dict, Tuple
import httpx
from loguru import logger
from sqlalchemy.orm import Session as SASession
from fastapi import (
    FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect, Response
)
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import AsyncCallbackHandler

from .core.db import (
    get_db, init_db, get_or_create_session, list_messages, append_message, touch_session, engine
)

from .llm.agent_tools import TOOLS
from .llm.rag import (
    build_functions_agent,
    make_answer_prompt,
    reindex,
    _strict_answer_with_tagged_context,
    docs_to_sources,
)
from .utils.citations import (
    book_source_from_filename,
    normalize_source,
    dedupe_sources,
    tools_trace_to_sources,
    _filter_sources_to_text,
    _had_rag_or_tools,
)
from .utils.utils import (
    to_lc_messages,
    _TokenBucket,
    _RedisTokenBucket,
    _require_admin,
)
from .utils.text import (
    _unsplit_link_urls,
    _postprocess_markdown,
    _unique_safe_names,
    _coerce_strings,
)
from .utils.tools_utils import (
    _tool_schema,
    _call_tool,
    _looks_toolable,
    _collect_tool_markdown,
    _extract_tool_events,
    _is_finance_query,
)
from .utils.rag_utils import (
    rag_filter_confident,
)
from .models.models import ChatInput
from .analytics.tokenlog import with_token_log, on_llm_end
from .routers.sessions import router as sessions_router
from .routers.price import price_router
from .routers.td import td_router
from .routers.calcs import calc_api_router
from .mcp_ext.langchain import build_langchain_tools
from .finnhub_client import _FinnhubClient
from .core.settings import settings
from .core.auth import session_tokens, _parse_bearer
from itsdangerous import BadSignature, SignatureExpired
from openai import OpenAI


# ----------- GLOBAL VARIABLES -----------
APP_NAME: str = os.getenv("APP_NAME", "finance-chatbot-api")
def _compute_allowed_origins() -> list[str]:
    """
    Build a strict CORS allowlist from configured origins.
    """
    origins: list[str] = []

    def _add(val: Any) -> None:
        if not val:
            return
        s = str(val).strip()
        if s and s not in origins:
            origins.append(s)

    _add(getattr(settings, "frontend_origin", None))
    for o in getattr(settings, "cors_allow_origins", []) or []:
        _add(o)
    return origins or ["http://localhost:3000"]


ALLOWED_ORIGINS = _compute_allowed_origins()
MAX_REQUEST_TOKENS: int = int(os.getenv("MAX_REQUEST_TOKENS", "8192"))
RATE_LIMIT_RPS: float = settings.rate_limit_rps
RATE_LIMIT_BURST: float = float(os.getenv("RATE_LIMIT_BURST", "10"))
retriever = None              
make_answer_prompt_fn = None  
rag_agent_chain = None 
rag_agent_chain_stream = None 
llm = None 

MIN_SCORE: float = float(getattr(settings, "rag_min_score", 0.25))
MIN_DOCS: int = 1

SAFE = re.compile(r'^[a-zA-Z0-9_-]+$')
OPENAI_SAFE = re.compile(r'^[a-zA-Z0-9_-]+$')

STOCK_TOOL_NAMES: set[str] = {
    "get_live_price", "search_symbol", "get_company_profile", "get_candles",
    "get_recommendation_trends", "get_company_news",
}

FINNHUB_SOURCE: dict[str, str] = {
    "kind": "tool",
    "type": "tool",
    "id": "tool:finnhub",
    "title": "Finnhub",
    "display": "Finnhub",
    "href": "https://finnhub.io/",
    "url": "https://finnhub.io/",
}

WORLD_BANK_SOURCE: dict[str, str] = {
    "kind": "tool",
    "type": "tool",
    "id": "tool:world_bank",
    "title": "World Bank",
    "display": "World Bank (MCP)",
    "href": "https://data.worldbank.org/",
    "url": "https://data.worldbank.org/",
}

SESSION_TOKEN_REQUIRED: bool = (
    settings.session_token_required or bool(settings.session_token_secret)
)
ADMIN_ROUTES_ENABLED: bool = bool(os.getenv("ADMIN_KEY"))
_moderation_client = None

# ----------- SETUP HELPERS ------------
def _build_rate_limiter():
    """
    Choose Redis-backed limiter when REDIS_URL is set; fall back to in-memory bucket otherwise.
    """
    redis_url = os.getenv("REDIS_URL")
    ns = os.getenv("RATE_LIMIT_NAMESPACE", "rate")
    if redis_url:
        try:
            logger.info("Using Redis rate limiter at %s (ns=%s)", redis_url, ns)
            return _RedisTokenBucket(redis_url, RATE_LIMIT_RPS, RATE_LIMIT_BURST, namespace=ns)
        except Exception as e:
            logger.warning("Redis rate limiter unavailable (%s); falling back to in-memory bucket.", e)
    return _TokenBucket(RATE_LIMIT_RPS, RATE_LIMIT_BURST)


_rate_limiter = _build_rate_limiter()


def _require_session_token(
    request: Request, session_id: Optional[str] = None
) -> Optional[str]:
    """
    Enforce signed session tokens for chat/export endpoints.

    - If a secret is configured (or SESSION_TOKEN_REQUIRED is set), a Bearer token
      is required and must match the session id (when provided).
    - Returns the session id from the token for use as the canonical session id.
    """
    if not session_tokens.enabled:
        if SESSION_TOKEN_REQUIRED:
            raise HTTPException(
                status_code=500,
                detail="Session tokens required but no SESSION_TOKEN_SECRET configured.",
            )
        return None

    token = _parse_bearer(request.headers.get("Authorization"))
    if not token:
        raise HTTPException(status_code=401, detail="Missing session token.")

    try:
        token_sid = session_tokens.verify(token)
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Session token expired.")
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid session token.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if session_id and token_sid != session_id:
        raise HTTPException(
            status_code=403, detail="Session token does not match session id."
        )
    return token_sid


def _get_retriever_lazy(force_rebuild: bool = False):
    """
    Ensure a retriever exists; rebuild on demand if startup failed.

    Cloud Run can occasionally hiccup during startup (e.g., transient Qdrant/OpenAI
    failures). Without this guard, a single failure would leave 'retriever' as None
    for the life of the container and RAG would never run.
    """
    global retriever
    if retriever is None or force_rebuild:
        try:
            from .llm.rag import get_retriever as _gr
            retriever = _gr(force_rebuild=force_rebuild)
        except Exception as e:
            logger.warning("Retriever unavailable: {}", e)
            retriever = None
    return retriever


def _get_secret_value(val: Any) -> Optional[str]:
    """
    Safely unwrap a SecretStr or return a plain string representation.
    """
    if val is None:
        return None
    try:
        return val.get_secret_value()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if isinstance(val, str):
            return val
        if val:
            return str(val)
    except Exception:
        pass
    return None


def _require_client_auth(
    request: Request, session_id: Optional[str] = None
) -> Optional[str]:
    """
    Enforce either a session token (preferred) or an API key for client calls.

    - If session tokens are enabled, defer to '_require_session_token'.
    - Else if API_KEY is configured, require it via 'X-API-Key' or Bearer.
    - If neither is configured, fail fast to avoid unauthenticated access.
    """
    if session_tokens.enabled:
        return _require_session_token(request, session_id)

    api_key = _get_secret_value(getattr(settings, "api_key", None))
    if api_key:
        provided = (
            request.headers.get("X-API-Key")
            or request.headers.get("X-Api-Key")
            or _parse_bearer(request.headers.get("Authorization") or "")
        )
        if not provided:
            raise HTTPException(status_code=401, detail="Missing API key.")
        if provided != api_key:
            raise HTTPException(status_code=403, detail="Invalid API key.")
        return session_id

    raise HTTPException(
        status_code=500,
        detail="Authentication not configured. Set SESSION_TOKEN_SECRET or API_KEY.",
    )


def _moderate_text(text: str) -> Optional[dict]:
    """
    Run OpenAI moderation (if enabled) and return the raw result dict.

    Returns:
        None if moderation disabled/unavailable; otherwise a dict with:
            {"flagged": bool, "categories": {...}}
    """
    if not settings.openai_moderation_enabled:
        return None
    if not settings.openai_api_key:
        logger.warning("Moderation enabled but OPENAI_API_KEY is missing.")
        return None
    try:
        global _moderation_client
        if _moderation_client is None:
            _moderation_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        resp = _moderation_client.moderations.create(
            model=settings.openai_moderation_model,
            input=text or "",
        )
        res0 = resp.results[0] if resp and resp.results else None
        if not res0:
            return None
        return {"flagged": bool(res0.flagged), "categories": res0.categories or {}}
    except Exception as e:
        logger.warning("Moderation check failed: %s", e)
        return None


# ---------- Settings / FINNHUB ------------------------------------------------
try:
    from .core.settings import settings
except Exception:
    @dataclass
    class _Settings:
        openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        finnhub_api_key: str | None = os.getenv("FINNHUB_API_KEY")
    settings = _Settings()

FINNHUB = _FinnhubClient(_get_secret_value(settings.finnhub_api_key))

# ---------------- FastAPI app with lifespan ----------------
def _openai_safe_name(name: str) -> str:
    """
    Convert a string into a strict OpenAI-safe function/tool name.

    Rules:
        - Allowed characters: [a-zA-Z0-9_-]
        - All other characters are replaced with underscores.
        - Leading/trailing underscores are trimmed.
        - If the result is empty, falls back to "tool".

    Args:
        name: Original name string.

    Returns:
        Sanitized name string safe for OpenAI APIs.
    """
    n = re.sub(r'[^a-zA-Z0-9_-]+', '_', (name or '').strip())
    n = n.strip('_')
    return n or 'tool'

def _sanitize_tool_objects_for_openai(tools: list) -> list:
    """
    Normalize a list of tool objects so all names are OpenAI-safe and unique.

    Behavior:
        - Each tool's '.name' is sanitized with '_openai_safe_name'.
        - If duplicates arise, numeric suffixes (_2, _3, …) are appended.
        - If renamed, original name is preserved in the '.description' as '[alias: <orig>]'.

    Args:
        tools: List of LangChain-compatible tool objects.

    Returns:
        List of sanitized tool objects with unique names.
    """
    seen: set[str] = set()
    for t in tools:
        orig = getattr(t, "name", "") or ""
        base = _openai_safe_name(orig)
        new = base
        i = 2
        while not OPENAI_SAFE.match(new) or new in seen:
            new = f"{base}_{i}"
            i += 1
        if new != orig:
            desc = (getattr(t, "description", "") or "").strip()
            setattr(t, "description", f"[alias: {orig}] {desc}".strip())
            setattr(t, "name", new)
        seen.add(new)
    return tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifespan manager.

    Startup phase:
        - Initialize database schema.
        - Create LLM clients (stream + non-stream).
        - Build retriever (if available).
        - Load MCP tools (World Bank).
        - Initialize LangChain agent executors.

    Shutdown phase:
        - Gracefully close FINNHUB client.

    Args:
        app: FastAPI application instance.

    Yields:
        None. Used by FastAPI for lifespan context management.
    """
    global retriever, make_answer_prompt_fn, rag_agent_chain, rag_agent_chain_stream, llm

    # --- DB init ---
    init_db()

    # --- LLM clients ---
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        timeout=90,
        streaming=False,
    )
    llm_stream = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        timeout=90,
        streaming=True,
    )

    # --- Retriever (optional) ---
    retriever = _get_retriever_lazy(force_rebuild=True)

    # --- MCP tools (World Bank) ---
    extra_tools: list = []

    async def _load_mcp_tools_with_retry(retries: int = 3, delay: float = 1.5) -> list:
        """
        Try to discover and build MCP tools a few times (MCP server may take a moment to come up).
        """
        from loguru import logger
        from back_app.mcp_ext.langchain import build_langchain_tools

        last_err = None
        for i in range(retries):
            try:
                maybe = build_langchain_tools("world_bank")
                tools = await maybe if inspect.isawaitable(maybe) else maybe
                logger.info("Loaded {} MCP tools from 'world_bank'.", len(tools or []))
                return tools or []
            except Exception as e:
                last_err = e
                logger.warning("MCP tool load attempt {}/{} failed: {}", i + 1, retries, e)
                await asyncio.sleep(delay)
        logger.error("Failed to load MCP tools after {} attempts: {!r}", retries, last_err)
        return []
    try:
        mcp_enabled = (
            os.getenv("MCP_ENABLE", "false").lower() == "true"
            or os.getenv("MCP_PROXY_ENABLE", "false").lower() == "true"
        )
        if mcp_enabled:
            mcp_tools = await _load_mcp_tools_with_retry()
            mcp_tools = _sanitize_tool_objects_for_openai(list(mcp_tools or []))
            extra_tools.extend(mcp_tools)
            logger.info(
                "MCP(World Bank) tools loaded: {}",
                [getattr(t, "name", "<unnamed>") for t in extra_tools],
            )
        else:
            logger.info("MCP disabled via env; skipping MCP tool load.")
    except Exception as e:
        logger.warning("MCP world_bank disabled: {}", e)


    # --- Agent executors ---
    builtins = list(TOOLS)  # copy
    ALL_TOOLS = _sanitize_tool_objects_for_openai([*builtins, *extra_tools])
    ALL_TOOL_NAMES = [getattr(t, "name", "<unnamed>") for t in ALL_TOOLS]
    logger.info("Agent tool registry initialized (n=%d): %s", len(ALL_TOOL_NAMES), ALL_TOOL_NAMES)
    debug_flag = (os.getenv("AGENT_DEBUG", "1") == "1")
    rag_agent_chain = build_functions_agent(llm, ALL_TOOLS, debug=debug_flag)
    rag_agent_chain_stream = build_functions_agent(llm_stream, ALL_TOOLS, debug=debug_flag)
    logger.info("Agent executors ready (streaming=%s, non-streaming=%s)", True, True)
    make_answer_prompt_fn = make_answer_prompt

    try:
        yield
    finally:
        try:
            if hasattr(FINNHUB, "aclose"):
                await FINNHUB.aclose()
        except Exception:
            pass


# ---------- FastAPI Application ----------

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------- Middleware: Rate Limiting ----------

@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    """
    Global rate-limiting middleware.

    Uses a token bucket to limit requests per client (session or IP).
    - Each incoming HTTP request "costs" one token.
    - If the bucket has no tokens, a 429 Too Many Requests is returned.
    - Otherwise, request is passed down the stack.

    Args:
        request: FastAPI request object.
        call_next: Callable to forward request to next handler.

    Returns:
        JSONResponse(429) if over limit, else the downstream response.
    """
    ip: str = request.client.host if request.client else "unknown"
    # Prefer a session identifier when present to avoid rate-sharing behind NATs.
    hdr_sid = (request.headers.get("X-Session-Id") or "").strip()
    auth_sid = _parse_bearer(request.headers.get("Authorization") or "")
    key = hdr_sid or auth_sid or f"ip:{ip}"

    if not _rate_limiter.allow(key, tokens=1.0):
        return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})

    return await call_next(request)


# ---------- Include Routers ----------

app.include_router(sessions_router, prefix="/api")
app.include_router(price_router, tags=["price"])
app.include_router(td_router, tags=["td"])
app.include_router(calc_api_router, tags=["calc"])
# Analytics routes disabled (no-op)

if (os.getenv("MCP_ENABLE", "false").lower() == "true" or os.getenv("MCP_PROXY_ENABLE", "false").lower() == "true"):
    from .mcp_ext.router import router as mcp_router
    app.include_router(mcp_router, prefix="/api/mcp")

# ---------- Exception Handlers ----------

@app.exception_handler(HTTPException)
async def _http_exc_handler(_req: Request, exc: HTTPException) -> JSONResponse:
    """
    Unified handler for HTTPException.

    Converts raised HTTPException into JSON error payload:
        {"detail": "<message>"}

    Ensures all error responses look consistent.

    Args:
        _req: Request object (unused).
        exc: Raised HTTPException.

    Returns:
        JSONResponse with the same status code + detail.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def _uncaught_exc_handler(_req: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort handler for *all* unhandled exceptions.

    Behavior:
    - Logs the full stack trace (via loguru).
    - Returns a generic 500 Internal Server Error response.
    - Prevents leaking stack traces / internals to clients.

    Args:
        _req: Request object (unused).
        exc: Exception object.

    Returns:
        JSONResponse(500, {"detail": "Internal Server Error"})
    """
    logger.exception("Unhandled error: %r", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

# ----------------------------
# Admin-only maintenance route
# ----------------------------


if ADMIN_ROUTES_ENABLED:
    @app.post("/api/admin/rag/reindex")
    def admin_reindex(request: Request) -> Dict[str, bool]:
        """
        Trigger a RAG reindex (admin only).

        Security:
            - Uses '_require_admin(request)' which checks 'X-Admin-Key' header
              against the 'ADMIN_KEY' environment variable.
            - If the key is invalid, raises 403 Forbidden.

        Behavior:
            - Calls 'reindex()' to refresh the knowledge base.
            - Returns {"ok": True} on success.

        Args:
            request: FastAPI request (used to check admin auth).

        Returns:
            Dict confirming success.
        """
        _require_admin(request)
        reindex()
        return {"ok": True}
else:
    logger.info("Admin routes disabled (ADMIN_KEY not set); skipping /api/admin/rag/reindex")




# ----------------------------
# MCP JSON-RPC over WebSocket
# ----------------------------

@app.websocket("/mcp")
async def mcp_ws(ws: WebSocket) -> None:
    """
    Minimal JSON-RPC 2.0 server over WebSocket for MCP tools.

    Supported methods:
        - "ping"       → returns "pong"
        - "tools/list" → lists registered tools + schemas
        - "tools/call" → executes a tool with arguments

    Args:
        ws: FastAPI WebSocket connection.
    """
    await ws.accept()
    tools, tmap = _collect_tools()
    await ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 0, "result": {"ready": True}}))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                req = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"}
                }))
                continue

            mid: Optional[int] = req.get("id")
            method: Optional[str] = req.get("method")
            params: Dict[str, Any] = req.get("params") or {}

            try:
                if method == "ping":
                    await ws.send_text(json.dumps({
                        "jsonrpc": "2.0", "id": mid, "result": "pong"
                    }))

                elif method == "tools/list":
                    out: list[Dict[str, Any]] = []
                    for t in tools:
                        out.append({
                            "name": t.name,
                            "description": getattr(t, "description", "") or "",
                            "parameters": _tool_schema(t), 
                        })
                    await ws.send_text(json.dumps({
                        "jsonrpc": "2.0", "id": mid, "result": {"tools": out}
                    }))

                elif method == "tools/call":
                    name = params.get("name")
                    arguments = params.get("arguments") or {}

                    if name not in tmap:
                        await ws.send_text(json.dumps({
                            "jsonrpc": "2.0",
                            "id": mid,
                            "error": {"code": -32602, "message": f"Unknown tool: {name}"}
                        }))
                        continue

                    tool = tmap[name]
                    try:
                        result = await _call_tool(tool, arguments)
                        try:
                            json.dumps(result)
                            payload = result
                        except TypeError:
                            payload = {"markdown": str(result)}

                        await ws.send_text(json.dumps({
                            "jsonrpc": "2.0", "id": mid, "result": payload
                        }))
                    except Exception as e:
                        logger.exception("tool error %s: %r", name, e)
                        await ws.send_text(json.dumps({
                            "jsonrpc": "2.0",
                            "id": mid,
                            "error": {"code": -32000,
                                      "message": f"tool error: {type(e).__name__}: {e}"}
                        }))

                else:
                    await ws.send_text(json.dumps({
                        "jsonrpc": "2.0", "id": mid,
                        "error": {"code": -32601, "message": "Method not found"}
                    }))

            except Exception as e:
                logger.exception("/mcp dispatch error: %r", e)
                await ws.send_text(json.dumps({
                    "jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32001, "message": "Internal error"}
                }))

    except WebSocketDisconnect:
        return


@app.delete("/mcp")
async def mcp_delete() -> Response:
    """
    Silences clients that send DELETE /mcp on shutdown.

    Some clients auto-try to clean up WebSocket routes.
    To avoid noisy logs, we return a '204 No Content'.

    Returns:
        Response with 204 status.
    """
    return Response(status_code=204)

# ----------------------------
# Session helper
# ----------------------------

def _ensure_session(db: SASession, sid: Optional[str], title: str = "Chat") -> str:
    """
    Ensure that a valid session exists (create if missing).

    Args:
        db: SQLAlchemy session object.
        sid: Existing session ID (can be None, empty, or invalid).
        title: Default session title to assign when creating a new one.

    Returns:
        A valid session ID string.
    """
    return get_or_create_session(db, sid, title)


# ----------------------------
# Helper functions
# ----------------------------

def _has_useful_docs(q: str) -> tuple[bool, float]:
    """
    Check whether the retriever has relevant documents for a query.

    Logic:
        - If retriever is unavailable, returns (False, 0.0).
        - Fetches docs from retriever.
        - Computes max score via '_doc_max_score'.
        - Returns (True, score) if at least one doc meets threshold.

    Args:
        q: Raw query string.

    Returns:
        (has_docs, max_score)
            has_docs: True if retriever returned at least one confident doc.
            max_score: Highest similarity score found.
    """
    try:
        r = _get_retriever_lazy()
        if not q or r is None:
            return (False, 0.0)

        if hasattr(r, "invoke"):
            raw = r.invoke(q, config=None)
        elif hasattr(r, "aget_relevant_documents"):
            raw = asyncio.get_event_loop().run_until_complete(
                r.aget_relevant_documents(q)
            )
        elif hasattr(r, "get_relevant_documents"):
            raw = r.get_relevant_documents(q)
        else:
            raw = []

        if isinstance(raw, dict):
            docs = raw.get("documents") or raw.get("docs") or []
        else:
            docs = raw or []

        from .llm.rag import _doc_max_score
        s = _doc_max_score(docs)
        return ((len(docs) > 0 and s >= 0.20), s)
    except Exception:
        return (False, 0.0)


def _collect_tools() -> Tuple[list, Dict[str, Any]]:
    """
    Collect available tools (LangChain + MCP).

    Behavior:
        - Starts with built-in 'TOOLS'.
        - If MCP tools are available, appends them.
        - Ensures unique + safe names via '_unique_safe_names'.
        - Builds a mapping {name → tool} for fast lookup.

    Returns:
        (tools, tmap)
            tools: list of tool objects.
            tmap: dict mapping tool name → tool object.
    """
    tools = list(TOOLS)

    if callable(build_langchain_tools):
        try:
            mcp_tools = build_langchain_tools()
            tools.extend(mcp_tools or [])
        except Exception as e:
            logger.warning(f"build_langchain_tools failed: {e!r}")

    tools = _unique_safe_names(tools)
    tools = _sanitize_tool_objects_for_openai(list(tools))
    tmap = {t.name: t for t in tools}
    return tools, tmap

# ----------------------------
# Token estimation helpers
# ----------------------------

def _approx_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """
    Approximate USD cost for a given model + token usage.

    - Looks up pricing from settings.PRICING (per-1k tokens).
    - Returns None if model pricing unavailable.

    Args:
        model: Model name string.
        input_tokens: Count of input (prompt) tokens.
        output_tokens: Count of output (completion) tokens.

    Returns:
        Approximate cost in USD, or None if model not priced.
    """
    try:
        from .core.settings import PRICING
        mp = PRICING.get(model)
        if not mp:
            return None
        return (
            (input_tokens / 1000.0) * mp.input_per_1k +
            (output_tokens / 1000.0) * mp.output_per_1k
        )
    except Exception:
        return None


def _approx_count_tokens(
    model: str,
    prompt_messages: list[dict],
    completion_text: str
) -> tuple[int, int]:
    """
    Fallback token counter (rough estimate, model-agnostic).

    - Uses 'tiktoken' if available.
    - Builds a plain text representation of messages with roles.
    - Encodes and counts input + output separately.
    - If fails, returns (0, 0).

    Args:
        model: Model name string (used for encoding selection).
        prompt_messages: List of messages (dict with role/content).
        completion_text: Assistant's reply text.

    Returns:
        (input_tokens, output_tokens)
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        def _role(r: str) -> str:
            r = (r or "").strip().lower()
            return r if r in ("system", "user", "assistant") else "user"

        prompt_text = "\n".join(
            f"{_role(m.get('role',''))}:{m.get('content','')}"
            for m in (prompt_messages or [])
        )
        pt = len(enc.encode(prompt_text or ""))
        ct = len(enc.encode(completion_text or ""))

        return pt, ct
    except Exception:
        return 0, 0


# ----------------------------
# Context Awareness (Out-of-Scope Guard)
# ----------------------------

_OOS_REPLY_MARKERS = [
    "i'm focused on investing",
    "i don't have that in my finance knowledge base",
    "out of scope",
    "not a finance question",
    "finance-only assistant",
    "i can't help with that topic",
    "try asking about portfolios, fees, stock prices",
]

_GUARD_REPLY = (
    "❌ Sorry — I couldn’t find anything about that in my knowledge base. "
    "I’m focused on investing and finance topics. Try asking about portfolios, fees, "
    "stock prices, or paste a document/link to index."
)


def _is_offtopic_reply(text: str) -> bool:
    """
    Determine if a reply is an off-topic refusal.

    Logic:
        - Normalizes reply to lowercase.
        - Checks against exact guard string or markers list.

    Args:
        text: Assistant reply text.

    Returns:
        True if text matches an off-topic refusal, else False.
    """
    s = (text or "").strip().lower()
    if not s:
        return False
    if s == _GUARD_REPLY.lower():
        return True
    return any(k in s for k in _OOS_REPLY_MARKERS)


# ----------------------------
# Build Sources from Tools + Docs
# ----------------------------

async def _build_sources_from_steps_and_docs(
    steps: list[Any],
    question: str
) -> list[dict]:
    """
    Build the 'sources' array to display in the UI.

    Rules:
        - If ANY tool produced a source → ONLY tool sources are returned (no RAG/docs).
        - If a STOCK tool ran → append a Finnhub source (once).
        - If a World Bank tool ran → append World Bank source (once).
        - Otherwise (no tool sources) → return confident RAG document sources.
    """
    traces: list[dict] = []
    tool_names_used: set[str] = set()

    # --- Normalize agent intermediate steps ---
    for step in steps or []:
        if isinstance(step, (list, tuple)) and len(step) == 2:
            action, observation = step
            tname = getattr(action, "tool", None)
            if tname:
                tool_names_used.add(str(tname))
            traces.append({
                "tool": tname,
                "args": getattr(action, "tool_input", {}) if action else {},
                "observation": observation,
            })
        else:
            traces.append({"tool": None, "args": None, "observation": step})

    # --- Convert tool traces → sources ---
    _raw_tool_items = tools_trace_to_sources(traces)
    tool_items = list(_raw_tool_items) if isinstance(_raw_tool_items, (list, tuple)) else []
    if not isinstance(_raw_tool_items, (list, tuple)):
        logger.warning("tools_trace_to_sources returned %r (type=%s); coerced to []",
                       _raw_tool_items, type(_raw_tool_items).__name__)

    # Normalize tool source fields
    for it in tool_items:
        it.setdefault("type", "tool")
        it.setdefault("title", it.get("display") or it.get("tool") or "Tool")
        disp = it.get("display")
        if not isinstance(disp, str):
            it["display"] = (str(disp)[:160] if disp is not None else it.get("title", "Tool"))
        it["title"] = str(it.get("title") or "Tool")[:160]

    # --- Append standard vendor badges when certain tool families were used ---
    if any(t in STOCK_TOOL_NAMES for t in tool_names_used):
        found = any((it.get("href") == FINNHUB_SOURCE["href"]) or
                    (it.get("url") == FINNHUB_SOURCE["url"]) or
                    (it.get("id") == FINNHUB_SOURCE["id"]) for it in tool_items)
        if not found:
            tool_items.append(dict(FINNHUB_SOURCE))

    if any("world_bank" in str(t).lower() or "worldbank" in str(t).lower() for t in tool_names_used):
        found = any((it.get("href") == WORLD_BANK_SOURCE["href"]) or
                    (it.get("url") == WORLD_BANK_SOURCE["url"]) or
                    (it.get("id") == WORLD_BANK_SOURCE["id"]) for it in tool_items)
        if not found:
            tool_items.append(dict(WORLD_BANK_SOURCE))

    # --- If we have any tool sources, return those only ---
    if tool_items:
        combined = tool_items
        all_items = dedupe_sources(combined) or []
        if not isinstance(all_items, list):
            logger.warning("dedupe_sources returned non-list (type=%s); falling back to combined",
                           type(all_items).__name__)
            all_items = combined
        return all_items or []

    # --- No tool sources, then fall back to confident RAG docs ---
    doc_items: list[dict] = []
    r = _get_retriever_lazy()
    if r:
        try:
            if hasattr(r, "ainvoke"):
                raw_docs = await r.ainvoke(question)
            elif hasattr(r, "invoke"):
                raw_docs = r.invoke(question)
            elif hasattr(r, "aget_relevant_documents"):
                raw_docs = await r.aget_relevant_documents(question)
            else:
                raw_docs = r.get_relevant_documents(question)
            raw_docs = (raw_docs or [])[:12]
            docs = rag_filter_confident(raw_docs)
            docs = (docs or [])[:4]
        except Exception:
            docs = []
    else:
        docs = []

    for d in docs or []:
        md = dict(getattr(d, "metadata", {}) or {})
        src = (md.get("source") or md.get("file") or md.get("path") or
               md.get("url") or md.get("title") or "")
        page = md.get("page")

        if isinstance(src, str) and src.lower().endswith(".pdf"):
            s = book_source_from_filename(src)
            base = os.path.basename(re.split(r"[?#]", src, 1)[0])
            name_no_ext = os.path.splitext(base)[0]
            title_like = re.sub(r"[_\\-]+", " ", name_no_ext).strip()
            try:
                from utils.utils import find_book_citations_in_text, book_source_from_title
                cands = find_book_citations_in_text(title_like)
                if cands:
                    s2 = book_source_from_title(cands[0])
                    if s2:
                        s = s2
            except Exception:
                pass
            s["type"] = "doc"
        else:
            s = normalize_source({"source": src, "url": md.get("url"), "title": md.get("title"), "page": page})

        # Attach page number to display where available
        if page not in (None, "", "?"):
            try:
                p = int(page)
            except Exception:
                p = page
            label = s.get("display") or s.get("title") or ""
            if label:
                s["display"] = f"{label} | Page {p}"
            s.setdefault("page", page)

        s.setdefault("title", s.get("display"))
        s.setdefault("type", "doc" if str(src).lower().endswith(".pdf") else "web")
        doc_items.append(s)

    combined = doc_items
    all_items = dedupe_sources(combined) or []
    if not isinstance(all_items, list):
        logger.warning("dedupe_sources returned non-list (type=%s); falling back to combined",
                       type(all_items).__name__)
        all_items = combined
    return all_items or []


# ----------------------------
# Agent Call Wrapper
# ----------------------------

async def _call_agent(
    agent: Any,
    inputs: dict,
    callbacks: Optional[list[Any]] = None,
    **kwargs
) -> Any:
    """
    Invoke a LangChain agent in a uniform async way.

    Supports agents exposing '.ainvoke', '.acall', '.invoke', or callable.

    Args:
        agent: Agent instance.
        inputs: Dict input (must include "input").
        callbacks: Optional callback handlers.
        **kwargs: Extra args.

    Returns:
        Agent result.
    """
    if isinstance(callbacks, (list, tuple)):
        cbs = list(callbacks)
    elif callbacks:
        cbs = [callbacks]
    else:
        cbs = []

    cfg = {"callbacks": cbs} if cbs else {}

    logger.info(
        "Agent call begin | keys=%s has_callbacks=%s",
        list(inputs.keys()),
        bool(cbs),
    )

    try:
        if hasattr(agent, "ainvoke"):
            res = await agent.ainvoke(inputs, config=cfg or None)
        elif hasattr(agent, "acall"):
            res = await agent.acall(inputs, config=cfg or None)
        elif hasattr(agent, "invoke"):
            res = agent.invoke(inputs, config=cfg or None)
        else:
            tmp = agent(inputs) if callable(agent) else None
            res = await tmp if inspect.isawaitable(tmp) else tmp
        return res
    finally:
        logger.info("Agent call end")


# ----------------------------
# SSE Token Handler
# ----------------------------

class SSETokenHandler(AsyncCallbackHandler):
    """
    LangChain callback handler that streams tokens via SSE.
    - Queues tokens into 'self.queue'.
    - Sends None at end-of-stream or on error.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called each time the LLM generates a token."""
        await self.queue.put(token)

    async def on_llm_end(self, *args, **kwargs) -> None:
        """Called when the LLM finishes (signal end-of-stream)."""
        await self.queue.put(None)

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called if an error occurs (still signal end-of-stream)."""
        await self.queue.put(None)



# ---------------- Chat endpoint ----------------

@app.post("/api/chat")
async def chat(
    request: Request,
    req: ChatInput,
    db: SASession = Depends(get_db),
) -> dict[str, Any]:
    """
    Core chat endpoint (non-streaming).

    Flow:
        1. Validate request & last user message.
        2. Apply guardrails:
            - If no docs & not a toolable query → refuse with _GUARD_REPLY.
        3. Manage session:
            - Ensure session exists.
            - Append user message to DB.
            - Rebuild history from DB.
        4. Call agent (LangChain executor).
        5. Collect results:
            - LLM output text.
            - Tool intermediate steps.
            - Token usage + cost (via with_token_log).
        6. Post-process response:
            - Merge tool markdown with model text.
            - Deduplicate/normalize text.
        7. Append assistant turn to DB.
        8. Build & return structured JSON response.

    Args:
        request: FastAPI request object.
        req: Pydantic model containing 'messages' list.
        db: SQLAlchemy session (injected via Depends).

    Returns:
        JSON dict with:
            {
              "sessionId": str,
              "text": str,
              "sources": list[dict],
              "usage": {...}
            }
    """
    t0 = time.perf_counter()

    if rag_agent_chain is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # --- Validate input ---
    messages = req.messages or []
    last = messages[-1] if messages else None
    if not last or (last.role or "").lower() != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user.")

    raw_question = last.content or ""

    # --- Moderation ---
    mod = _moderate_text(raw_question)
    if mod and mod.get("flagged"):
        session_id = _ensure_session(db, (request.headers.get("X-Session-Id") or "").strip(), "Chat")
        append_message(db, session_id, "user", raw_question, None)
        text = "I’m here to help with safe, finance-focused questions. Please rephrase or ask about investing, markets, or personal finance topics."
        append_message(db, session_id, "assistant", text, None)
        touch_session(db, session_id)
        return {
            "sessionId": session_id,
            "text": text,
            "sources": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "model": "moderation",
                "input_tokens": 0,
                "output_tokens": 0,
                "synthetic": True,
            },
        }

    auth_sid = _require_client_auth(request)
    raw_header_sid = (request.headers.get("X-Session-Id") or "").strip()
    invalid = raw_header_sid.lower() in {"", "null", "undefined", "none"}
    header_sid = None if invalid else raw_header_sid
    if session_tokens.enabled and auth_sid and header_sid and auth_sid != header_sid:
        raise HTTPException(
            status_code=403,
            detail="Session token does not match X-Session-Id.",
        )
    sid_hint = auth_sid or header_sid

    if re.match(r"^\s*(hi|hello|hey|howdy|hiya)\b", raw_question, re.I):
        session_id = _ensure_session(
            db, sid_hint, "Chat"
        )
        append_message(db, session_id, "user", raw_question, None)
        text = "Hello! How can I help with your investing or market questions today?"
        append_message(db, session_id, "assistant", text, None)
        touch_session(db, session_id)
        return {
            "sessionId": session_id,
            "text": text,
            "sources": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "model": "synthetic",
                "input_tokens": 0,
                "output_tokens": 0,
                "synthetic": True,
            },
        }

    try:
        from .llm.rag import translate_query
        question = translate_query(raw_question) or raw_question
    except Exception:
        question = raw_question
    is_summary = "summar" in (question or "").lower()
    allow_tools = _looks_toolable(raw_question) or _looks_toolable(question)

    # --- Guardrails ---
    # 1) Domain guard: refuse clearly non-finance questions early.
    #    If toolable or a summary request, allow even if the finance heuristic is unsure.
    try:
        tools_exist = bool(TOOLS)
    except Exception:
        tools_exist = False

    finance_ok = _is_finance_query(raw_question) or _is_finance_query(question)
    if not finance_ok and not allow_tools and not is_summary and not tools_exist:
        logger.info("GUARD_METRIC | kind=non_finance q=%s", question[:120])
        session_id = _ensure_session(
            db,
            sid_hint,
            "Chat",
        )
        append_message(db, session_id, "user", raw_question, None)
        append_message(db, session_id, "assistant", _GUARD_REPLY, None)

        # Emit analytics-style event so we can track guard frequency.
        meta = {
            "type": "guard",
            "session_id": session_id,
            "guard_kind": "non_finance",
            "model": "guard",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }
        # analytics disabled

        return {
            "sessionId": session_id,
            "text": _GUARD_REPLY,
            "sources": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "model": "guard",
                "input_tokens": 0,
                "output_tokens": 0,
                "synthetic": True,
            },
        }

    # 2) Capability guard: if no docs and no tools, also refuse.
    allow_summary = is_summary
    has_docs, max_score = _has_useful_docs(question)
    if (not has_docs) and (not allow_tools) and (not tools_exist) and (not allow_summary):
        logger.info(
            "GUARD_METRIC | kind=no_rag_no_tools max_score=%.3f q=%s",
            max_score,
            question[:120],
        )
        session_id = _ensure_session(
            db,
            sid_hint,
            "Chat",
        )
        append_message(db, session_id, "user", raw_question, None)
        append_message(db, session_id, "assistant", _GUARD_REPLY, None)

        meta = {
            "type": "guard",
            "session_id": session_id,
            "guard_kind": "no_rag_no_tools",
            "model": "guard",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }
        # analytics disabled

        return {
            "sessionId": session_id,
            "text": _GUARD_REPLY,
            "sources": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "model": "guard",
                "input_tokens": 0,
                "output_tokens": 0,
                "synthetic": True,
            },
        }

    # --- Session handling ---
    session_id = _ensure_session(db, sid_hint, "Chat")
    append_message(db, session_id, "user", raw_question, None)

    # --- Rebuild history from DB ---
    rows = list_messages(db, session_id, limit=2000)
    history = [{"role": r.role, "content": r.content} for r in rows]
    lc_history = to_lc_messages(history)

    # --- Input token size guard ---
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        joined = "\n".join(f"{m['role']}:{m['content']}" for m in history)
        approx_in = len(enc.encode(joined))
    except Exception:
        approx_in = sum(len((m["content"] or "")) for m in history) // 4

    if approx_in > MAX_REQUEST_TOKENS:
        raise HTTPException(status_code=413, detail="Request too large.")

    # --- Call agent with token logging ---
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    with with_token_log(model_name) as usage:
        cbs = usage.get("callbacks") or []
        logger.info("CHAT start sid=%s model=%s", session_id, model_name)
        logger.info("with_token_log: callbacks attached? %s (n=%d)", bool(cbs), len(cbs))

        result: dict = await _call_agent(
            rag_agent_chain,
            {"input": question, "history": lc_history},
            callbacks=cbs,
        )

    # --- Extract text + steps ---
    if isinstance(result, dict):
        text = (
            result.get("output")
            or result.get("output_text")
            or result.get("response")
            or ""
        )
        steps = result.get("intermediate_steps") or []
    else:
        text = str(result or "")
        steps = []

    # analytics disabled

    # --- Post-process response text ---
    addon = _collect_tool_markdown(steps)
    any_tool_used = bool(steps)

    if any_tool_used and addon and addon.strip():
        text = addon.strip()
    elif addon and addon.strip() and addon.strip() not in (text or ""):
        text = (text.rstrip() + "\n\n" + addon.strip()) if text else addon.strip()

    if not (text and text.strip()) and steps:
        try:
            raw_obs = []
            for st in steps:
                if isinstance(st, (list, tuple)) and len(st) == 2:
                    _, observation = st
                elif isinstance(st, dict):
                    observation = st.get("observation")
                else:
                    observation = None
                if isinstance(observation, str) and observation.strip():
                    raw_obs.append(observation.strip())
            if raw_obs:
                text = "\n\n".join(raw_obs)
                logger.warning("Using raw tool observations as response [non-stream]")
        except Exception as e:
            logger.warning("Failed to extract raw observations [non-stream]: %s", e)

    text = _postprocess_markdown(text)
    text = _unsplit_link_urls(text)

    # --- Build sources (but suppress for off-topic/refusal) ---
    if _is_offtopic_reply(text):
        tool_sources: list[dict] = []
    else:
        pre_sources = await _build_sources_from_steps_and_docs(steps, question)
        pre_sources = _coerce_strings(pre_sources)
        tool_sources = _filter_sources_to_text(
            text, pre_sources, keep_docs_if_present=True
        )
        if not tool_sources:
            tool_sources = pre_sources

    # Fallback RAG: if no sources/tools were used but retriever exists, try strict RAG with citations.
    if not tool_sources:
        try:
            r = _get_retriever_lazy()
            if r is not None:
                rag_text, used_docs = _strict_answer_with_tagged_context(
                    question, r, max_docs=6
                )
            else:
                rag_text, used_docs = ("", [])
            if rag_text and used_docs:
                text = _postprocess_markdown(rag_text.strip())
                tool_sources = docs_to_sources(used_docs)
        except Exception as e:
            logger.warning("RAG fallback failed: {}", e)

    append_message(db, session_id, "assistant", text, None)
    touch_session(db, session_id)


    # --- Token usage + cost ---
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    tt = int(usage.get("total_tokens", pt + ct))
    cu = usage.get("cost_usd")
    model_used = usage.get("model") or model_name

    if pt == 0 and ct == 0:
        logger.warning(
            "Token usage is zero AFTER agent call. Falling back to approx count."
        )
        approx_pt, approx_ct = _approx_count_tokens(model_used, history, text)
        if approx_pt or approx_ct:
            pt, ct = approx_pt, approx_ct
            tt = pt + ct
            cu = _approx_cost_usd(model_used, pt, ct)
            logger.warning("Approximated token usage via tiktoken: pt=%d ct=%d", pt, ct)

    # --- Analytics ---
    latency_ms = int((time.perf_counter() - t0) * 1000)
    had_rag = _had_rag_or_tools(tool_sources)
    meta = {
        "type": "turn",
        "session_id": session_id,
        "model": model_used,
        "tokens_in": pt,
        "tokens_out": ct,
        "cost_usd": cu or 0.0,
        "latency_ms": latency_ms,
        "had_rag": had_rag,
        "tools_used": [],
        "response_preview": (text or "")[:500],
        "error": None,
    }
    # analytics disabled
    try:
        on_llm_end(meta)
    except Exception as e:
        logger.warning("on_llm_end failed: %s", e)

    return {
        "sessionId": session_id,
        "text": text,
        "sources": tool_sources,
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "cost_usd": cu,
            "model": model_used,
            "input_tokens": pt,
            "output_tokens": ct,
        },
    }



# ---------------- Streaming Chat endpoint ----------------

@app.post("/api/chat/stream")
async def chat_stream(
    request: Request,
    req: ChatInput,
    db: SASession = Depends(get_db),
) -> StreamingResponse:
    """
    Chat endpoint with Server-Sent Events (SSE) streaming.

    Flow:
        1. Validate request & last user message.
        2. Apply guardrails:
            - If no docs & not toolable → stream refusal tokens.
        3. Manage session:
            - Ensure session exists.
            - Append user turn to DB.
            - Rebuild history from DB.
        4. Spawn agent task with SSETokenHandler callback.
        5. Continuously read token queue:
            - Forward tokens as SSE events.
            - Heartbeat ping every 1.5s to keep connection alive.
        6. On completion:
            - Collect final text & tool steps.
            - Normalize response.
            - Append assistant turn to DB.
            - Stream final frame with sources + usage.

    Returns:
        StreamingResponse with 'text/event-stream' media type.
    """
    t0 = time.perf_counter()

    if rag_agent_chain_stream is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # --- Validate input ---
    messages = req.messages or []
    last = messages[-1] if messages else None
    if not last or (last.role or "").lower() != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user.")
    raw_question = last.content or ""

    # --- Moderation ---
    mod = _moderate_text(raw_question)
    if mod and mod.get("flagged"):
        async def g_moderation():
            def sse(obj: dict) -> bytes:
                return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b":" + b" " * 2048 + b"\n\n"
            yield sse({"token": ""})
            text = "I’m here to help with safe, finance-focused questions. Please rephrase or ask about investing, markets, or personal finance topics."
            for ch in text:
                yield sse({"token": ch})
                await asyncio.sleep(0)
            yield sse({
                "done": True,
                "text": text,
                "final": text,
                "sources": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "model": "moderation",
                    "synthetic": True,
                },
            })
        return StreamingResponse(
            g_moderation(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    auth_sid = _require_client_auth(request)
    raw_header_sid = (request.headers.get("X-Session-Id") or "").strip()
    invalid = raw_header_sid.lower() in {"", "null", "undefined", "none"}
    header_sid = None if invalid else raw_header_sid
    if session_tokens.enabled and auth_sid and header_sid and auth_sid != header_sid:
        raise HTTPException(
            status_code=403,
            detail="Session token does not match X-Session-Id.",
        )
    sid_hint = auth_sid or header_sid

    if re.match(r"^\s*(hi|hello|hey|howdy|hiya)\b", raw_question, re.I):
        async def g():
            def sse(obj: dict) -> bytes:
                return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b":" + b" " * 2048 + b"\n\n"
            yield sse({"token": ""})
            text = "Hello! How can I help with your investing or market questions today?"
            for ch in text:
                yield sse({"token": ch})
                await asyncio.sleep(0)
            yield sse({"done": True, "text": text, "final": text, "sources": [], "usage": {"total_tokens": 0}})
        return StreamingResponse(
            g(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        from .llm.rag import translate_query
        question = translate_query(raw_question) or raw_question
    except Exception:
        question = raw_question
    is_summary = "summar" in (question or "").lower()
    allow_tools = _looks_toolable(raw_question) or _looks_toolable(question)

    # --- Domain guard (before doing heavier work) ---
    try:
        tools_exist = bool(TOOLS)
    except Exception:
        tools_exist = False

    finance_ok = _is_finance_query(raw_question) or _is_finance_query(question)
    if not finance_ok and not allow_tools and not is_summary and not tools_exist:
        logger.info("GUARD_METRIC | kind=non_finance_stream q=%s", question[:120])

        async def g_nonfinance():
            def sse(obj: dict) -> bytes:
                return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b":" + b" " * 2048 + b"\n\n"
            yield sse({"token": ""})
            for ch in _GUARD_REPLY:
                yield sse({"token": ch})
                await asyncio.sleep(0)
            yield sse({
                "done": True,
                "text": _GUARD_REPLY,
                "final": _GUARD_REPLY,
                "sources": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "model": "guard",
                    "synthetic": True,
                },
            })

        # Emit a lightweight analytics event for guard usage.
        try:
            meta = {
                "type": "guard",
                "session_id": sid_hint,
                "guard_kind": "non_finance",
                "model": "guard",
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
            }
            # analytics disabled
        except Exception:
            pass

        return StreamingResponse(
            g_nonfinance(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Session management and history rebuild
    session_id = _ensure_session(db, sid_hint, "Chat")

    append_message(db, session_id, "user", raw_question, None)
    rows = list_messages(db, session_id, limit=2000)
    history = [{"role": r.role, "content": r.content} for r in rows]
    lc_history = to_lc_messages(history)

    try:
        # Size guard
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        joined = "\n".join(f"{m['role']}:{m['content']}" for m in history)
        approx_in = len(enc.encode(joined))
    except Exception:
        approx_in = sum(len((m["content"] or "")) for m in history) // 4

    if approx_in > MAX_REQUEST_TOKENS:
        raise HTTPException(status_code=413, detail="Request too large.")

    # --- SSE generator ---
    async def eventgen():
        emitted_any = False

        def sse(obj: dict) -> bytes:
            """Helper: format a dict as SSE JSON line."""
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

        yield b":" + b" " * 2048 + b"\n\n"
        yield sse({"token": ""})
        await asyncio.sleep(0)

        # --- Guardrails (streaming refusal to prevent hallucinations) ---
        allow_tools = _looks_toolable(raw_question) or _looks_toolable(question)
        allow_summary = is_summary
        has_docs, max_score = _has_useful_docs(question)
        try:
            tools_exist = bool(TOOLS)
        except Exception:
            tools_exist = False

        if (not has_docs) and (not allow_tools) and (not tools_exist) and (not allow_summary):
            logger.info(
                "GUARD_METRIC | kind=no_rag_no_tools max_score=%.3f q=%s",
                max_score,
                question[:120],
            )
            # Let summary requests through even if no docs/tools.
            if allow_summary:
                yield sse({"token": ""})
            else:
                for ch in _GUARD_REPLY:
                    yield sse({"token": ch})
                    await asyncio.sleep(0)
                yield sse({
                    "done": True,
                    "sources": [],
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                        "model": "guard",
                        "synthetic": True,
                    },
                })
                return

        try:
            logger.info("CHAT_STREAM start sid=%s", session_id)
            # Launch agent, start streaming
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            with with_token_log(model_name) as usage:
                stream_cb = SSETokenHandler()
                cbs_core = usage.get("callbacks") or []
                cbs = cbs_core + [stream_cb]

                agent_task = asyncio.create_task(
                    _call_agent(
                        rag_agent_chain_stream,
                        {"input": question, "history": lc_history},
                        callbacks=cbs,
                    )
                )
                agent_task.add_done_callback(lambda _f: stream_cb.queue.put_nowait(None))

                buf_tokens: list[str] = []

                while True:
                    token_task = asyncio.create_task(stream_cb.queue.get())
                    ping_task = asyncio.create_task(asyncio.sleep(1.5))
                    done, _ = await asyncio.wait(
                        {token_task, ping_task}, return_when=asyncio.FIRST_COMPLETED
                    )

                    if token_task in done:
                        token = token_task.result()
                        if token is None:
                            break
                        if token:
                            emitted_any = True
                            buf_tokens.append(token)
                            yield sse({"token": token})
                    else:
                        yield sse({"ping": int(time.time())})
                        token_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await token_task

                result: dict = await agent_task
            # consolidate text and tool traces
            if isinstance(result, dict):
                text = (
                    result.get("output")
                    or result.get("output_text")
                    or result.get("response")
                    or ""
                )
                steps = result.get("intermediate_steps") or []
            else:
                text = str(result or "")
                steps = []

            # analytics disabled

            addon = _collect_tool_markdown(steps)
            any_tool_used = bool(steps)

            if any_tool_used and addon and addon.strip():
                text = addon.strip()
            elif addon and addon.strip() and addon.strip() not in (text or ""):
                text = (text.rstrip() + "\n\n" + addon.strip()) if text else addon.strip()

            if not (text and text.strip()):
                text = "".join(buf_tokens).strip()

            if not (text and text.strip()) and steps:
                try:
                    raw_obs = []
                    for st in steps:
                        if isinstance(st, (list, tuple)) and len(st) == 2:
                            _, observation = st
                        elif isinstance(st, dict):
                            observation = st.get("observation")
                        else:
                            observation = None
                        if isinstance(observation, str) and observation.strip():
                            raw_obs.append(observation.strip())
                    if raw_obs:
                        text = "\n\n".join(raw_obs)
                        logger.warning("Using raw tool observations as response [stream]")
                except Exception as e:
                    logger.warning("Failed to extract raw observations [stream]: %s", e)
            # Clean up formatting
            text = _postprocess_markdown(text)
            text = _unsplit_link_urls(text)
            if text:
                CHUNK = 160
                if not emitted_any:
                    for i in range(0, len(text), CHUNK):
                        yield sse({"token": text[i:i+CHUNK]})
                        await asyncio.sleep(0)
                    emitted_any = True
            else:
                logger.warning("Empty text after tools/tokens; emitting nothing.")

            # Persist assistant turn
            append_message(db, session_id, "assistant", text, None)
            touch_session(db, session_id)

            # --- Build sources ---
            if _is_offtopic_reply(text):
                tool_sources: list[dict] = []
            else:
                pre_sources = await _build_sources_from_steps_and_docs(steps, question)
                pre_sources = _coerce_strings(pre_sources)
                tool_sources = _filter_sources_to_text(
                    text, pre_sources, keep_docs_if_present=True
                )

            if not tool_sources:
                tool_sources = pre_sources

            # Fallback RAG: if no sources/tools were used but retriever exists, try strict RAG with citations.
            if not tool_sources:
                try:
                    r = _get_retriever_lazy()
                    if r is not None:
                        rag_text, used_docs = _strict_answer_with_tagged_context(
                            question, r, max_docs=6
                        )
                    else:
                        rag_text, used_docs = ("", [])
                    if rag_text and used_docs:
                        text = _postprocess_markdown(rag_text.strip())
                        tool_sources = docs_to_sources(used_docs)
                except Exception as e:
                    logger.warning("RAG fallback (stream) failed: {}", e)

            # --- Token usage & cost ---
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            tt = int(usage.get("total_tokens", pt + ct))
            cu = usage.get("cost_usd") or usage.get("total_cost")
            model_used = usage.get("model") or model_name

            if pt == 0 and ct == 0:
                logger.warning("Token usage is zero AFTER stream call. Approximating.")
                approx_pt, approx_ct = _approx_count_tokens(model_used, history, text)
                if approx_pt or approx_ct:
                    pt, ct = approx_pt, approx_ct
                    tt = pt + ct
                    cu = _approx_cost_usd(model_used, pt, ct)

            logger.info(
                "USAGE sid=%s model=%s pt=%d ct=%d tt=%d cost=$%.6f",
                session_id, model_used, pt, ct, tt, float(cu or 0.0),
            )

            yield sse({
                "done": True,
                "text": text,
                "final": text,
                "sources": tool_sources,
                "usage": {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt,
                    "cost_usd": cu,
                    "model": model_used,
                    "input_tokens": pt,
                    "output_tokens": ct,
                },
            })

            # --- Analytics ---
            latency_ms = int((time.perf_counter() - t0) * 1000)
            had_rag = _had_rag_or_tools(tool_sources)
            meta = {
                "type": "turn",
                "session_id": session_id,
                "model": model_used,
                "tokens_in": pt,
                "tokens_out": ct,
                "cost_usd": cu or 0.0,
                "latency_ms": latency_ms,
                "had_rag": had_rag,
                "tools_used": [],
                "response_preview": (text or "")[:500],
                "error": None,
            }
            # analytics disabled
            on_llm_end(meta)

            logger.info("CHAT_STREAM end sid=%s", session_id)

        except Exception as e:
            yield sse({"error": str(e)})

            partial = ""
            try:
                buf = "".join(locals().get("buf_tokens", []))
                if buf.strip():
                    partial = _postprocess_markdown(buf)
            except Exception:
                pass

            yield sse({
                "done": True,
                "text": partial,
                "final": partial,
                "sources": [],
                "usage": {"total_tokens": 0},
            })
            return

        finally:
            try:
                if 'agent_task' in locals() and isinstance(agent_task, asyncio.Task) and not agent_task.done():
                    agent_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await agent_task
            except Exception:
                pass
            try:
                if 'stream_cb' in locals() and hasattr(stream_cb, 'queue'):
                    stream_cb.queue.put_nowait(None)
            except Exception:
                pass

    return StreamingResponse(
        eventgen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ----------------------------
# Chat Export Endpoints
# ----------------------------

@app.get("/api/chat/{session_id}/export.json")
def export_json(
    session_id: str, request: Request, db: SASession = Depends(get_db)
) -> JSONResponse:
    """
    Export conversation as JSON.

    Args:
        session_id: ID of the chat session.
        db: SQLAlchemy session.

    Returns:
        JSONResponse with:
            {
              "sessionId": str,
              "messages": [
                {"id": int, "role": str, "content": str, "created_at": str}
              ]
            }
    """
    _require_client_auth(request, session_id)
    rows = list_messages(db, session_id, limit=2000)
    return JSONResponse({
        "sessionId": session_id,
        "messages": [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
    })


@app.get("/api/chat/{session_id}/export.pdf")
def export_pdf(
    session_id: str, request: Request, db: SASession = Depends(get_db)
):
    """
    Export conversation as a styled PDF (via WeasyPrint).

    Args:
        session_id: ID of the chat session.
        db: SQLAlchemy session.

    Returns:
        StreamingResponse with PDF attachment.
        If WeasyPrint is not installed → 501 Not Implemented.
    """
    _require_client_auth(request, session_id)
    rows = list_messages(db, session_id, limit=2000)

    html = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Inter,ui-sans-serif,system-ui; padding:24px;}",
        "h1{font-size:20px;margin-bottom:12px}",
        ".msg{margin:12px 0;}.role{font-weight:600;margin-right:8px;}",
        ".user{color:#0b6bcb}.assistant{color:#16a34a}.system{color:#6b7280}",
        "</style></head><body>",
        f"<h1>Chat Export — {session_id}</h1>",
    ]

    for r in rows:
        role = (r.role or "").lower()
        cls = (
            "user" if role == "user"
            else "assistant" if role == "assistant"
            else "system"
        )
        safe = (
            (r.content or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        html.append(
            f"<div class='msg'><span class='role {cls}'>{role}:</span>"
            f"<span class='content'>{safe}</span></div>"
        )

    html.append("</body></html>")
    html_str = "".join(html)

    try:
        from weasyprint import HTML
        pdf_io = io.BytesIO()
        HTML(string=html_str).write_pdf(pdf_io)
        pdf_io.seek(0)
        return StreamingResponse(
            pdf_io,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="chat-{session_id}.pdf"'
            },
        )
    except Exception as e:
        return PlainTextResponse(
            f"WeasyPrint not available: {e}",
            status_code=501,
        )


@app.get("/api/chat/{session_id}/export.csv")
def export_csv(
    session_id: str, request: Request, db: SASession = Depends(get_db)
) -> StreamingResponse:
    """
    Export conversation as CSV.

    Columns: id, created_at, role, content

    Args:
        session_id: ID of the chat session.
        db: SQLAlchemy session.

    Returns:
        StreamingResponse with 'text/csv' attachment.
    """
    _require_client_auth(request, session_id)
    rows = list_messages(db, session_id, limit=2000)

    buf = io.StringIO(newline="")
    writer = csv.writer(buf)

    writer.writerow(["id", "created_at", "role", "content"])

    for r in rows:
        writer.writerow([
            r.id,
            r.created_at.isoformat(),
            (r.role or ""),
            (r.content or ""),
        ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="chat-{session_id}.csv"'
        },
    )

@app.get("/api/chat/{session_id}/export.txt")
def export_txt(
    session_id: str, request: Request, db: SASession = Depends(get_db)
) -> PlainTextResponse:
    """
    Export conversation as plain text.
    """
    _require_client_auth(request, session_id)
    rows = list_messages(db, session_id, limit=2000)
    lines = [f"{(r.role or '').upper()}: {r.content or ''}".strip() for r in rows]
    body = "\n".join(lines)
    return PlainTextResponse(
        body,
        headers={"Content-Disposition": f'attachment; filename="chat-{session_id}.txt"'},
    )


@app.get("/api/chat/{session_id}/export.md")
def export_md(
    session_id: str, request: Request, db: SASession = Depends(get_db)
) -> PlainTextResponse:
    """
    Export conversation as Markdown.
    """
    _require_client_auth(request, session_id)
    rows = list_messages(db, session_id, limit=2000)
    parts = [f"# Chat {session_id}", ""]
    for r in rows:
        role = (r.role or "").strip().title() or "Message"
        content = r.content or ""
        parts.append(f"**{role}:** {content}")
        parts.append("")
    body = "\n".join(parts).strip()
    return PlainTextResponse(
        body,
        headers={"Content-Disposition": f'attachment; filename="chat-{session_id}.md"'},
        media_type="text/markdown",
    )

# ----------------------------
# Health Check
# ----------------------------

@app.get("/api/health")
def health() -> dict[str, bool]:
    """
    Simple health check endpoint.

    Returns:
        {"ok": True}
    """
    return {"ok": True}

@app.get("/api/version")
def version() -> dict[str, str]:
    return {
        "app": APP_NAME,
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


# ----------------------------
# Debug: Stream Simulation
# ----------------------------

@app.get("/api/_debug/stream")
async def _debug_stream() -> StreamingResponse:
    """
    Simulate a streaming response for frontend testing.

    Sends a small sequence of tokens over SSE:
        - Empty ack token
        - "Hello world!"
        - Final done frame
    """
    async def g():
        def sse(obj: dict) -> bytes:
            return f"data: {json.dumps(obj)}\n\n".encode("utf-8")

        yield sse({"token": ""})
        await asyncio.sleep(0.05)

        for t in ["Hello", " world", "!"]:
            yield sse({"token": t})
            await asyncio.sleep(0.05)

        yield sse({"done": True, "sources": [], "usage": {"total_tokens": 0}})

    return StreamingResponse(
        g(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ----------------------------
# Debug: Token Usage Probe
# ----------------------------

@app.get("/api/_debug/usage")
async def _debug_usage() -> dict[str, Any]:
    """
    Probe token logging with a minimal LLM call.

    Flow:
        1. Open a with_token_log context.
        2. Call the raw LLM with callbacks attached.
        3. Return observed token counts + cost.
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    from .analytics.tokenlog import with_token_log

    with with_token_log(model_name) as usage:
        if (
            int(usage.get("prompt_tokens", 0)) == 0
            and int(usage.get("completion_tokens", 0)) == 0
        ):
            logger.warning(
                "Token usage still zero; callbacks not propagated. model=%s",
                usage.get("model"),
            )

        cbs = usage.get("callbacks")

        try:
            if hasattr(llm, "ainvoke"):
                try:
                    await llm.ainvoke(
                        "Say hi in one short sentence.",
                        config={"callbacks": cbs} if cbs else None,
                    )
                except TypeError:
                    await llm.ainvoke("Say hi in one short sentence.")
            else:
                try:
                    llm.invoke(
                        "Say hi in one short sentence.",
                        config={"callbacks": cbs} if cbs else None,
                    )
                except TypeError:
                    llm.invoke("Say hi in one short sentence.")
        except Exception as e:
            return {"error": str(e), "model": model_name}

    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "cost_usd": usage.get("cost_usd"),
        "model": usage.get("model"),
    }


# ----------------------------
# Debug: Finnhub Probe
# ----------------------------

@app.get("/api/_debug/finnhub_news")
async def _debug_finnhub_news(
    symbol: str = "AAPL",
    _from: Optional[str] = None,
    _to: Optional[str] = None,
) -> JSONResponse:
    """
    Probe Finnhub news endpoint to debug timeouts.

    Example:
        /api/_debug/finnhub_news?symbol=AAPL&_from=2025-08-20&_to=2025-08-25

    Args:
        symbol: Stock ticker (default: AAPL).
        _from: Start date (YYYY-MM-DD).
        _to: End date (YYYY-MM-DD).

    Returns:
        JSONResponse:
            - {ok: True, count: int, sample: [...]} if successful
            - {ok: False, error: str} if timeout or other error
    """
    from datetime import date, timedelta

    if _from is None or _to is None:
        today = date.today()
        _to = today.isoformat()
        _from = (today - timedelta(days=7)).isoformat()

    try:
        data = await FINNHUB.get("/company-news", {"symbol": symbol, "from": _from, "to": _to})
        return {"ok": True, "count": len(data or []), "sample": (data or [])[:3]}
    except httpx.TimeoutException as e:
        return JSONResponse({"ok": False, "kind": "timeout", "error": str(e)}, status_code=504)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.get("/api/_debug/rag_status")
def _debug_rag_status():
    try:
        from .llm.rag import get_vectorstore, _estimate_collection_size
        vs = get_vectorstore()
        size = _estimate_collection_size(vs)
        backend = (os.getenv("VECTORSTORE", "qdrant") or "qdrant").lower()
        meta = {
            "backend": backend,
            "collection": os.getenv("QDRANT_COLLECTION") if backend == "qdrant" else None,
            "estimated_vectors": size,
            "retriever_ready": _get_retriever_lazy() is not None,
        }
        return {"ok": True, **meta}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/_debug/tools")
async def _debug_tools():
    tools, _ = _collect_tools()
    return {"count": len(tools), "names": [getattr(t, "name", "<unnamed>") for t in tools]}



# ---------- MCP Debug ----------
from .mcp_ext.registry import resolve_server
from .mcp_ext.client import list_tools as _mcp_list_tools, call_tool as _mcp_call_tool

@app.get("/api/_debug/mcp_ping")
async def _debug_mcp_ping() -> dict:
    """
    Verify the backend can reach the MCP proxy and list tools.
    Returns: { ok, server, count, names, sample }
    """
    try:
        server = resolve_server("world_bank")
    except Exception as e:
        return {"ok": False, "error": f"resolve_server failed: {e}"}

    try:
        tools = await _mcp_list_tools(server)
        names = [t.get("name") for t in (tools or []) if isinstance(t, dict)]
        return {
            "ok": True,
            "server": server,
            "count": len(names),
            "names": names,
            "sample": (tools or [])[:2],
        }
    except Exception as e:
        return {"ok": False, "server": server, "error": str(e)}

@app.post("/api/_debug/mcp_call")
async def _debug_mcp_call(payload: dict) -> dict:
    """
    Call a specific MCP tool directly for debugging.
    Body: {"name":"<toolName>", "arguments":{...}}
    """
    try:
        server = resolve_server("world_bank")
        name = str(payload.get("name") or "").strip()
        args = payload.get("arguments") or {}
        if not name:
            return {"ok": False, "error": "Missing 'name'."}
        result = await _mcp_call_tool(server, name, args)
        return {"ok": True, "server": server, "name": name, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}
