import os
from dotenv import load_dotenv

# IMPORTANT: Load env before importing modules that read env at import time
_ENV_ROOT = os.path.join(os.path.dirname(__file__), "..", ".env")
_ENV_BACKEND = os.path.join(os.path.dirname(__file__), ".env")
for _p in (_ENV_ROOT, _ENV_BACKEND):
    try:
        if os.path.exists(_p):
            load_dotenv(dotenv_path=_p, override=False)
    except Exception:
        pass

from typing import Callable, Awaitable
from fastapi import FastAPI, Request, Response
import uuid
import logging
from fastapi.middleware.cors import CORSMiddleware
from routers.owner_availability import router as sched_availability
from routers.admin_book import router as sched_admin_book
from routers.client import router as client_router
from routers.owner_appointments import router as sched_owner_appointments
from routers.owner_timeoff import router as sched_timeoff
from routers.public import router as sched_public
from routers.owner_openings import router as sched_openings
from routers.owner_clients import router as owner_clients
from routers.owner_analytics import router as owner_analytics_router
from routers.owner_pricing import router as owner_pricing_router
from routers.client_email import router as client_email_router
from routers.agent_chat import router as agent_chat
from routers.dev_inspect import router as dev_inspect_router
from routers.router_env_debug import router as env_router
from routers.me import router as me
from routers.owner import router as owner
from routers.fun import router as fun_router
from routers.outbox import router as outbox_router
from app.db import init_db, DB_URL
from app.migrations_runner import run_alembic_upgrade_head
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)


def _mask_dsn(url: str | None) -> str:
    """
    Produce a safe-to-log rendering of a database DSN/URL.

    Preserves the scheme, user (without password), host, optional port, and database
    name, but avoids printing sensitive components such as passwords or query strings.
    Args:
        url: A database URL (e.g., postgres DSN) or None.

    Returns:
        A masked string suitable for console logging.
    """

    if not url:
        return "unset"
    try:
        parsed = urlparse(url)
        user = parsed.username or ""
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        db = parsed.path.lstrip("/")
        user_part = f"{user}@" if user else ""
        return f"{parsed.scheme}://{user_part}{host}{port}/{db}"
    except Exception:
        return "masked"


def _parse_origins(raw: str | None) -> list[str]:
    """
    Parse a comma-separated list of origins into a list, or provide sane defaults.

    When 'raw' is None/empty, defaults to localhost origins commonly used for
    Next.js/Vite dev servers.

    Args:
        raw: Raw comma-separated origins (e.g., "https://a.com, https://b.com").

    Returns:
        A list of normalized, non-empty origin strings.
    """
    if not raw:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


_ALLOWED_ORIGINS = _parse_origins(os.getenv("CORS_ALLOWED_ORIGINS"))

app = FastAPI(title="AI Scheduling Assistant", version="0.1.0")


@app.middleware("http")
async def add_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Middleware: attach a per-request correlation ID.

    Behavior:
    - If the client provided 'X-Request-ID', reuse it; otherwise generate a UUID4.
    - Store the value in 'request.state.request_id' for downstream access.
    - Ensure the response includes the same ID in the 'X-Request-ID' header.

    Args:
        request: Incoming FastAPI request object.
        call_next: Middleware continuation delegate.

    Returns:
        The HTTP response produced by downstream handlers, with header added.
    """
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    try:
        request.state.request_id = rid
    except Exception:
        pass
    resp = await call_next(request)
    resp.headers.setdefault("X-Request-ID", rid)
    return resp


@app.middleware("http")
async def log_errors(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Middleware: catch unhandled exceptions and return a JSON 500.

    When 'DEV_ERROR_DETAILS=1', the returned JSON will include a 'trace' field
    containing the Python stack trace. This is helpful in development but should
    be disabled in production to avoid leaking internals.

    Args:
        request: Incoming FastAPI request object.
        call_next: Middleware continuation delegate.

    Returns:
        A normal response if no error occurs; otherwise a JSON 500 payload.

    Raises:
        Re-raises any exception if the error handler itself fails.
    """
    try:
        return await call_next(request)
    except Exception as exc:
        try:
            import traceback
            import os as _os

            tb = "".join(traceback.format_exc())
            print(f"[ERROR] {request.method} {request.url.path} → {exc}\n{tb}")
            from fastapi.responses import JSONResponse

            show_tb = _os.getenv("DEV_ERROR_DETAILS", "0") == "1"
            payload = {"error": str(exc)}
            if show_tb:
                payload["trace"] = tb
            return JSONResponse(payload, status_code=500)
        except Exception:
            raise


@app.on_event("startup")
def on_startup() -> None:
    """
    Application startup hook.

    Side effects:
    - Prints masked database URL and masked NEXTAUTH_SECRET to the console
      to confirm the runtime environment.
    - Prints the value of AUTH_DISABLED to make auth toggles visible at boot.
    - Initializes the database engine/metadata.
    - Runs Alembic migrations to upgrade the schema to the latest head.

    Notes:
    - '_mask_dsn' ensures no secrets are leaked when printing DB_URL.
    - Migration failures here should crash the process to avoid serving a bad state.
    """
    secret = os.getenv("NEXTAUTH_SECRET") or ""
    masked = (
        (secret[:3] + "…" + secret[-3:])
        if len(secret) > 8
        else ("set" if secret else "unset")
    )
    print(f"[DB] Using DATABASE_URL = {_mask_dsn(DB_URL)}")
    print(f"[AUTH] NEXTAUTH_SECRET = {masked}")
    print(f"[AUTH] AUTH_DISABLED = {os.getenv('AUTH_DISABLED')}")
    init_db()
    run_alembic_upgrade_head()


@app.middleware("http")
async def add_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Middleware: add conservative security headers to all HTTP responses.

    The goal is to improve baseline security without breaking existing behavior.
    - X-Content-Type-Options: Prevents MIME type sniffing.
    - X-Frame-Options: Reduces clickjacking; SAMEORIGIN is safe for dashboards.
    - Referrer-Policy: Limits cross-origin leakage of referrers.
    - Permissions-Policy: Disables sensitive device features by default.
    - Strict-Transport-Security: Optional; enabled when ENABLE_HSTS=1.

    Args:
        request: Incoming FastAPI request object.
        call_next: Middleware continuation delegate.

    Returns:
        The downstream HTTP response with headers set if missing.
    """
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault(
        "Permissions-Policy", "microphone=(), camera=(), geolocation=()"
    )
    if os.getenv("ENABLE_HSTS", "0") == "1":
        resp.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
        )
    return resp


app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(env_router)
app.include_router(owner)
app.include_router(me)
app.include_router(sched_availability)
app.include_router(sched_admin_book)
app.include_router(client_router)
app.include_router(sched_owner_appointments)
app.include_router(sched_timeoff)
app.include_router(sched_public)
app.include_router(sched_openings)
app.include_router(owner_clients)
app.include_router(owner_analytics_router)
app.include_router(owner_pricing_router)
app.include_router(agent_chat)
app.include_router(dev_inspect_router)
app.include_router(outbox_router)
app.include_router(client_email_router)
app.include_router(fun_router)


@app.get("/healthz")
def health() -> dict[str, str]:
    """
    Liveness probe endpoint (k8s/Cloud Run friendly).

    Returns:
        JSON document indicating the service is up.
    """
    return {"status": "ok"}


@app.get("/health")
def health_alt() -> dict[str, str]:
    """
    Alternative liveness endpoint for environments expecting '/health'.

    Returns:
        JSON document indicating the service is up.
    """
    return {"status": "ok"}
