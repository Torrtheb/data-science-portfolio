from fastapi import APIRouter, Depends, HTTPException
import os
from app.core.auth import require_owner, TokenUser, get_current_user
from sqlalchemy import text
from app.db import get_db

router = APIRouter(prefix="/api/debug", tags=["debug"])

DEBUG_ENDPOINTS = os.getenv("DEBUG_ENDPOINTS", "0") == "1"


@router.get("/env")
def env(user: TokenUser = Depends(require_owner)):
    """Return select environment settings with sensitive values masked.

    Requires `DEBUG_ENDPOINTS=1`. Intended to quickly confirm key variables
    like LangSmith/feature flags are wired in the deployment environment.
    """
    if not DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")

    def mask(v: str | None, keep: int = 4):
        if not v:
            return False
        return v[:keep] + "…" + v[-keep:]

    return {
        "AUTH_DISABLED": os.getenv("AUTH_DISABLED"),
        "NEXTAUTH_SECRET_is_set": bool(os.getenv("NEXTAUTH_SECRET")),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT"),
        "LANGSMITH_API_KEY_masked": mask(os.getenv("LANGSMITH_API_KEY")),
        "FEATURE_AUTO_APPLY_WALLET_ON_BOOK": os.getenv(
            "FEATURE_AUTO_APPLY_WALLET_ON_BOOK", "false"
        ),
        "FEATURE_WALLET_DEPOSITS_AS_PAID": os.getenv(
            "FEATURE_WALLET_DEPOSITS_AS_PAID", "false"
        ),
    }


@router.get("/me")
def me(user: TokenUser = Depends(require_owner)):
    """Echo token-derived user fields to validate auth in debug mode."""
    if not DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "sub": user.sub,
        "email": user.email,
        "role": user.role,
        "timezone": user.timezone,
    }


@router.get("/db_check")
def db_check(user: TokenUser = Depends(require_owner), db=Depends(get_db)):
    """Lightweight DB diagnostics to help local setup.

    Returns presence of key tables, search_path, and selected auth.User columns.
    Guarded by DEBUG_ENDPOINTS and owner auth.
    """
    if not DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")

    out: dict[str, object] = {}
    try:
        sp = db.execute(text("select current_schemas(true)")).scalar()
        out["current_schemas"] = sp
    except Exception as exc:
        out["current_schemas_error"] = str(exc)

    def reg(name: str) -> object:
        try:
            return bool(
                db.execute(
                    text("select to_regclass(:n) is not null"), {"n": name}
                ).scalar()
            )
        except Exception as e:
            return f"error: {e}"

    out["tables"] = {
        "auth.User": reg('auth."User"'),
        "public.availability_rules": reg("public.availability_rules"),
        "public.client_accounts": reg("public.client_accounts"),
        "public.service_options": reg("public.service_options"),
        "public.agent_threads": reg("public.agent_threads"),
        "public.agent_messages": reg("public.agent_messages"),
    }

    try:
        ver = db.execute(text("select version_num from public.alembic_version"))
        out["alembic_version"] = [row[0] for row in ver]
    except Exception as exc:
        out["alembic_version_error"] = str(exc)

    try:
        tabs = db.execute(
            text(
                """
            select table_schema, table_name
            from information_schema.tables
            where table_schema not in ('pg_catalog','information_schema')
            order by table_schema, table_name
            """
            )
        ).all()
        out["all_tables"] = [(r.table_schema, r.table_name) for r in tabs]
    except Exception as exc:
        out["all_tables_error"] = str(exc)

    try:
        out["database"] = db.execute(text("select current_database()")).scalar()
    except Exception as exc:
        out["database_error"] = str(exc)

    try:
        cols = (
            db.execute(
                text(
                    """
            select column_name
            from information_schema.columns
            where table_schema = 'auth' and table_name = 'User'
            order by column_name
            """
                )
            )
            .scalars()
            .all()
        )
        out["auth_user_columns"] = cols
    except Exception as exc:
        out["auth_user_columns_error"] = str(exc)

    try:
        cnt = db.execute(text('select count(*) from auth."User"')).scalar()
        out["auth_user_count"] = int(cnt or 0)
    except Exception as exc:
        out["auth_user_count_error"] = str(exc)

    return out


@router.get("/me")
def me(user: TokenUser = Depends(get_current_user)):
    """Return the authenticated user's identity fields from the token."""
    return {
        "sub": user.sub,
        "email": user.email,
        "role": user.role,
        "timezone": user.timezone,
    }
