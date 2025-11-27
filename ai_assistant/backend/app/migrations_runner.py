from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from sqlalchemy.engine import Connection

from app.db import engine, DB_URL


def _load_alembic_lib():
    """
    Import Alembic's 'Config' class and 'command' module even if a local
    'backend/alembic/' folder is present.

    When running from the backend directory, the local migrations folder
    ('backend/alembic') can shadow the installed Alembic package on 'sys.path'.
    This function temporarily removes the backend directory from the *front* of
    'sys.path' to ensure the genuine Alembic library is imported.

    Returns:
        A tuple '(Config, command)' where:
            - 'Config' is Alembic's configuration class type.
            - 'command' is Alembic's command module (object with 'upgrade', etc.).
    """
    import sys

    backend_dir = str(Path(__file__).resolve().parents[1])
    removed_front = None
    if sys.path and sys.path[0] == backend_dir:
        removed_front = sys.path.pop(0)
    try:
        from alembic.config import Config  # type: ignore
        from alembic import command  # type: ignore

        return Config, command
    finally:
        if removed_front is not None:
            sys.path.insert(0, removed_front)


def _mask_dsn(url: Optional[str]) -> str:
    """
    Produce a safe-to-log rendering of a database URL.

    Preserves the scheme, user (without password), host, optional port, and
    database name, while omitting sensitive details.

    Args:
        url: The raw database URL or None.

    Returns:
        A masked string suitable for logs (e.g., 'postgresql://user@host:5432/db').
    """
    if not url:
        return "unset"
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        user = parsed.username or ""
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        db = parsed.path.lstrip("/")
        user_part = f"{user}@" if user else ""
        return f"{parsed.scheme}://{user_part}{host}{port}/{db}"
    except Exception:
        return "masked"


def _prisma_ready(conn: Connection) -> bool:
    """
    Check whether Prisma's auth schema is present (auth.\"User\" exists).

    Uses Postgres 'to_regclass('auth.User')' to detect the existence of the table.

    Args:
        conn: An open SQLAlchemy connection.

    Returns:
        True if the auth.\"User\" table exists; False otherwise.
    """
    try:
        res = conn.exec_driver_sql(
            "SELECT to_regclass('auth.User') IS NOT NULL"
        ).scalar()
        return bool(res)
    except Exception:
        return False


def _ensure_search_path(conn: Connection) -> None:
    """
    Set the Postgres search path so unqualified names resolve to 'public, auth'.

    This mirrors the runtime behavior elsewhere in the app and allows queries to
    see both your domain tables ('public') and Prisma's auth tables ('auth').

    Args:
        conn: An open SQLAlchemy connection.
    """
    try:
        conn.exec_driver_sql("SET search_path TO public, auth")
    except Exception:
        pass


def _maybe_create_version_table(conn: Connection) -> None:
    """
    Ensure 'public.alembic_version' exists and supports long revision IDs.

    This block is idempotent and safe to run multiple times. It creates the table
    if missing or widens 'version_num' to VARCHAR(128) if present.

    Args:
        conn: An open SQLAlchemy connection.
    """
    try:
        conn.exec_driver_sql(
            """
            DO $$
            BEGIN
              IF to_regclass('public.alembic_version') IS NULL THEN
                CREATE TABLE public.alembic_version (version_num VARCHAR(128) NOT NULL);
              ELSE
                BEGIN
                  ALTER TABLE public.alembic_version
                    ALTER COLUMN version_num TYPE VARCHAR(128);
                EXCEPTION WHEN undefined_table THEN
                  NULL;
                END;
              END IF;
            END$$;
            """
        )
    except Exception:
        pass


def run_alembic_upgrade_head() -> None:
    """
    Run 'alembic upgrade head' using the configured database URL.

    Behavior:
        - Skips migrations if 'RUN_DB_MIGRATIONS=0'.
        - Prints the masked database URL for visibility.
        - Opens a connection to:
            * Set search_path to 'public, auth'.
            * Ensure the Alembic version table exists/is compatible.
            * Wait (up to 'RUN_DB_MIGRATE_WAIT_SECS', default 60s) for Prisma's
              'auth' schema ('auth."User"') to appear in greenfield environments.
        - Invokes Alembic's 'command.upgrade(cfg, "head")'.
        - Logs a warning on failure instead of crashing the process.

    Env vars:
        RUN_DB_MIGRATIONS: "1" (default) to run, "0" to skip.
        RUN_DB_MIGRATE_WAIT_SECS: integer seconds to wait for Prisma (default 60).

    Returns:
        None
    """
    if os.getenv("RUN_DB_MIGRATIONS", "1") != "1":
        print("[migrate] Skipping migrations (RUN_DB_MIGRATIONS=0)")
        return

    wait_secs = int(os.getenv("RUN_DB_MIGRATE_WAIT_SECS", "60"))
    deadline = time.time() + wait_secs

    print(f"[migrate] DB = {_mask_dsn(DB_URL)}")

    Config, command = _load_alembic_lib()

    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", DB_URL)

    skip_prisma_wait = (
        os.getenv("SKIP_PRISMA_WAIT", "0") == "1"
        or os.getenv("AUTH_DISABLED", "0") == "1"
    )

    with engine.connect() as conn:
        _ensure_search_path(conn)
        _maybe_create_version_table(conn)

        if skip_prisma_wait:
            print(
                "[migrate] Skipping Prisma auth schema wait (SKIP_PRISMA_WAIT=1 or AUTH_DISABLED=1)"
            )
        else:
            while time.time() < deadline:
                if _prisma_ready(conn):
                    break
                print("[migrate] Waiting for Prisma auth schema (auth.User)…")
                time.sleep(3)

    try:
        command.upgrade(cfg, "head")
        print("[migrate] Alembic upgrade head complete")
    except Exception as exc:
        print(f"[migrate] WARNING: alembic upgrade failed: {exc}")
