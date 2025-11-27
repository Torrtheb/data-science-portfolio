from __future__ import annotations

"""
One-time bootstrap to create all app tables in the public schema on Neon
without relying on Alembic's online/offline nuances.

What it does:
- Connects using BACKEND_DATABASE_URL (psycopg v3 URL recommended)
- Forces search_path to public,auth for the session
- Creates all SQLAlchemy models that live in the public schema
- Ensures public.alembic_version exists and stamps it to the Alembic head

Safe to run multiple times.
"""

import os  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

# Ensure we import the same Base/models as Alembic uses
# Add backend root to sys.path so "import app" works when executing from scripts/
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import Base  # type: ignore  # noqa: E402
from app import models  # noqa: F401,E402  (register mappers)


def _load_alembic_lib():
    """Import Alembic library even when a local folder named 'alembic' exists.

    When running from the backend directory, the local migrations folder
    `backend/alembic` can shadow the installed Alembic package. Temporarily
    remove the backend dir from the front of sys.path so the real library wins.
    """
    import sys

    backend_dir = str(Path(__file__).resolve().parents[1])
    removed_front = None
    if sys.path and sys.path[0] == backend_dir:
        removed_front = sys.path.pop(0)
    try:
        from alembic.config import Config  # type: ignore
        from alembic.script import ScriptDirectory  # type: ignore

        return Config, ScriptDirectory
    finally:
        if removed_front is not None:
            sys.path.insert(0, removed_front)


def _get_head_revision() -> str:
    """Return the single current head revision id from migrations directory.

    Raises if multiple heads are present (should not be the case in this repo).
    """
    Config, ScriptDirectory = _load_alembic_lib()
    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    if not heads:
        raise RuntimeError("No Alembic heads found; check your migrations directory")
    if len(heads) > 1:
        raise RuntimeError(f"Multiple Alembic heads found: {heads}. Merge heads first.")
    return heads[0]


def _get_url() -> str:
    url = os.getenv("BACKEND_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("BACKEND_DATABASE_URL/DATABASE_URL not set")
    return url


def main() -> None:
    url = _get_url()
    engine = create_engine(url, pool_pre_ping=True)

    # Compute only the public tables from metadata
    public_tables = [
        t for t in Base.metadata.sorted_tables if (t.schema or "public") == "public"
    ]
    if not public_tables:
        raise RuntimeError(
            "No public tables discovered from metadata; check model imports"
        )

    with engine.begin() as conn:
        # Force creates into public unless explicitly schema-qualified
        conn.exec_driver_sql("SET search_path TO public, auth")

        # Create public tables
        Base.metadata.create_all(bind=conn, tables=public_tables)

        # Ensure version table exists and stamp head
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS public.alembic_version (version_num VARCHAR(128) NOT NULL)"
        )
        # Replace existing row (table contains a single row)
        conn.exec_driver_sql("TRUNCATE public.alembic_version")
        head_rev = _get_head_revision()
        # Use SQLAlchemy text() with execute() for parameter binding (exec_driver_sql expects raw SQL string)
        conn.execute(
            text("INSERT INTO public.alembic_version (version_num) VALUES (:rev)"),
            {"rev": head_rev},
        )

    # sanity output
    with Session(engine) as s:
        s.execute(text("SET search_path TO public, auth"))
        tables = (
            s.execute(
                text(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1"
                )
            )
            .scalars()
            .all()
        )
    print(f"[bootstrap] Created public tables: {tables}")
    print(f"[bootstrap] Stamped alembic_version to {_get_head_revision()}")


if __name__ == "__main__":
    main()
