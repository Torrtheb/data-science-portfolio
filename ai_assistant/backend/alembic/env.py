from __future__ import annotations

import os
import sys
from pathlib import Path
from logging.config import fileConfig

THIS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv

_env_path = BACKEND_ROOT / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from app.db import Base
from app import models

target_metadata = Base.metadata

PRISMA_TABLES = {
    "User",
    "Account",
    "Session",
    "VerificationToken",
    "_prisma_migrations",
}
PRISMA_SCHEMA = "auth"


def include_object(obj, name, type_, reflected, compare_to):
    """Filter objects Alembic considers during autogenerate.

    Args:
        obj: The SQLAlchemy object being inspected (table/index/constraint).
        name: The object's name.
        type_: Kind of object (e.g., "table", "index").
        reflected: Whether the object was reflected from the database.
        compare_to: The target object to compare to (if any).

    Returns:
        True if the object should be included in migrations; False to exclude.

    Raises:
        None
    """
    owner_table = None
    owner_schema = None
    owner_name = None

    if type_ == "table":
        owner_table = obj
    elif type_ in (
        "index",
        "unique_constraint",
        "foreign_key_constraint",
        "check_constraint",
    ):
        owner_table = getattr(obj, "table", None)

    if owner_table is not None:
        owner_schema = getattr(owner_table, "schema", None)
        owner_name = getattr(owner_table, "name", None)

    # 1) Skip anything in the Prisma schema entirely
    if owner_schema == PRISMA_SCHEMA or getattr(obj, "schema", None) == PRISMA_SCHEMA:
        return False

    # 2) Skip specific Prisma tables by name (defensive: in case they're in public)
    if type_ == "table" and name in PRISMA_TABLES:
        return False
    if (
        type_
        in ("index", "unique_constraint", "foreign_key_constraint", "check_constraint")
        and owner_name in PRISMA_TABLES
    ):
        return False

    return True


def _include_object(object_, name, type_, reflected, compare_to):
    """Additional include filter for specific tables used by vector stores.

    Args:
        object_: SQLAlchemy object being inspected.
        name: Object name.
        type_: Kind of object (e.g., "table", "index").
        reflected: Whether the object was reflected from the database.
        compare_to: Target object for comparison (unused).

    Returns:
        False for known vector-store or checkpoint tables; True otherwise.

    Raises:
        None
    """
    if name is None:
        return True
    if name.startswith("langchain_pg_"):
        return False
    if name in {
        "agent_messages",
        "agent_threads",
        "checkpoints",
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoint_migrations",
    }:
        return False
    return True


def get_url() -> str:
    """Resolve the database URL from environment variables.

    Checks 'BACKEND_DATABASE_URL' and falls back to 'DATABASE_URL'.

    Returns:
        The database URL string.

    Raises:
        RuntimeError: When neither environment variable is set.
    """
    url = os.getenv("BACKEND_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("BACKEND_DATABASE_URL/DATABASE_URL not set")
    return url


def _print_db_target(url: str) -> None:
    """Print a masked database target for operator visibility.

    Args:
        url: Full database URL.

    Returns:
        None

    Raises:
        None
    """
    try:
        from urllib.parse import urlparse

        u = urlparse(url)
        safe = f"{u.scheme}://{u.username or ''}@{u.hostname}:{u.port}/{u.path.lstrip('/')}"
        print(f"[alembic] Connecting to: {safe}", flush=True)
    except Exception:
        print("[alembic] Connecting to (could not parse)")


def run_migrations_offline() -> None:
    """Run Alembic migrations in offline mode.

    Configures the context with a DB URL and emits SQL without creating an
    actual DB connection.

    Returns:
        None

    Raises:
        None
    """
    url = get_url()
    _print_db_target(url)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
        include_schemas=True,
        version_table="alembic_version",
        version_table_schema="public",
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run Alembic migrations in online mode.

    Establishes a DB connection using the resolved URL, applies search_path
    and version-table safeguards, and executes migrations within a transaction.

    Returns:
        None

    Raises:
        None
    """
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    _print_db_target(configuration["sqlalchemy.url"])

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        try:
            connection.exec_driver_sql("SET search_path TO public, auth")
        except Exception:
            pass

        try:
            connection.exec_driver_sql(
                """
                DO $$
                BEGIN
                  IF to_regclass('public.alembic_version') IS NULL THEN
                    CREATE TABLE public.alembic_version (version_num VARCHAR(128) NOT NULL);
                  ELSE
                    IF to_regclass('public.alembic_version') IS NOT NULL THEN
                      BEGIN
                        ALTER TABLE public.alembic_version
                          ALTER COLUMN version_num TYPE VARCHAR(128);
                      EXCEPTION WHEN undefined_table THEN
                        NULL;
                      END;
                    END IF;
                  END IF;
                END$$;
                """
            )
        except Exception:
            pass

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=True,
            version_table="alembic_version",
            version_table_schema="public",
            include_object=_include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
