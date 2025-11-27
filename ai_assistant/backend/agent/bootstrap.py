from __future__ import annotations
import os
import re
from dotenv import load_dotenv

load_dotenv()


def to_psycopg_dsn(url: str) -> str:
    """Normalize a SQLAlchemy Postgres URL to a psycopg-compatible DSN.

    Many libraries (e.g., SQLAlchemy) prefix Postgres URLs with a driver
    indicator such as postgresql+psycopg://. Psycopg expects
    postgresql://. This helper strips the +driver portion when present.

    Args:
        url: The input database URL, potentially containing a +driver.

    Returns:
        A Postgres DSN string using the postgresql:// scheme.
    """
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", url)


def bootstrap_checkpointer(sa_url: str) -> None:
    """Create checkpointer schema for LangGraph or LangChain-Postgres.

    The function attempts, in order, to initialize the Postgres checkpointer
    schema using:

    1) langgraph.checkpoint.postgres.PostgresSaver
    2) langchain_postgres.checkpoint.PostgresSaver

    It will call create_tables or setup depending on the installed
    version of the respective package. If neither package is available, a
    SystemExit is raised with installation guidance.

    Args:
        sa_url: A SQLAlchemy-style Postgres URL; PG_DSN env var, if set,
            takes precedence and is used directly.

    Raises:
        SystemExit: If no compatible Postgres checkpointer package is available.

    Returns:
        None. Prints status messages describing the outcome.
    """
    dsn = os.environ.get("PG_DSN") or to_psycopg_dsn(sa_url)

    try:
        from langgraph.checkpoint.postgres import PostgresSaver as LGPostgresSaver  # type: ignore

        with LGPostgresSaver.from_conn_string(dsn) as cp:
            create = getattr(cp, "create_tables", None) or getattr(cp, "setup", None)
            if callable(create):
                create()
            print("✅ LangGraph checkpointer schema ready.")
            return
    except Exception:
        pass

    try:
        from langchain_postgres.checkpoint import PostgresSaver as LCPostgresSaver  # type: ignore

        cp = LCPostgresSaver.from_conn_string(dsn)
        create = getattr(cp, "create_tables", None) or getattr(cp, "setup", None)
        if callable(create):
            create()
        print("✅ LangChain-Postgres checkpointer schema ready.")
        return
    except Exception:
        pass

    raise SystemExit(
        "No Postgres checkpointer package available. "
        "Install 'langgraph-checkpoint-postgres' or 'langchain-postgres'."
    )


def bootstrap_pgvector(sa_url: str) -> None:
    """Optionally create PGVector default schema if the package is installed.

    This pre-creates the default PGVector tables for the collection
    owner_memory using JSONB storage. If PGVector or its import path is not
    available, the function logs a warning and continues without raising.

    Args:
        sa_url: A SQLAlchemy-style Postgres URL used to connect for schema
            creation.

    Returns:
        None. Prints status messages describing the outcome.
    """
    try:
        try:
            from langchain_postgres import PGVector
        except Exception:
            from langchain_postgres.vectorstores import PGVector  # type: ignore
        vs = PGVector(
            collection_name="owner_memory",
            connection=sa_url,
            embeddings=None,
            use_jsonb=True,
        )
        create = getattr(vs, "create_default_schema", None)
        if callable(create):
            create()
            print("✅ PGVector schema ready.")
        else:
            print("ℹ️ Skipping PGVector create_default_schema (not available).")
    except Exception as e:
        print(f"⚠️ Skipping PGVector bootstrap: {e}")


def main() -> None:
    """Entry point to bootstrap Postgres checkpointer and PGVector schemas.

    Environment:
        - DATABASE_URL or BACKEND_DATABASE_URL: Source SQLAlchemy URL.
        - PG_DSN (optional): If provided, overrides URL with a psycopg DSN.

    Raises:
        SystemExit: If no database URL environment variable is set, or when no
            compatible checkpointer package is available.

    Returns:
        None. Prints a final completion message on success.
    """
    sa_url = os.environ.get("DATABASE_URL") or os.environ.get("BACKEND_DATABASE_URL")
    if not sa_url:
        raise SystemExit("DATABASE_URL (or BACKEND_DATABASE_URL) is not set.")
    bootstrap_checkpointer(sa_url)
    bootstrap_pgvector(sa_url)
    print("✅ Bootstrap complete.")


if __name__ == "__main__":
    main()
