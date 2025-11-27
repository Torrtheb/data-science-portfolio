import os
from pathlib import Path
from dotenv import load_dotenv

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from collections.abc import Iterator

_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)

DB_URL = os.getenv("BACKEND_DATABASE_URL")
if not DB_URL:
    raise RuntimeError("BACKEND_DATABASE_URL not set")

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_search_path(dbapi_conn, _):
    """
    SQLAlchemy 'connect' event listener to set Postgres search_path.

    At connection time, attempts to execute:
        SET search_path TO public, auth

    This allows unqualified table names to resolve first to 'public' (your
    domain tables) and then to 'auth' (Prisma-managed auth tables), so typical
    queries work without explicit schema qualification.

    Notes:
        - This is best-effort: if the driver/DB does not support it, we swallow
          the exception and rely on server defaults.
        - Implemented for psycopg3 connections; harmless no-op on other drivers.

    Args:
        dbapi_conn: The raw DB-API connection (psycopg3 connection object).
        _: SQLAlchemy connection record (unused).
    """

    try:
        dbapi_conn.execute("SET search_path TO public, auth")
    except Exception:
        pass


SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
)
Base = declarative_base()


def get_db() -> Iterator[Session]:
    """
    FastAPI dependency that yields a SQLAlchemy Session.

    Usage in a route:
        def endpoint(db: Session = Depends(get_db)):
            ...

    Lifecycle:
        - Creates a Session from 'SessionLocal'.
        - Yields it to the caller.
        - Ensures the Session is closed on function exit.

    Notes:
        - This helper does not implicitly commit or roll back. Your application
          code or service layer should decide when to commit/rollback.
          A common pattern is to 'db.rollback()' on exceptions where you perform
          writes, or to wrap service methods in explicit transaction management.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Import ORM models to register mappers with SQLAlchemy.

    Why import-only?
        - SQLAlchemy needs model classes imported at process start so that
          mapper configuration is ready before first use.
        - We deliberately DO NOT call 'Base.metadata.create_all()' here because
          the schema is managed externally:
              * Alembic handles migrations for your domain models ('public').
              * Prisma manages the auth schema ('auth').

    Side effects:
        - Populates SQLAlchemy's mapper registry by importing 'app.models'.
    """
    import app.models
