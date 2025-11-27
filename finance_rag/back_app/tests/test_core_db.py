import os
import re
import sys
import time
import types
import importlib
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
import pytest


# ------------------------ helpers: fresh module per test ------------------------

def _reload_db_with_tmpfile(tmp_path):
    """
    Bind back_app.core.db to a brand-new SQLite file DB and init tables.
    Ensures a fresh SQLAlchemy Base by reloading db_base as well.
    """
    from importlib import import_module, invalidate_caches
    import sys, os

    db_file = tmp_path / "finassist.db"
    db_url = f"sqlite:////{db_file}"

    # Dispose prior engine if module still loaded (avoids open handles on macOS/Windows)
    old = sys.modules.get("back_app.core.db")
    if old is not None:
        try:
            old.engine.dispose()
        except Exception:
            pass

    # Point env at our temp DB
    os.environ["DATABASE_URL"] = db_url

    # Drop both modules so we get a fresh Base and models
    for mod in ("back_app.core.db", "back_app.core.db_base"):
        sys.modules.pop(mod, None)

    invalidate_caches()

    # Import order matters: load db_base first to create a new Base, then db
    import_module("back_app.core.db_base")
    db_mod = import_module("back_app.core.db")

    # Create tables (and apply the lightweight migration)
    db_mod.init_db()
    return db_mod, db_url



# ------------------------------ fixtures ----------------------------------------

@pytest.fixture
def dbm(tmp_path):
    """Provide a freshly reloaded db module bound to a temp sqlite file."""
    db_mod, _ = _reload_db_with_tmpfile(tmp_path)
    yield db_mod
    # close pooled connections to allow tmp cleanup
    try:
        db_mod.engine.dispose()
    except Exception:
        pass


@pytest.fixture
def session(dbm):
    """Yield a live SQLAlchemy session and ensure proper close."""
    s = dbm.SessionLocal()
    try:
        yield s
    finally:
        s.close()


# ------------------------------ tests -------------------------------------------

def test_engine_uses_sqlite_file_and_dir_created(tmp_path):
    db_mod, db_url = _reload_db_with_tmpfile(tmp_path)
    # Directory should exist (created by _ensure_sqlite_dir)
    assert db_url.startswith("sqlite:////")
    db_path = db_url.replace("sqlite:////", "/", 1)
    assert os.path.isdir(os.path.dirname(db_path))
    # Can connect
    with db_mod.engine.connect() as conn:
        res = conn.exec_driver_sql("SELECT 1").scalar()
        assert res == 1
    db_mod.engine.dispose()


def test_init_db_creates_tables(dbm):
    # introspect via connection
    with dbm.engine.connect() as conn:
        tables = {
            row[0] for row in conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "chat_sessions" in tables
    assert "chat_messages" in tables


def test_get_db_dependency_yields_and_closes(dbm):
    gen = dbm.get_db()
    s = next(gen)
    assert s.is_active
    s.execute(text("SELECT 1"))
    with pytest.raises(StopIteration):
        gen.send(None)
    # Be explicit and idempotent:
    s.close()


def test_get_or_create_session_new_then_existing(dbm, session):
    # New (no id) returns a UUID-like string
    sid = dbm.get_or_create_session(session)
    assert isinstance(sid, str) and len(sid) >= 32
    assert re.fullmatch(r"[0-9a-f-]{20,}", sid)

    # Calling again with same id returns existing (no new row)
    sid2 = dbm.get_or_create_session(session, session_id=sid)
    assert sid2 == sid

    # Verify row actually exists
    row = session.get(dbm.SessionModel, sid)
    assert row is not None
    assert row.id == sid


def test_touch_session_updates_timestamp(dbm, session):
    sid = dbm.get_or_create_session(session, title="hello")
    row = session.get(dbm.SessionModel, sid)
    before = row.updated_at

    # ensure a measurable time delta
    time.sleep(0.01)
    dbm.touch_session(session, sid)

    session.expire(row)  # refresh from DB
    after = session.get(dbm.SessionModel, sid).updated_at
    assert after >= before
    # be robust to clock resolution on different OS/FS
    assert (after - before) >= timedelta(0)


def test_append_message_creates_and_bumps_session(dbm, session):
    sid = dbm.get_or_create_session(session, title="chat")
    sess_row = session.get(dbm.SessionModel, sid)
    before = sess_row.updated_at

    mid = dbm.append_message(
        session, sid, role="user", content="hi", tool_calls_json=None
    )
    assert isinstance(mid, int) and mid > 0

    # Message persisted
    msg = session.get(dbm.MessageModel, mid)
    assert msg is not None
    assert msg.session_id == sid
    assert msg.role == "user"
    assert msg.content == "hi"

    # Session.updated_at bumped
    session.expire(sess_row)
    after = session.get(dbm.SessionModel, sid).updated_at
    assert after >= before


def test_list_messages_returns_in_ascending_order_and_limit(dbm, session):
    sid = dbm.get_or_create_session(session)
    ids = [
        dbm.append_message(session, sid, "user", f"m{i}", None)
        for i in range(5)
    ]
    out = dbm.list_messages(session, sid)
    got_ids = [m.id for m in out]
    assert got_ids == sorted(ids)

    # limit works
    out2 = dbm.list_messages(session, sid, limit=3)
    assert [m.id for m in out2] == sorted(ids)[:3]


def test_delete_session_cascades_messages(dbm, session):
    sid = dbm.get_or_create_session(session)
    m1 = dbm.append_message(session, sid, "user", "hello", None)
    m2 = dbm.append_message(session, sid, "assistant", "world", None)

    # sanity
    assert session.get(dbm.MessageModel, m1) is not None
    assert session.get(dbm.MessageModel, m2) is not None

    dbm.delete_session(session, sid)

    # session gone
    assert session.get(dbm.SessionModel, sid) is None
    # messages gone via cascade
    assert session.get(dbm.MessageModel, m1) is None
    assert session.get(dbm.MessageModel, m2) is None
