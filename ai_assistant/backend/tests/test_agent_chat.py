from __future__ import annotations
import os

# Ensure auth module imports don't explode during tests
os.environ.setdefault("AUTH_DISABLED", "1")
os.environ.setdefault("DEV_FAKE_OWNER_ID", "owner-test")
os.environ.setdefault("NEXTAUTH_SECRET", "test-secret")

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from routers import agent_chat
from app.core.auth import TokenUser


class _NoDB:
    """Dummy DB that should never be touched in these fast-path tests."""

    def __getattr__(self, name):
        raise AssertionError(f"DB should not be touched ({name})")


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(agent_chat.router)

    app.dependency_overrides[agent_chat.require_owner] = lambda: TokenUser(
        sub="owner-test"
    )

    async def _fake_db():
        yield _NoDB()

    app.dependency_overrides[agent_chat.get_db] = _fake_db
    return app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_CHAT_PER_MIN", raising=False)
    agent_chat._CHAT_RL = {}
    app = _make_app()
    with TestClient(app) as c:
        yield c


def test_chat_rejects_bad_session_uuid(client, monkeypatch):
    resp = client.get("/api/agent/chat", params={"q": "hi", "session": "not-a-uuid"})
    assert resp.status_code == 400
    assert "session" in resp.text.lower()


def test_chat_rate_limit_blocks_after_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_MIN", "1")
    agent_chat._CHAT_RL = {}

    app = _make_app()
    with TestClient(app) as c:
        first = c.get("/api/agent/chat", params={"q": "hi", "session": "bad"})
        assert first.status_code == 400  # invalid session still counts against limit

        second = c.get("/api/agent/chat", params={"q": "again", "session": "bad"})
        assert second.status_code == 429
        assert "rate limit" in second.text.lower()
