from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(fastapi_app):
    return TestClient(fastapi_app)


def test_chat_non_finance_query_triggers_guard_and_skips_agent(monkeypatch, client):
    """
    /api/chat should short-circuit non-finance queries using the guard reply
    without calling the agent (LLM).
    """
    import back_app.main as main

    # Ensure agent placeholders are present to avoid 503 during test.
    main.rag_agent_chain = object()
    main.rag_agent_chain_stream = object()

    # Force domain guard to treat this query as non-finance.
    monkeypatch.setattr(main, "_is_finance_query", lambda q: False, raising=True)

    # If the agent is ever called, fail the test.
    def _boom(*args, **kwargs):
        raise AssertionError("Agent should not be called for non-finance queries")

    monkeypatch.setattr(main, "_call_agent", _boom, raising=True)

    body = {"messages": [{"role": "user", "content": "tell me a joke about cats"}]}
    # Obtain a session + token first (required for chat/export).
    ses = client.post("/api/chat/sessions", json={"title": "Chat"}).json()
    headers = {
        "Authorization": f"Bearer {ses['token']}",
        "X-Session-Id": ses["sessionId"],
    }

    res = client.post("/api/chat", json=body, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["text"] == main._GUARD_REPLY
    assert data["sources"] == []
    usage = data.get("usage") or {}
    assert usage.get("model") == "guard"
    assert usage.get("synthetic", True) is True
