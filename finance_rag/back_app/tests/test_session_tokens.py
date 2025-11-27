from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from back_app.core import db as dbm


@pytest.fixture
def client(fastapi_app):
    return TestClient(fastapi_app)


def _new_session(client: TestClient):
    res = client.post("/api/chat/sessions", json={"title": "Auth"})
    assert res.status_code == 201
    data = res.json()
    return data["sessionId"], data["token"]


def test_export_requires_valid_session_token(client):
    sid, token = _new_session(client)

    # Seed a message so export isn't empty.
    with dbm.SessionLocal() as s:
        dbm.append_message(s, sid, "user", "hello", None)

    # Missing token → 401
    res = client.get(f"/api/chat/{sid}/export.json")
    assert res.status_code == 401

    # Wrong token → 403
    sid2, token2 = _new_session(client)
    res = client.get(
        f"/api/chat/{sid}/export.json",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert res.status_code == 403

    # Correct token → 200 with data
    res = client.get(
        f"/api/chat/{sid}/export.json",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["sessionId"] == sid
    assert len(data.get("messages") or []) >= 1
