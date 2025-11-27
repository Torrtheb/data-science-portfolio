# back_app/tests/test_utils.py
import types
import pytest
from types import SimpleNamespace

# Adjust this import if your path differs (e.g., `from utils import ...`)
from back_app.utils.utils import (
    to_lc_messages,
    _TokenBucket,
    parse_percent,
    _emit_analytics,
    _require_admin,
)

# For type checks
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# -----------------------------
# to_lc_messages
# -----------------------------
def test_to_lc_messages_basic():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "system", "content": "be concise"},
        {"role": "tool", "content": '{"ok":true}', "tool_call_id": "call-123"},
        {"role": "weird", "content": "ignore me"},
    ]
    out = to_lc_messages(history)

    assert isinstance(out[0], HumanMessage) and out[0].content == "hi"
    assert isinstance(out[1], AIMessage) and out[1].content == "hello!"
    assert isinstance(out[2], SystemMessage) and out[2].content == "be concise"
    assert isinstance(out[3], ToolMessage) and out[3].content == '{"ok":true}'
    # ToolMessage should carry through tool_call_id if present
    assert getattr(out[3], "tool_call_id", None) == "call-123"
    # Unknown roles are ignored
    assert len(out) == 4


# -----------------------------
# _TokenBucket
# -----------------------------
def test_token_bucket_allows_and_refills(monkeypatch):
    t = {"now": 1_000.0}

    def fake_monotonic():
        return t["now"]

    # Patch the module that actually defines _TokenBucket
    import back_app.utils.utils as utils_mod
    monkeypatch.setattr(utils_mod.time, "monotonic", fake_monotonic)

    from back_app.utils.utils import _TokenBucket  # keep public import for the class
    tb = _TokenBucket(rate_per_sec=2.0, burst=3.0)
    key = "ip:1.2.3.4"

    assert tb.allow(key) is True
    assert tb.allow(key) is True
    assert tb.allow(key) is True
    assert tb.allow(key) is False

    t["now"] += 0.5
    assert tb.allow(key) is True
    assert tb.allow(key) is False

    t["now"] += 2.0
    assert tb.allow(key) is True
    assert tb.allow(key) is True
    assert tb.allow(key) is True
    assert tb.allow(key) is False



def test_token_bucket_min_clamps():
    # Ensure clamping (rate >= 0.1, burst >= 1.0)
    tb = _TokenBucket(rate_per_sec=0.0, burst=0.0)
    assert tb.rate == 0.1
    assert tb.burst == 1.0


# -----------------------------
# parse_percent
# -----------------------------
@pytest.mark.parametrize(
    "inp,expected,use_approx",
    [
        ("7%", 7.0, False),
        (" 0.5% ", 0.5, False),
        ("0.07", 7.0, True),   # fraction → percent points
        (0.07, 7.0, True),     # fraction → percent points
        ("1", 100.0, False),   # <-- update this
        (1.0, 100.0, True),
        ("7", 7.0, False),
        (7, 7.0, False),
        (-5, -5.0, False),
        (None, None, False),
        ("", None, False),
        (" not-a-number ", None, False),
    ],
)
def test_parse_percent_variants(inp, expected, use_approx):
    from back_app.utils.utils import parse_percent
    out = parse_percent(inp)
    if expected is None:
        assert out is None
    elif use_approx:
        assert out == pytest.approx(expected, rel=0, abs=1e-9)
    else:
        assert out == expected



def test_parse_percent_empty_override():
    assert parse_percent(None, empty_returns=0.0) == 0.0
    assert parse_percent("   ", empty_returns=-1.0) == -1.0


# -----------------------------
# _emit_analytics (async)
# -----------------------------
@pytest.mark.anyio
async def test_emit_analytics_success(monkeypatch):
    monkeypatch.setenv("API_BASE", "http://api.test")
    calls = {"post": []}

    class DummyClient:
        def __init__(self, timeout=None):  # match signature
            self.timeout = timeout
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        async def post(self, url, json):
            calls["post"].append((url, json))
            return SimpleNamespace(status_code=204, text="")

    import back_app.utils.utils as utils_mod
    monkeypatch.setattr(utils_mod.httpx, "AsyncClient", lambda timeout=None: DummyClient())

    from back_app.utils.utils import _emit_analytics
    await _emit_analytics({"event": "ok"})
    assert calls["post"] == [("http://api.test/api/analytics/ingest", {"event": "ok"})]


@pytest.mark.anyio
async def test_emit_analytics_non_2xx_logs(monkeypatch):
    monkeypatch.setenv("API_BASE", "http://api.test")
    warnings = []

    class DummyClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        async def post(self, url, json):
            return SimpleNamespace(status_code=500, text="oops")

    from back_app.utils.utils import _emit_analytics

    monkeypatch.setattr(_emit_analytics.httpx, "AsyncClient", lambda timeout=None: DummyClient())
    monkeypatch.setattr(_emit_analytics.logger, "warning", lambda *args, **kwargs: warnings.append((args, kwargs)))
    await _emit_analytics({"event": "bad"})
    assert any("rejected" in (args[0] if args else "") for args, _ in warnings)


@pytest.mark.anyio
async def test_emit_analytics_non_2xx_logs(monkeypatch):
    monkeypatch.setenv("API_BASE", "http://api.test")
    warnings = []

    class DummyClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        async def post(self, url, json):
            return SimpleNamespace(status_code=500, text="oops")

    import back_app.utils.utils as utils_mod
    # Patch on the module, not on the function
    monkeypatch.setattr(utils_mod.httpx, "AsyncClient", lambda timeout=None: DummyClient())
    monkeypatch.setattr(utils_mod.logger, "warning", lambda *args, **kwargs: warnings.append((args, kwargs)))

    from back_app.utils.utils import _emit_analytics
    await _emit_analytics({"event": "bad"})

    assert any("rejected" in (args[0] if args else "") for args, _ in warnings)


# -----------------------------
# _require_admin
# -----------------------------
def test_require_admin_no_key_allows(monkeypatch):
    # When ADMIN_KEY is unset, it's a no-op
    monkeypatch.delenv("ADMIN_KEY", raising=False)

    class DummyReq:
        def __init__(self):
            self.headers = {}

    _require_admin(DummyReq())  # should not raise


def test_require_admin_enforces_header(monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "sekret")

    class DummyReq:
        def __init__(self, headers):
            self.headers = headers

    # Wrong / missing header → 403
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        _require_admin(DummyReq(headers={}))
    assert ei.value.status_code == 403

    with pytest.raises(HTTPException) as ei:
        _require_admin(DummyReq(headers={"X-Admin-Key": "nope"}))
    assert ei.value.status_code == 403

    # Correct header → ok
    _require_admin(DummyReq(headers={"X-Admin-Key": "sekret"}))
