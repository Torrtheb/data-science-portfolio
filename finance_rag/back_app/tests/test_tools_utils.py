# tests/test_tools_utils.py
from __future__ import annotations

import json
from types import SimpleNamespace
import pytest

# Import the file under test
import back_app.utils.tools_utils as tools_utils


# -----------------------------
# _tool_schema
# -----------------------------
def test_tool_schema_prefers_pydantic_v2():
    class V2Args:
        def model_json_schema(self):
            return {"type": "object", "properties": {"q": {"type": "string"}}}

    class Tool:
        args_schema = V2Args()

    out = tools_utils._tool_schema(Tool())
    assert out == {"type": "object", "properties": {"q": {"type": "string"}}}


def test_tool_schema_falls_back_to_pydantic_v1():
    class V1Args:
        def schema(self):
            return {"type": "object", "properties": {"n": {"type": "integer"}}}

    class Tool:
        args_schema = V1Args()

    out = tools_utils._tool_schema(Tool())
    assert out == {"type": "object", "properties": {"n": {"type": "integer"}}}


def test_tool_schema_no_args_returns_empty():
    class Tool:
        pass

    out = tools_utils._tool_schema(Tool())
    assert out == {}


# -----------------------------
# _call_tool
# -----------------------------
@pytest.mark.anyio
async def test_call_tool_prefers_coroutine_over_func():
    calls = {"coro": 0, "func": 0}

    async def coro(**kwargs):
        calls["coro"] += 1
        return {"via": "coroutine", "kwargs": kwargs}

    def func(**kwargs):
        calls["func"] += 1
        return {"via": "func", "kwargs": kwargs}

    t = SimpleNamespace(coroutine=coro, func=func)
    res = await tools_utils._call_tool(t, {"x": 1})
    assert res["via"] == "coroutine"
    assert calls == {"coro": 1, "func": 0}


@pytest.mark.anyio
async def test_call_tool_uses_func_when_present():
    def func(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    t = SimpleNamespace(func=func)
    res = await tools_utils._call_tool(t, {"a": 2})
    assert res == {"ok": True, "kwargs": {"a": 2}}


@pytest.mark.anyio
async def test_call_tool_uses_ainvoke_when_available():
    async def ainvoke(args):
        return {"via": "ainvoke", "args": args}

    t = SimpleNamespace(ainvoke=ainvoke)
    res = await tools_utils._call_tool(t, {"k": "v"})
    assert res == {"via": "ainvoke", "args": {"k": "v"}}


@pytest.mark.anyio
async def test_call_tool_uses_invoke_when_available():
    def invoke(args):
        return {"via": "invoke", "args": args}

    t = SimpleNamespace(invoke=invoke)
    res = await tools_utils._call_tool(t, {"p": 42})
    assert res == {"via": "invoke", "args": {"p": 42}}


@pytest.mark.anyio
async def test_call_tool_no_entry_raises():
    with pytest.raises(RuntimeError):
        await tools_utils._call_tool(SimpleNamespace(), {"x": 1})


# -----------------------------
# _collect_tool_markdown
# -----------------------------
def test_collect_tool_markdown_basic(monkeypatch):
    # Patch helpers inside the module under test to make output deterministic
    monkeypatch.setattr(tools_utils, "_clean_title", lambda s: s.strip())
    monkeypatch.setattr(tools_utils, "_prettify_provider", lambda s: s.strip().title())
    monkeypatch.setattr(
        tools_utils,
        "_to_iso_day",
        lambda x: "2025-01-02" if x == "t1" else "2025-01-01",
    )
    monkeypatch.setattr(
        tools_utils,
        "_pretty_date",
        lambda x: {"2025-01-02": "Jan 2, 2025", "2025-01-01": "Jan 1, 2025"}.get(x, ""),
    )

    steps = [
        (
            SimpleNamespace(tool="get_company_news"),
            json.dumps(
                {
                    "news": [
                        {
                            "headline": "Apple hits record",
                            "source": "reuters",
                            "datetime": "t2",
                        },
                        {
                            "title": "NVIDIA unveils new GPU",
                            "source": "the verge",
                            "datetime": "t1",
                        },
                    ]
                }
            ),
        )
    ]

    md = tools_utils._collect_tool_markdown(steps)
    assert "## 🗞️ News Headlines:" in md
    assert "- NVIDIA unveils new GPU" in md
    assert "- Apple hits record" in md


def test_collect_tool_markdown_returns_empty_on_no_items():
    steps = [{"action": {"tool": "x"}, "observation": {"foo": "bar"}}]
    md = tools_utils._collect_tool_markdown(steps)
    assert md == ""


# -----------------------------
# _looks_toolable
# -----------------------------
@pytest.mark.parametrize(
    "q,expected",
    [
        ("What's AAPL price today?", True),
        ("quote for BNS.TO please", True),
        ("price of happiness", False),
        ("how to compute CAGR over 10y", True),
        ("NPV of cash flows", True),
        ("", False),
    ],
)
def test_looks_toolable(q, expected):
    assert tools_utils._looks_toolable(q) is expected


# -----------------------------
# _is_finance_query
# -----------------------------


@pytest.mark.parametrize(
    "q,expected",
    [
        ("What's AAPL price today?", True),  # ticker
        ("How do I invest in ETFs?", True),  # investing keyword
        ("Should I increase my 401k contributions?", True),  # retirement / 401k
        ("Explain compound interest and inflation", True),  # core finance concepts
        ("What is GDP growth for Canada?", True),  # macro keyword
        ("Tell me a joke about cats", False),
        ("Explain photosynthesis", False),
        ("How to cook pasta?", False),
        ("", False),
    ],
)
def test_is_finance_query(q, expected):
    assert tools_utils._is_finance_query(q) is expected


# -----------------------------
# _extract_tool_events
# -----------------------------
def test_extract_tool_events_various():
    steps = [
        (SimpleNamespace(tool="get_company_news"), {"ok": True, "elapsed_ms": 123}),
        {
            "action": {"tool": "get_price"},
            "observation": json.dumps({"error": "rate limited"}),
        },
        (SimpleNamespace(tool="screen_equities"), "plain text"),
    ]

    out = tools_utils._extract_tool_events(steps, session_id="abc123")
    assert len(out) == 3

    e0 = out[0]
    assert e0["tool_name"] == "get_company_news"
    assert e0["ok"] is True
    assert e0["latency_ms"] == 123

    e1 = out[1]
    assert e1["tool_name"] == "get_price"
    assert e1["ok"] is False
    assert e1["error"] == "rate limited"

    e2 = out[2]
    assert e2["tool_name"] == "screen_equities"
    assert e2["ok"] is True
