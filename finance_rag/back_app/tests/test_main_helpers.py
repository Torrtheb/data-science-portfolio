# tests/test_main_helpers.py
from __future__ import annotations
import sys
import types
import asyncio
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_globals(monkeypatch):
    """
    Keep main module globals controllable between tests:
    - Reset 'retriever'
    - Provide a tiny, predictable TOOLS list
    """
    import back_app.main as main

    monkeypatch.setattr(main, "retriever", None, raising=True)

    # Minimal tool objects with .name
    class DummyTool:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(main, "TOOLS", [DummyTool("foo")], raising=True)


# ---------- _is_offtopic_reply ----------


def test_is_offtopic_reply():
    import back_app.main as main

    assert main._is_offtopic_reply(main._GUARD_REPLY) is True
    assert (
        main._is_offtopic_reply("This looks out of scope for a finance-only assistant")
        is True
    )
    assert main._is_offtopic_reply("Totally normal helpful market answer") is False
    assert main._is_offtopic_reply("") is False


# ---------- _approx_count_tokens + _approx_cost_usd ----------


def test_approx_count_tokens_uses_tiktoken_when_present(monkeypatch):
    """
    Provide a tiny fake tiktoken so _approx_count_tokens returns non-zero counts.
    """
    import back_app.main as main

    fake_tiktoken = types.SimpleNamespace()

    class FakeEnc:
        def encode(self, s: str):
            # 1 token per 3 chars (arbitrary but deterministic)
            return list(range(max(0, len(s) // 3)))

    def get_encoding(name: str):
        return FakeEnc()

    fake_tiktoken.get_encoding = get_encoding

    # Also make encoding_for_model raise to force fallback to get_encoding
    def encoding_for_model(model: str):
        raise RuntimeError("force fallback")

    fake_tiktoken.encoding_for_model = encoding_for_model

    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    pt, ct = main._approx_count_tokens(
        model="gpt-4o-mini",
        prompt_messages=[{"role": "user", "content": "Hello there"}],
        completion_text="Hi!",
    )
    # With the fake encoder, both should be non-zero and deterministic
    assert pt > 0
    assert ct > 0


def test_approx_cost_usd_reads_PRICING(monkeypatch):
    """
    Inject a fake 'back_app.core.settings' module with PRICING map.
    """
    import back_app.main as main

    # Fake PRICING object: PRICING[model] -> object with input_per_1k/output_per_1k
    model = "gpt-4o-mini"
    price_row = SimpleNamespace(input_per_1k=0.005, output_per_1k=0.015)
    fake_settings_mod = types.SimpleNamespace(PRICING={model: price_row})

    # Ensure the import inside _approx_cost_usd resolves to our fake module
    monkeypatch.setitem(sys.modules, "back_app.core.settings", fake_settings_mod)

    # 2k in, 1k out => cost = 2 * 0.005 + 1 * 0.015 = 0.025
    cost = main._approx_cost_usd(model, input_tokens=2000, output_tokens=1000)
    assert cost == pytest.approx(0.025, rel=1e-6)


# ---------- _has_useful_docs ----------


def test_has_useful_docs_true_when_score_high(monkeypatch):
    import back_app.main as main

    class FakeRetriever:
        def invoke(self, q, config=None):
            return [SimpleNamespace(metadata={"score": 0.9})]

    # Force _doc_max_score to return a “high” similarity
    fake_rag = types.SimpleNamespace(_doc_max_score=lambda docs: 0.9)
    monkeypatch.setitem(sys.modules, "back_app.llm.rag", fake_rag)
    monkeypatch.setattr(main, "retriever", FakeRetriever(), raising=True)

    ok, score = main._has_useful_docs("what is beta?")
    assert ok is True
    assert score == pytest.approx(0.9, rel=1e-6)


def test_has_useful_docs_false_when_score_low(monkeypatch):
    import back_app.main as main

    class FakeRetriever:
        def invoke(self, q, config=None):
            return [SimpleNamespace(metadata={"score": 0.1})]

    fake_rag = types.SimpleNamespace(_doc_max_score=lambda docs: 0.1)
    monkeypatch.setitem(sys.modules, "back_app.llm.rag", fake_rag)
    monkeypatch.setattr(main, "retriever", FakeRetriever(), raising=True)

    ok, score = main._has_useful_docs("some query")
    assert ok is False
    assert score == pytest.approx(0.1, rel=1e-6)


# ---------- _collect_tools ----------


def test_collect_tools_merges_local_and_mcp(monkeypatch):
    import back_app.main as main

    class DummyTool:
        def __init__(self, name):
            self.name = name

    # Replace build_langchain_tools to return two extra tools (one duplicate name)
    def fake_build():
        return [DummyTool("foo"), DummyTool("bar")]

    monkeypatch.setattr(main, "build_langchain_tools", fake_build, raising=True)
    # Ensure local TOOLS is a single “foo”
    main.TOOLS = [DummyTool("foo")]

    tools, tmap = main._collect_tools()
    names = [getattr(t, "name") for t in tools]

    # Expect *both* foos preserved, with a suffix applied to the duplicate
    assert "bar" in names
    foos = [n for n in names if n == "foo" or n.startswith("foo_")]
    assert len(foos) == 2
    assert "foo" in foos
    assert any(n.startswith("foo_") for n in foos)

    # Mapping should include bar, the original foo, and the suffixed foo entry
    assert "bar" in tmap
    assert "foo" in tmap
    assert any(k.startswith("foo_") for k in tmap.keys())


# ---------- _build_sources_from_steps_and_docs ----------


@pytest.mark.anyio
async def test_build_sources_appends_finnhub_when_stock_tool_used(monkeypatch):
    import back_app.main as main

    # No tool-sourced items returned → badge logic should still add Finnhub
    monkeypatch.setattr(main, "tools_trace_to_sources", lambda traces: [], raising=True)

    # Prepare a step that looks like an agent action calling a stock tool
    action = SimpleNamespace(tool="get_live_price", tool_input={"symbol": "AAPL"})
    steps = [(action, "42.00")]

    out = await main._build_sources_from_steps_and_docs(steps, "price?")
    assert any(item.get("id") == "tool:finnhub" for item in out)


@pytest.mark.anyio
async def test_build_sources_appends_world_bank_when_world_bank_tool_used(monkeypatch):
    import back_app.main as main

    monkeypatch.setattr(main, "tools_trace_to_sources", lambda traces: [], raising=True)
    action = SimpleNamespace(
        tool="world_bank.get_indicator", tool_input={"code": "NY.GDP.MKTP.CD"}
    )
    steps = [(action, "ok")]

    out = await main._build_sources_from_steps_and_docs(steps, "gdp?")
    assert any(item.get("id") == "tool:world_bank" for item in out)


@pytest.mark.anyio
async def test_build_sources_uses_confident_docs_when_no_tools(monkeypatch):
    import back_app.main as main
    from types import SimpleNamespace

    # No tool traces -> fall back to retriever path.
    doc1 = SimpleNamespace(metadata={"url": "https://example.com/a"})
    doc2 = SimpleNamespace(metadata={"url": "https://example.com/b"})

    class FakeRetriever:
        def invoke(self, q):  # one of the supported code paths in main.py
            return [doc1, doc2]

    monkeypatch.setattr(main, "retriever", FakeRetriever(), raising=True)

    # Pass-through confidence filter
    from back_app import main as main_mod

    monkeypatch.setattr(
        main_mod, "rag_filter_confident", lambda docs: docs, raising=False
    )

    out = await main._build_sources_from_steps_and_docs([], "docs?")

    # Accept either 'url' or 'href' because normalize_source may populate 'href'
    links = {(item.get("url") or item.get("href")) for item in out}
    assert "https://example.com/a" in links
    assert "https://example.com/b" in links


@pytest.mark.anyio
async def test_call_agent_handles_ainvoke(monkeypatch):
    import back_app.main as main

    class AgentAInvoke:
        def __init__(self):
            self.seen = None

        async def ainvoke(self, inputs, config=None):
            self.seen = (inputs, config)
            return {"output": "ok2", "intermediate_steps": []}

    ag = AgentAInvoke()
    res = await main._call_agent(ag, {"input": "yo"})
    assert isinstance(res, dict) and res.get("output") == "ok2"
    assert ag.seen is not None


# ---------- SSETokenHandler ----------


@pytest.mark.anyio
async def test_sse_token_handler_basic_flow():
    import back_app.main as main

    h = main.SSETokenHandler()
    await h.on_llm_new_token("A")
    await h.on_llm_new_token("B")
    await h.on_llm_end()

    tokens = []
    while True:
        t = await h.queue.get()
        if t is None:
            break
        tokens.append(t)

    assert tokens == ["A", "B"]
