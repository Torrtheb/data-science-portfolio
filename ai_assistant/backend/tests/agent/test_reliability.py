from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def test_chatopenai_uses_retry_knob(monkeypatch):
    """Ensure AGENT_OPENAI_MAX_RETRIES is forwarded to ChatOpenAI + tool binding."""
    from agent import llm as llm_mod

    captured = {}

    class FakeLLM:
        def __init__(self, *, max_retries: int, **kwargs):
            captured["max_retries"] = max_retries
            captured["kwargs"] = kwargs

        def bind_tools(self, tools, tool_choice=None):
            captured["tool_choice"] = tool_choice
            captured["tools_len"] = len(tools)
            return self

    monkeypatch.setenv("AGENT_OPENAI_MAX_RETRIES", "7")
    monkeypatch.setattr(llm_mod, "ChatOpenAI", FakeLLM)
    # Keep tools lightweight for the binding call
    monkeypatch.setattr(llm_mod, "ALL_TOOLS", [object(), object()])

    llm = llm_mod.bound_llm_with_tools()

    assert llm is not None
    assert captured["max_retries"] == 7
    assert captured["tool_choice"] == "auto"
    assert captured["tools_len"] == 2


def test_embeddings_use_retry_knob(monkeypatch):
    """Ensure OpenAIEmbeddings picks up AGENT_OPENAI_MAX_RETRIES and plumbs through PGVector."""
    from agent import facts as facts_mod

    facts_mod._fact_store.cache_clear()

    class FakeEmbeddings:
        def __init__(self, *, model: str, max_retries: int):
            self.model = model
            self.max_retries = max_retries

    captured = {}

    class FakePGVector:
        def __init__(self, *, embeddings, **kwargs):
            captured["embeddings"] = embeddings
            captured["kwargs"] = kwargs

    monkeypatch.setenv("AGENT_OPENAI_MAX_RETRIES", "9")
    monkeypatch.setenv("PGVECTOR_URL", "postgresql://example")
    monkeypatch.setattr(facts_mod, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(facts_mod, "PGVector", FakePGVector)

    store = facts_mod._fact_store()

    assert isinstance(captured["embeddings"], FakeEmbeddings)
    assert captured["embeddings"].max_retries == 9
    # verify the store was constructed (regression guard)
    assert isinstance(store, FakePGVector)
    facts_mod._fact_store.cache_clear()


def test_llm_failure_emits_error_message(monkeypatch):
    """Simulate llm.invoke failure and verify graph returns an error AIMessage."""
    monkeypatch.setenv("AGENT_CHECKPOINTER", "memory")
    from agent import graph as graph_mod

    class RaisingLLM:
        def bind_tools(self, tools, tool_choice=None):
            return self

        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(graph_mod, "bound_llm_with_tools", lambda: RaisingLLM())

    g = graph_mod.build_graph()
    result = g.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "t", "owner_id": "o", "user_id": "u"}},
    )

    ai = result["messages"][-1]
    assert isinstance(ai, AIMessage)
    assert ai.additional_kwargs.get("event") == "error"
    assert "reach the model" in (ai.content or "")


@pytest.mark.parametrize(
    "text,expected_tool",
    [
        ("book a lesson tomorrow at 10", "book_appointment"),
        ("add time off next Friday", "add_time_off"),
    ],
)
def test_tool_hint_includes_top_tools(text: str, expected_tool: str):
    """Router hint should surface likely tools based on keyword metadata."""
    from agent import graph as graph_mod

    hint = graph_mod._tool_hint_for_text(text)
    assert hint is not None
    assert expected_tool in hint
