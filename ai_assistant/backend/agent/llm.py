from __future__ import annotations
import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from agent.tool_registry import ALL_TOOLS


def bound_llm_with_tools() -> Runnable:
    """Create a chat model bound to registered agent tools.

    Model precedence (left to right):
    - AGENT_MODEL (e.g., "gpt-4o-mini" or provider-prefixed like "openai:gpt-4o-mini")
    - OPENAI_MODEL
    - default "gpt-4o-mini"

    If a provider prefix is present (e.g., "openai:…"), it is stripped for
    ChatOpenAI which expects the bare model id.

    Returns:
        Runnable chat model bound with tools and tool_choice="auto".
    """
    raw = os.getenv("AGENT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    model = raw.split(":", 1)[1] if ":" in raw else raw
    max_retries = int(os.getenv("AGENT_OPENAI_MAX_RETRIES", "3") or "3")

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_retries=max_retries,
        model_kwargs={
            "parallel_tool_calls": False,
        },
    )
    return llm.bind_tools(ALL_TOOLS, tool_choice="auto")
