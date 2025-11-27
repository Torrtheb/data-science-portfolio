from __future__ import annotations
import json
from typing import Any, Callable, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langchain_core.messages import AIMessage
from agent.graph_parts.types import AgentState


def wire_graph(
    *,
    route_or_llm: Callable[[AgentState, Optional[RunnableConfig]], dict[str, Any]],
    call_model: Callable[[AgentState, Optional[RunnableConfig]], dict[str, Any]],
    run_tools: Callable[[AgentState, Optional[RunnableConfig]], dict[str, Any]],
    post_tools: Callable[[AgentState, Optional[RunnableConfig]], dict[str, Any]],
    checkpointer: Any,
) -> Any:
    """Assemble and compile the agent LangGraph with provided node functions.

    Args
    - route_or_llm: Node that decides whether to call tools, call the model,
      or end early based on the latest message.
    - call_model: Node that calls the LLM and appends its response.
    - run_tools: Node that executes any model-requested tools and appends
      tool results.
    - post_tools: Node that interprets tool results and decides next steps
      (e.g., produce UI messages, prompt LLM again, or request more tools).
    - checkpointer: LangGraph-compatible checkpointer instance.

    Returns
    - The compiled graph object ready to run with the supplied checkpointer.

    Raises
    - None

    Behavior
    - START → router
    - router → tools if last AI message has tool_calls
    - router → END if last message is AIMessage without tool calls
    - otherwise router → llm
    - llm → (tools | END) via tools_condition
    - tools → post_tools
    - post_tools → tools when requesting tools; → llm for follow-ups; else END
    - Guards against infinite loops by deduplicating identical consecutive
      tool call requests in post_tools
    """
    g = StateGraph(AgentState)

    g.add_node("router", route_or_llm)
    g.add_node("llm", call_model)
    g.add_node("tools", run_tools)
    g.add_node("post_tools", post_tools)

    g.add_edge(START, "router")

    def _router_cond(state: AgentState) -> str:
        """Decide next step after the router node.

        Args
        - state: Current agent state.

        Returns
        - 'tools' if the last AI message requested tools; '__end__' if the last
          message is an AI message with no tool calls; otherwise 'llm'.

        Raises
        - None
        """
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        if isinstance(last, AIMessage):
            return "__end__"
        return "llm"

    g.add_conditional_edges(
        "router", _router_cond, {"tools": "tools", "llm": "llm", "__end__": END}
    )

    g.add_conditional_edges("llm", tools_condition, {"tools": "tools", "__end__": END})

    g.add_edge("tools", "post_tools")

    def _post_tools_cond(state: AgentState) -> str:
        """Decide next step after post_tools based on emitted messages.

        Args
        - state: Current agent state.

        Returns
        - 'tools' if new tool calls are requested; 'llm' for follow-up prompts;
          '__end__' to stop.

        Raises
        - None
        """
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            ai_msgs = [
                m
                for m in state["messages"]
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            ]
            if len(ai_msgs) >= 2:
                import json as _json

                a = _json.dumps(ai_msgs[-1].tool_calls, sort_keys=True)
                b = _json.dumps(ai_msgs[-2].tool_calls, sort_keys=True)
                if a == b:
                    return "__end__"
            return "tools"

        if isinstance(last, AIMessage):
            text = last.content or ""
            if isinstance(text, str) and text.startswith(
                (
                    "PENDING_CLIENT:",
                    "PENDING_SLOTS:",
                    "PENDING_EMAIL_SEND:",
                    "UI:EMAIL_DRAFT:",
                    "PENDING_CHOICES:",
                    "PENDING_IDENTITY:",
                    "PENDING_OVERLAP_AT:",
                )
            ):
                if isinstance(text, str) and text.startswith("UI:EMAIL_DRAFT:"):
                    return "__end__"
                return "llm"
            return "__end__"

        if isinstance(last, AIMessage) and isinstance(
            getattr(last, "content", None), str
        ):
            s = last.content.strip()
            if s.startswith("{"):
                try:
                    j = json.loads(s)
                    if isinstance(j, dict) and j.get("marker") == "email_draft":
                        return "__end__"
                except Exception:
                    pass
        return "llm"

    g.add_conditional_edges(
        "post_tools", _post_tools_cond, {"tools": "tools", "llm": "llm", "__end__": END}
    )

    return g.compile(checkpointer=checkpointer)
