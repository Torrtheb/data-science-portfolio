from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.graph_parts.router import route_time_off_status


def test_route_time_off_status_triggers_on_next_time_off_question():
    msg = HumanMessage(content="When is my next time off?")
    res = route_time_off_status(msg)
    assert res is not None
    sys_msgs = res.get("messages") or []
    assert any(
        "next_time_off" in getattr(m, "content", "")
        for m in sys_msgs
        if hasattr(m, "content")
    )


def test_route_time_off_status_ignores_add_requests():
    msg = HumanMessage(content="Add time off next Friday")
    assert route_time_off_status(msg) is None
