from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from agent.graph_parts import post_tools
from agent.graph_parts.router import route_holiday_intents


def test_next_holiday_skips_past_dates(monkeypatch):
    """Ensure past holidays are ignored when picking the next holiday."""
    monkeypatch.setitem(
        post_tools.__dict__, "_today_owner_date", lambda: datetime(2025, 11, 25).date()
    )
    assert post_tools._today_owner_date() == datetime(2025, 11, 25).date()
    holidays = [
        {"date": "2025-01-01", "name": "New Year's Day"},
        {"date": "2025-12-25", "name": "Christmas Day"},
    ]
    today = post_tools._today_owner_date()
    manual = []
    for h in holidays:
        manual.append((h["date"], datetime.fromisoformat(h["date"]).date() >= today))
    assert manual[-1][1] is True  # future date should be detected as upcoming
    last = ToolMessage(
        content={"holidays": holidays},
        name="get_public_holidays",
        tool_call_id="call_holidays",
    )
    marker = AIMessage(
        content='PENDING_HOLIDAY_SCOPE:{"mode": "next", "scope": "week", "which": "this"}'
    )
    out = post_tools.handle_get_public_holidays(last, [marker, last])
    msgs = (out or {}).get("messages") or []
    assert msgs, "Expected a response message"
    assert "2025-12-25" in (msgs[0].content or "")


def test_holiday_router_passes_country_code():
    msg = HumanMessage(content="When is the next holiday in Canada?")
    out = route_holiday_intents(msg)
    assert out is not None
    tool_msg = out["messages"][-1]
    tcalls = getattr(tool_msg, "tool_calls", None) or getattr(
        tool_msg, "additional_kwargs", {}
    ).get("tool_calls")
    assert tcalls, "Expected a get_public_holidays tool call"
    args = tcalls[0].get("args") or {}
    assert args.get("country_code") == "CA"


def test_holiday_handler_accepts_list_payload(monkeypatch):
    """Ensure holiday handler works when ToolMessage content is a list."""
    monkeypatch.setitem(
        post_tools.__dict__, "_today_owner_date", lambda: datetime(2025, 11, 25).date()
    )
    holidays = [
        {"date": "2025-01-01", "name": "New Year's Day"},
        {"date": "2025-12-25", "name": "Christmas Day"},
    ]
    last = ToolMessage(content=holidays, name="get_public_holidays", tool_call_id="c")
    marker = AIMessage(
        content='PENDING_HOLIDAY_SCOPE:{"mode": "next", "scope": "week", "which": "this"}'
    )
    out = post_tools.handle_get_public_holidays(last, [marker, last])
    msgs = (out or {}).get("messages") or []
    assert msgs
    assert "2025-12-25" in (msgs[0].content or "")
