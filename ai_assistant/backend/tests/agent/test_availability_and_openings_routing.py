from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import _tool_hint_for_text
from agent.graph_parts.router import (
    route_availability_overview,
    route_cancel_appointment_intent,
    route_time_off_intent,
    route_schedule_overview,
)
from agent.graph_parts.intent_patterns import _needs_duration_question
from agent.graph_parts.router import route_pending_client_instruction


def test_availability_overview_uses_calendar_snapshot_keywords_only():
    """Pure availability questions should result in a calendar_snapshot tool call.

    This exercises the updated route_availability_overview logic directly,
    without invoking the full graph/LLM.
    """
    msg = HumanMessage(content="When am I free tomorrow?")
    out = route_availability_overview(msg)
    assert out is not None
    msgs = out.get("messages") or []
    assert msgs, "Expected router to emit messages"
    ai = msgs[-1]
    tool_calls = getattr(ai, "tool_calls", None) or getattr(
        ai, "additional_kwargs", {}
    ).get("tool_calls")
    assert tool_calls, "Expected calendar_snapshot tool call for availability query"
    names = [tc.get("function", {}).get("name") or tc.get("name") for tc in tool_calls]
    assert "calendar_snapshot" in names


def test_opening_like_request_has_tool_hints_for_opening_tools():
    """Opening-style requests should produce tool intent hints mentioning opening tools.

    The router no longer hard-routes opening creation; instead the graph uses
    tool intent metadata to build a hint block. We exercise the helper directly.
    """
    hint = _tool_hint_for_text(
        "Add an opening next Friday 9am-12pm for 60 minute lessons."
    )
    assert (
        hint is not None
    ), "Expected a tool intent hint SystemMessage for opening-like query"
    # The hint should mention at least one of the opening-related tools we
    # configured in TOOL_INTENTS (e.g., add_special_opening).
    assert "add_special_opening" in hint or "create_recurring_openings" in hint


def test_cancel_intent_uses_cancel_appointment_flow_keywords():
    """Cancel phrasing should trigger the cancel_appointment routing plan."""
    msg = HumanMessage(content="Please cancel my appointment tomorrow.")
    out = route_cancel_appointment_intent(msg)
    assert out is not None
    msgs = out.get("messages") or []
    assert msgs and isinstance(msgs[0], SystemMessage)
    text = msgs[0].content or ""
    assert "cancel_appointment" in text
    assert "list_appointments" in text


def test_time_off_intent_uses_add_time_off_keywords():
    """Time-off phrasing should route to add_time_off instructions."""
    msg = HumanMessage(content="I need time off next Friday from 9am to 5pm.")
    out = route_time_off_intent(msg)
    assert out is not None
    msgs = out.get("messages") or []
    assert msgs and isinstance(msgs[0], SystemMessage)
    text = msgs[0].content or ""
    assert "add_time_off" in text


def test_time_off_intent_handles_block_off_phrase():
    """'Block off' phrasing should still route to time-off instructions."""
    msg = HumanMessage(content="Block off tomorrow from 4pm to 5pm.")
    out = route_time_off_intent(msg)
    assert out is not None
    msgs = out.get("messages") or []
    assert msgs and isinstance(msgs[0], SystemMessage)
    text = msgs[0].content or ""
    assert "add_time_off" in text


def test_finance_hint_includes_balances():
    """Finance phrasing should surface balance-related tool hints."""
    hint = _tool_hint_for_text("Does anyone owe me money?")
    assert hint is not None
    assert "customer_balances" in hint or "total_owed" in hint


def test_schedule_overview_vs_booking_duration_question():
    """Schedule queries should not trigger duration questions, while booking-like queries should."""
    # Schedule overview: should not need duration
    schedule_msgs = [HumanMessage(content="What's my schedule this week?")]
    assert _needs_duration_question(schedule_msgs) is False

    # Booking without duration: should need clarification
    booking_msgs = [HumanMessage(content="Book Fluffy tomorrow at 3pm")]
    assert _needs_duration_question(booking_msgs) is True

    # Booking with duration: no clarification needed
    booking_with_dur = [
        HumanMessage(content="Book Fluffy tomorrow at 3pm for 60 minutes")
    ]
    assert _needs_duration_question(booking_with_dur) is False


def test_pending_client_instruction_builds_booking_instruction():
    """A PENDING_CLIENT marker plus a booking-like human message should yield a booking instruction."""
    # Simulate a pending client marker from a prior step
    pending_blob = {
        "name": "Fluffy",
        "client_email": "fluffy@example.com",
        "people": [],
    }
    marker = SystemMessage(
        content="PENDING_CLIENT:" + __import__("json").dumps(pending_blob)
    )
    human = HumanMessage(content="Book Fluffy tomorrow at 10am for 60 minutes.")
    msgs = [marker, human]

    out = route_pending_client_instruction(marker, msgs)
    assert out is not None
    msgs_out = out.get("messages") or []
    assert msgs_out and isinstance(msgs_out[0], SystemMessage)
    text = msgs_out[0].content or ""
    # The instruction should mention either book_appointment or book_recurring_appointments.
    assert "book_appointment" in text or "book_recurring_appointments" in text
