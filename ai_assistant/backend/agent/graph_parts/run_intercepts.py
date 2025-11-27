from __future__ import annotations
import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph_parts.tool_calls import _tc, _tc_name, _tc_args
from agent.graph_parts.intent_patterns import _is_booking_intent
from agent.graph_parts.time_parse import _extract_slot_minutes

USER_CANCEL_RE = re.compile(r"\b(cancel|stop|nevermind|never mind)\b", re.I)


def handle_pre_tool_intercepts(
    msgs: list[Any], last: AIMessage, config_conf: dict | None, tz_var: Any
) -> dict[str, list[Any]] | None:
    """Intercept risky tool calls to add holiday checks and confirmations.

    Args
    - msgs: Conversation history used to infer context and avoid duplicates.
    - last: The latest 'AIMessage' containing pending 'tool_calls'.
    - config_conf: Optional configuration carrying timezone info.
    - tz_var: A timezone context accessor (e.g., contextvar) with '.get()'.

    Returns
    - Dict containing injected messages (markers, confirmations, or reroutes)
      when an intercept applies; otherwise 'None' to proceed normally.

    Raises
    - None
    """
    if not (isinstance(last, AIMessage) and getattr(last, "tool_calls", None)):
        return None

    # Determine intent of tool calls
    wants_booking = any(
        (tc.get("function", {}) or {}).get("name") == "book_appointment"
        or tc.get("name") == "book_appointment"
        for tc in (last.tool_calls or [])
    )
    wants_reschedule = any(
        (tc.get("function", {}) or {}).get("name") == "reschedule_appointment"
        or tc.get("name") == "reschedule_appointment"
        for tc in (last.tool_calls or [])
    )
    wants_opening_once = any(
        (tc.get("function", {}) or {}).get("name") == "add_special_opening"
        or tc.get("name") == "add_special_opening"
        for tc in (last.tool_calls or [])
    )
    wants_recurring_openings = any(
        (tc.get("function", {}) or {}).get("name") == "create_recurring_openings"
        or tc.get("name") == "create_recurring_openings"
        for tc in (last.tool_calls or [])
    )

    # Helper to get today's date in owner's tz
    def _owner_today_iso() -> str:
        try:
            from zoneinfo import ZoneInfo

            tzname = (
                tz_var.get()
                or (
                    (config_conf or {}).get("tz")
                    if isinstance(config_conf, dict)
                    else None
                )
                or "America/Toronto"
            )
            from datetime import datetime as _d

            return _d.now(ZoneInfo(tzname)).date().isoformat()
        except Exception:
            from datetime import date as _date

            return _date.today().isoformat()

    # Build a holiday date and remember intended action
    holiday_date: str | None = None
    intended_tool: str | None = None
    intended_args: dict | None = None
    if wants_booking or wants_reschedule:
        for tc in last.tool_calls or []:
            nm = _tc_name(tc)
            if nm in ("book_appointment", "reschedule_appointment"):
                intended_tool = nm
                intended_args = _tc_args(tc)
                start_local = (intended_args or {}).get("start_local")
                if (
                    isinstance(start_local, str)
                    and len(start_local) >= 10
                    and start_local[4] == "-"
                ):
                    holiday_date = start_local[:10]
                break
    elif wants_opening_once:
        for tc in last.tool_calls or []:
            if _tc_name(tc) == "add_special_opening":
                intended_tool = "add_special_opening"
                intended_args = _tc_args(tc)
                sl = (intended_args or {}).get("start_local")
                if isinstance(sl, str) and len(sl) >= 10 and sl[4] == "-":
                    holiday_date = sl[:10]
                break
    elif wants_recurring_openings:
        for tc in last.tool_calls or []:
            if _tc_name(tc) == "create_recurring_openings":
                intended_tool = "create_recurring_openings"
                intended_args = _tc_args(tc)
                sd = (intended_args or {}).get("start_date")
                if isinstance(sd, str) and re.match(r"\d{4}-\d{2}-\d{2}$", sd):
                    holiday_date = sd
                else:
                    try:
                        wd = int((intended_args or {}).get("weekday") or 0)
                        from datetime import datetime as _dt, timedelta as _td

                        today_iso = _owner_today_iso()
                        y, m, d = [int(x) for x in today_iso.split("-")]
                        base = _dt(y, m, d)
                        delta = (wd - base.weekday()) % 7
                        first = base if delta == 0 else base + _td(days=delta)
                        holiday_date = first.date().isoformat()
                    except Exception:
                        holiday_date = None
                break

    # Always check public holiday once for these actions
    if (
        wants_booking
        or wants_reschedule
        or wants_opening_once
        or wants_recurring_openings
    ) and holiday_date:
        call_id = f"call_holiday_{holiday_date}"
        already = any(
            isinstance(m, AIMessage)
            and getattr(m, "name", "") == "is_public_holiday"
            and getattr(m, "tool_call_id", "") == call_id
            for m in msgs[-10:]
        ) or any(
            isinstance(m, SystemMessage)
            and isinstance(m.content, str)
            and m.content.startswith("PENDING_INTENT_AFTER_HOLIDAY:")
            for m in msgs[-5:]
        )
        if not already:
            cc = os.getenv("HOLIDAYS_DEFAULT_COUNTRY", "CA")
            rc = os.getenv("HOLIDAYS_DEFAULT_REGION", "CA-NB")
            marker_payload = {
                "tool": intended_tool,
                "args": intended_args or {},
                "date": holiday_date,
            }
            marker = SystemMessage(
                content="PENDING_INTENT_AFTER_HOLIDAY:" + json.dumps(marker_payload)
            )
            holiday_call = AIMessage(
                content="",
                tool_calls=[
                    _tc(
                        "is_public_holiday",
                        {"date": holiday_date, "country_code": cc, "region_code": rc},
                        id=call_id,
                    )
                ],
            )
            return {"messages": [marker, holiday_call]}

    # Booking confirmation gate (only for book_appointment)
    if wants_booking:
        try:
            last_human = next(
                (m for m in reversed(msgs[:-1]) if isinstance(m, HumanMessage)), None
            )
            txt = (last_human.content or "") if last_human else ""
            low = txt.lower()
            opening_intent = any(
                k in low
                for k in [
                    "opening",
                    "openings",
                    "availability",
                    "available",
                    "open my calendar",
                    "open up slots",
                ]
            )
            has_timeoff_word = any(
                k in low
                for k in [
                    "time off",
                    "pto",
                    "vacation",
                    "day off",
                    "out of office",
                    "ooo",
                    "block off my calendar",
                ]
            )
            if opening_intent and not has_timeoff_word:
                slot = _extract_slot_minutes(txt)
                if slot is None:
                    return {
                        "messages": [
                            AIMessage(
                                content="What slot length should I use for this opening (e.g., 30, 45, 60 minutes)?"
                            )
                        ]
                    }
                return {
                    "messages": [
                        SystemMessage(
                            content=(
                                "Do not book an appointment. Create a one-off opening instead. "
                                "Call 'add_special_opening' exactly once with times parsed from the user message and "
                                f"slot_minutes={slot}, buffer_minutes=0."
                            )
                        )
                    ]
                }
        except Exception:
            pass

        has_approval = False
        for m in reversed(msgs[:-1]):
            if isinstance(m, HumanMessage) and re.search(
                r"\b(confirm|proceed|yes|go ahead|do it)\b", (m.content or ""), re.I
            ):
                has_approval = True
                break
            if (
                isinstance(m, AIMessage)
                and isinstance(m.content, str)
                and m.content.startswith("CONFIRM_REQUIRED:")
            ):
                break

        tc_args = None
        for tc in last.tool_calls or []:
            if _tc_name(tc) == "book_appointment":
                tc_args = _tc_args(tc)
                break

        auto_ok = False
        if not has_approval and tc_args:
            has_start = bool(tc_args.get("start_local"))
            has_dur = bool(tc_args.get("duration_min"))
            if has_start and has_dur:
                for hm in (
                    m for m in reversed(msgs[:-1]) if isinstance(m, HumanMessage)
                ):
                    text = hm.content or ""
                    if USER_CANCEL_RE.search(text):
                        break
                    if _is_booking_intent(text):
                        auto_ok = True
                        break

        logging.getLogger(__name__).info(
            "run_tools booking gate: has_approval=%s auto_ok=%s args=%s",
            has_approval,
            auto_ok,
            (
                {
                    k: tc_args.get(k)
                    for k in (
                        "start_local",
                        "duration_min",
                        "client_email",
                        "client_name",
                    )
                }
                if tc_args
                else None
            ),
        )

        if not has_approval and not auto_ok:
            kept = {
                "start_local": (tc_args or {}).get("start_local"),
                "duration_min": (tc_args or {}).get("duration_min"),
                "client_name": (tc_args or {}).get("client_name"),
                "client_email": (tc_args or {}).get("client_email"),
                "client_query": (tc_args or {}).get("client_query"),
                "notes": (tc_args or {}).get("notes"),
            }
            human_when = (kept.get("start_local") or "").replace("T", " ")
            dur = kept.get("duration_min")
            who = kept.get("client_name") or kept.get("client_query") or "the client"
            eml = kept.get("client_email")
            who_str = f"{who} <{eml}>" if eml else who
            line = f"Please confirm: book {who_str} at {human_when} for {dur} minutes?"
            confirm_marker = SystemMessage(
                content="CONFIRM_REQUIRED:" + json.dumps(kept)
            )
            ask = AIMessage(content=line + " (yes / no)")
            return {"messages": [confirm_marker, ask]}

    return None
