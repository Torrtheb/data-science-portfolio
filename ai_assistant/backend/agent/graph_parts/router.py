from __future__ import annotations

import json
import os
import re
from datetime import datetime as _dt
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent.graph_parts.tool_calls import _tc
from agent.graph_parts.time_parse import (
    TIME_FROM_TO_RE,
    TIME_RANGE_RE,
    TIME_AT_RE,
    WEEKDAY_WORD_RE,
    _norm_hhmm_12to24,
    _extract_slot_minutes,
    _parse_weekday_from_text,
)
from agent.graph_parts.intent_patterns import _is_booking_intent, _extract_duration_min

try:
    import agent.tool_ctx as _tool_ctx_mod

    tz_var = _tool_ctx_mod.tz_var
except Exception:

    class _TzVarStub:
        def get(self):
            return None

    tz_var = _TzVarStub()


TOMORROW_RE = re.compile(r"\b(tomorrow|today|next\s+\w+|\d{4}-\d{2}-\d{2})\b", re.I)
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
REMIND_WHO_RE = re.compile(r"(?:to|for)\s+(?:remind\s+)?([A-Za-z][\w .'-]+)", re.I)
BROADCAST_CLIENTS_RE = re.compile(r"\b(?:all|my)\s+clients\b|\bclients\b", re.I)
CLIENT_NEXT_RE = re.compile(
    r"when\s+is\s+(.+?)'s\s+next\s+(appointment|lesson)\??", re.I
)
CLIENT_DAY_RE = re.compile(
    r"does\s+(.+?)\s+have\s+(?:an\s+|any\s+)?(appointments?|lessons?)\s+(today|tomorrow|\d{4}-\d{2}-\d{2})\??",
    re.I,
)

_OPENING_KEYWORDS = ("opening", "openings", "availability", "available", "open up")


def _has_opening_word(text: str) -> bool:
    """Lightweight opening/availability detector without regex dependence."""
    if not text:
        return False
    low = text.lower()
    return any(k in low for k in _OPENING_KEYWORDS)


def _has_day_hint(text: str) -> bool:
    """Check for common day hints (today/tomorrow/weekdays/ISO dates)."""
    if not text:
        return False
    low = text.lower()
    if any(k in low for k in ("tomorrow", "today", "next ")):
        return True
    if WEEKDAY_WORD_RE.search(text):
        return True
    return bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", text))


def route_holiday_intents(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Handle holiday overview intents and trigger 'get_public_holidays'.

    Args
    - last: The latest user message.

    Returns
    - Dict with a 'PENDING_HOLIDAY_SCOPE:{...}' marker and a 'get_public_holidays'
      tool call when matched; otherwise None.

    Raises
    - None
    """
    text_raw = last.content or ""
    text = text_raw.lower().strip()
    if not text:
        return None
    if "holiday" not in text and "holidays" not in text:
        return None

    which = "this"
    scope = "week"
    mode = None
    # Lightweight country detection to respect explicit user phrasing.
    country_code = None
    if "canada" in text:
        country_code = "CA"
    elif "united states" in text or "usa" in text or "us " in text:
        country_code = "US"
    elif "uk" in text or "britain" in text or "united kingdom" in text:
        country_code = "GB"

    if "next week" in text:
        which = "next"
    elif "today" in text:
        scope = "today"
        which = "this"
    elif "tomorrow" in text:
        scope = "today"
        which = "tomorrow"
    elif "next" in text and ("holiday" in text or "holidays" in text):
        mode = "next"
    payload = {"scope": scope, "which": which}
    if mode:
        payload["mode"] = mode
    if country_code:
        payload["country_code"] = country_code
    marker = AIMessage(content="PENDING_HOLIDAY_SCOPE:" + json.dumps(payload))
    call_args = {}
    if country_code:
        call_args["country_code"] = country_code
    return {
        "messages": [
            marker,
            AIMessage(
                content="",
                tool_calls=[_tc("get_public_holidays", call_args, id="call_holidays")],
            ),
        ]
    }


def route_one_off_opening(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route explicit requests to create a one-off opening.

    Args
    - last: The latest user message.

    Returns
    - Dict with a clarifying question, or with a holiday pre-check + pending
      intent marker and tool call; otherwise None.

    Raises
    - None
    """
    raw_text = last.content or ""
    wants_opening = _has_opening_word(raw_text)
    if not wants_opening:
        return None
    has_range = bool(TIME_FROM_TO_RE.search(raw_text) or TIME_RANGE_RE.search(raw_text))
    has_day_hint = _has_day_hint(raw_text)
    if not (has_range or has_day_hint):
        return None

    slot = _extract_slot_minutes(raw_text)
    if slot is None:
        prev_instruction = SystemMessage(
            content=(
                "Waiting for slot length for an opening request. After the user replies with a number of minutes, "
                "call 'add_special_opening' exactly once using start/end parsed from this earlier text: "
                f"{raw_text!s}"
            )
        )
        return {
            "messages": [
                prev_instruction,
                AIMessage(
                    content="What slot length should I use for this opening (e.g., 30, 45, 60 minutes)?"
                ),
            ]
        }

    m = TIME_FROM_TO_RE.search(raw_text) or TIME_RANGE_RE.search(raw_text)
    if not m:
        return {
            "messages": [
                AIMessage(
                    content="What start and end times should I use (e.g., 9am–5pm)?"
                )
            ]
        }
    s12, e12 = m.group(1), m.group(2)
    s24 = _norm_hhmm_12to24(s12)
    e24 = _norm_hhmm_12to24(e12)
    if not s24 or not e24:
        return {
            "messages": [
                AIMessage(
                    content="Please provide clear times like '9am to 5pm' or '09:00-17:00'."
                )
            ]
        }

    # Resolve date anchor deterministically in owner's timezone
    anchor_iso = None
    m_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw_text)
    if m_iso:
        anchor_iso = m_iso.group(1)
    if not anchor_iso:
        low = raw_text.lower()
        if re.search(r"\btomorrow\b", low):
            try:
                from zoneinfo import ZoneInfo

                tzname = tz_var.get() or "America/Toronto"
                today = _dt.now(ZoneInfo(tzname)).date()
                anchor_iso = (
                    today + __import__("datetime").timedelta(days=1)
                ).isoformat()
            except Exception:
                anchor_iso = (
                    _dt.now().date() + __import__("datetime").timedelta(days=1)
                ).isoformat()
        elif re.search(r"\btoday\b", low):
            try:
                from zoneinfo import ZoneInfo

                tzname = tz_var.get() or "America/Toronto"
                anchor_iso = _dt.now(ZoneInfo(tzname)).date().isoformat()
            except Exception:
                anchor_iso = _dt.now().date().isoformat()
        else:
            wd = _parse_weekday_from_text(raw_text)
            if wd is not None:
                try:
                    from zoneinfo import ZoneInfo

                    tzname = tz_var.get() or "America/Toronto"
                    base = _dt.now(ZoneInfo(tzname)).date()
                except Exception:
                    base = _dt.now().date()
                delta = (wd - base.weekday()) % 7
                delta = delta or 7
                anchor_iso = (
                    base + __import__("datetime").timedelta(days=delta)
                ).isoformat()

    if not anchor_iso:
        return {
            "messages": [
                AIMessage(
                    content="Which day should I add this opening for (today, tomorrow, or YYYY-MM-DD)?"
                )
            ]
        }

    args = {
        "start_local": f"{anchor_iso}T{s24}",
        "end_local": f"{anchor_iso}T{e24}",
        "slot_minutes": int(slot),
        "buffer_minutes": 0,
    }
    # Holiday pre-check at routing time
    day = anchor_iso
    cc = os.getenv("HOLIDAYS_DEFAULT_COUNTRY", "CA")
    rc = os.getenv("HOLIDAYS_DEFAULT_REGION", "CA-NB")
    call_id = f"call_holiday_{day}"
    marker = SystemMessage(
        content="PENDING_INTENT_AFTER_HOLIDAY:"
        + json.dumps({"tool": "add_special_opening", "args": args, "date": day})
    )
    holiday_call = AIMessage(
        content="",
        tool_calls=[
            _tc(
                "is_public_holiday",
                {"date": day, "country_code": cc, "region_code": rc},
                id=call_id,
            )
        ],
    )
    return {"messages": [marker, holiday_call]}


def route_opening_modification(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route requests to modify an existing opening.

    Args
    - last: The latest user message.

    Returns
    - Dict with a follow-up question for slot length or deterministic
      instructions to list and update the opening; otherwise None.

    Raises
    - None
    """
    raw_text = last.content or ""
    low = raw_text.lower()
    wants_opening = re.search(r"\bopening\b", low) or re.search(r"\bopen\b", low)
    wants_modify = bool(re.search(r"\b(modify|change|adjust|update)\b", low))
    has_range = bool(TIME_FROM_TO_RE.search(raw_text) or TIME_RANGE_RE.search(raw_text))
    has_day_hint = bool(
        TOMORROW_RE.search(low)
        or WEEKDAY_WORD_RE.search(low)
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", low)
    )
    if not (wants_opening and wants_modify and (has_range or has_day_hint)):
        return None
    slot = _extract_slot_minutes(raw_text)
    if slot is None:
        prev_instruction = SystemMessage(
            content=(
                "Waiting for slot length for an opening update. After the user replies with minutes, "
                "call 'list_openings' for the owner-local day parsed from this earlier text, pick the opening on that day, then call 'update_opening' exactly once with new start_local/end_local parsed from this text.\n"
                f"{raw_text!s}"
            )
        )
        return {
            "messages": [
                prev_instruction,
                AIMessage(
                    content="What slot length should I use for this opening (e.g., 30, 45, 60 minutes)?"
                ),
            ]
        }
    return {
        "messages": [
            SystemMessage(
                content=(
                    "User asked to modify an opening. Perform the following tool calls exactly once each: \n"
                    "1) 'list_openings' with the owner-local day parsed from the user's message (YYYY-MM-DD).\n"
                    "2) From the result, select the opening that overlaps the requested new window (or the only one if single).\n"
                    "3) 'update_opening' with: opening_id from step 2; start_local and end_local parsed from the user's message (owner local tz); "
                    f"slot_minutes={slot}; buffer_minutes=0."
                )
            )
        ]
    }


def route_opening_deletion(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route deletion requests for a one-off opening on a given day.

    Args
    - last: The latest user message.

    Returns
    - Dict with instructions to list and delete the opening; otherwise None.

    Raises
    - None
    """
    raw_text = last.content or ""
    low = raw_text.lower()
    mentions_opening = bool(re.search(r"\bopening\b|\bopen\b", low))
    wants_delete = bool(re.search(r"\b(delete|remove|cancel)\b", low))
    if not (mentions_opening and wants_delete):
        return None
    has_day_hint = bool(
        TOMORROW_RE.search(low)
        or WEEKDAY_WORD_RE.search(low)
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", low)
    )
    if not has_day_hint:
        return None
    return {
        "messages": [
            SystemMessage(
                content=(
                    "User asked to delete an opening. Perform these tool calls exactly: \n"
                    "1) 'list_openings' with the owner-local day parsed from the user's message (YYYY-MM-DD).\n"
                    "2) If exactly one opening exists, call 'delete_opening' with that opening_id. If multiple exist, prefer one that overlaps any time mentioned; otherwise ask the user to specify which time to delete."
                )
            )
        ]
    }


def route_add_client_refusal(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Politely refuse adding new clients via chat and offer alternatives.

    Args
    - last: The latest user message possibly requesting a new client.

    Returns
    - Dict with a courteous refusal and alternatives; otherwise None.

    Raises
    - None
    """
    raw = last.content or ""
    low = raw.lower()
    # Lightweight detection: phrases like "add client", "create client", "new client".
    if "add client" in low or "create client" in low or "new client" in low:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I can’t add new clients via chat. Please add clients from the dashboard. "
                        "I can help find or update existing clients, or handle scheduling and messages."
                    )
                )
            ]
        }
    return None


def route_email_message_intent(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route email/message composition requests.

    Args
    - last: The latest user message requesting to send an email/message.

    Returns
    - Dict with deterministic steps to resolve recipients and call
      'create_email_draft'; supports broadcast and single-recipient flows;
      otherwise None.

    Raises
    - None
    """
    raw = last.content or ""
    text = raw.strip()
    low = text.lower()
    if not any(k in low for k in ["email", "message", "notify", "remind", "reach out"]):
        return None
    m_email = EMAIL_RE.search(text)
    to_hint = m_email.group(0) if m_email else None
    m_who = REMIND_WHO_RE.search(low)
    who_hint = None
    if m_who:
        who_hint = m_who.group(1).strip(" .,'\"")
    if BROADCAST_CLIENTS_RE.search(low) and not to_hint:
        prompt = (
            "User asked to send a message to multiple clients (broadcast).\n"
            "Do NOT look up a single client.\n"
            "Steps:\n"
            "1) Call 'list_clients' once (limit=200).\n"
            "2) Build 'recipients' = [{email,name?}] from results that have a primary email.\n"
            "3) Compose a concise subject and a short, friendly body (2–5 lines) reflecting the user's message.\n"
            "4) Call 'create_email_draft' exactly once with: 'subject', 'lines', and 'recipients'.\n"
            "Do NOT comma‑join addresses into 'to'. Use the 'recipients' array."
        )
        return {"messages": [SystemMessage(content=prompt)]}
    prompt = (
        "User asked to send an email/message.\n"
        "Resolve the recipient:\n"
        "- If an email is present in the message, use it.\n"
        "- Else, look up a single matching client (by exact name) and use the primary email.\n"
        "If multiple matches, ask user to clarify the exact person.\n"
        "Then compose a concise subject and a short, friendly body (2–5 lines). "
        "If the user mentions 'tomorrow' or 'today', reflect that in the body. "
        "Use the person's preferred name if known.\n"
        "Finally, call 'create_email_draft' exactly once with: 'to', 'subject', 'lines', 'to_name' (optional)."
    )
    if to_hint or who_hint:
        prompt += f"\nHints → to: {to_hint or '} | who: {who_hint or '}"
    return {"messages": [SystemMessage(content=prompt)]}


def route_time_off_intent(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route requests to add time off (PTO/OOO) with clear instructions.

    Args
    - last: The latest user message possibly requesting time off.

    Returns
    - Dict with a deterministic instruction to call 'add_time_off'; otherwise None.

    Raises
    - None
    """
    txt_raw = last.content or ""
    low = txt_raw.lower()
    # Use simple keywords aligned with the 'add_time_off' tool intent metadata.
    if not any(
        k in low
        for k in [
            "time off",
            "pto",
            "vacation",
            "day off",
            "out of office",
            "ooo",
            "block off my calendar",
            "block off",
        ]
    ):
        return None
    return {
        "messages": [
            SystemMessage(
                content=(
                    "The user asked to add time off (PTO/OOO). "
                    "Extract a single start and end in the owner's LOCAL time from the last user message "
                    "and call the tool 'add_time_off' exactly once with:\n"
                    "  - start_local: e.g., 'YYYY-MM-DDTHH:MM' or natural text like 'Saturday 9am'\n"
                    "  - end_local:   same format, e.g., 'Sunday 9pm'\n"
                    "Resolve relative days (today/tomorrow/this Saturday/next Sunday) in the owner's timezone.\n"
                    "If either date or time is missing, ask ONE clarifying question to get it.\n"
                    "If the tool reports overlapping booked appointments, do NOT create time off; "
                    "explain which appointments block the change and that they must be cancelled first."
                )
            )
        ]
    }


def route_time_off_status(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route queries asking about existing/upcoming time off to a tool."""

    txt_raw = last.content or ""
    low = txt_raw.lower()
    if not any(
        k in low
        for k in [
            "next time off",
            "upcoming time off",
            "do i have time off",
            "when is my time off",
            "time off scheduled",
            "vacation",
            "pto",
            "day off",
            "out of office",
            "ooo",
        ]
    ):
        return None
    # If the user is clearly asking to create/add, let the creation router handle it.
    if any(k in low for k in ["add", "create", "set", "make", "block", "schedule"]):
        return None
    return {
        "messages": [
            SystemMessage(
                content=(
                    "User asked about existing time off. "
                    "Call 'next_time_off' exactly once to find the next upcoming block. "
                    "If found, summarize the local start and end and include any note. "
                    "If none is scheduled, say so and offer to add time off."
                )
            )
        ]
    }


def route_truncate_after(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route requests like “not available after HH:MM” to 'truncate_after'.

    Args
    - last: The latest user message containing a cutoff time.

    Returns
    - Dict instructing to call 'truncate_after' with 'local_hhmm'; otherwise None.

    Raises
    - None
    """
    txt = (last.content or "").strip()
    low = txt.lower()
    if not any(
        k in low
        for k in [
            "not available after",
            "no availability after",
            "block off after",
            "off after",
        ]
    ):
        return None

    m = re.search(r"(\d{1,2}:\d{2})", txt)
    if not m:
        return None
    hhmm = m.group(1)
    return {
        "messages": [
            SystemMessage(
                content=(
                    f"The user said they are not available after {hhmm} today (owner-local). "
                    f"Call the tool 'truncate_after' with local_hhmm='{hhmm}'. "
                    "If the tool returns requires_cancellation=true, tell the user which booked appointments "
                    "block the change and that they must cancel them first. "
                    "Do not attempt to modify openings when appointments block the cutoff."
                )
            )
        ]
    }


def route_client_next(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route queries for a client’s next appointment.

    Args
    - last: The latest user message containing a client name.

    Returns
    - Dict instructing 'get_next_appointment_for_client'; otherwise None.

    Raises
    - None
    """
    txt = last.content or ""
    m = CLIENT_NEXT_RE.search(txt)
    if not m:
        return None
    who = m.group(1).strip().strip("?")
    return {
        "messages": [
            SystemMessage(
                content=(
                    f"User asked for next appointment for '{who}'. "
                    "Call 'get_next_appointment_for_client' once with "
                    f"'client_query'='{who}', 'include_canceled'=False."
                )
            )
        ]
    }


def route_client_day(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route queries asking if a client has appointments on a given day.

    Args
    - last: The latest user message containing client and day.

    Returns
    - Dict instructing 'list_appointments' with filters; otherwise None.

    Raises
    - None
    """
    txt = last.content or ""
    m = CLIENT_DAY_RE.search(txt)
    if not m:
        return None
    who = m.group(1).strip()
    day = m.group(2).lower()
    return {
        "messages": [
            SystemMessage(
                content=(
                    f"User asked if '{who}' has appointments on '{day}'. "
                    "Call 'list_appointments' once with "
                    f"'day'='{day}', 'include_canceled'=False, 'client_query'='{who}'. "
                    "Then answer strictly based on the tool output."
                )
            )
        ]
    }


def route_opening_slot_length_followup(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Handle replies that only provide slot length minutes for openings.

    Args
    - last: The latest user message that likely contains only minutes.
    - msgs: Conversation history used to recover the earlier opening request text.

    Returns
    - Dict with instructions to create or update an opening; otherwise None.

    Raises
    - None
    """
    slot = None
    m = re.search(r"\b(\d{1,3})\s*(?:mins?|minutes?)\b", last.content or "", re.I)
    try:
        if m:
            slot = int(m.group(1))
    except Exception:
        slot = None
    if slot is None:
        return None
    prior_opening_text = None
    for mm in reversed(msgs[:-1]):
        if isinstance(mm, HumanMessage):
            txt = mm.content or ""
            if _has_opening_word(txt) and (
                TIME_FROM_TO_RE.search(txt)
                or TIME_RANGE_RE.search(txt)
                or _has_day_hint(txt)
                or _extract_explicit_date(txt)
            ):
                prior_opening_text = txt
                break
    if not prior_opening_text:
        return None
    if re.search(r"\b(modify|change|adjust|update)\b", prior_opening_text, re.I):
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "Update an existing opening now. Do: \n"
                        "1) 'list_openings' for the owner-local day parsed from this earlier text.\n"
                        "2) Pick the opening on that day (prefer the one overlapping the requested window).\n"
                        "3) 'update_opening' with that opening_id, start_local/end_local parsed from this earlier text, "
                        f"slot_minutes: {slot}, buffer_minutes: 0.\n"
                        f"Earlier text: {prior_opening_text}"
                    )
                )
            ]
        }
    return {
        "messages": [
            SystemMessage(
                content=(
                    "Create a one-off opening now. Call 'add_special_opening' exactly once with: \n"
                    "- start_local and end_local parsed from this earlier text (owner local tz):\n"
                    f"{prior_opening_text}\n"
                    f"- slot_minutes: {slot}\n"
                    "- buffer_minutes: 0\n"
                )
            )
        ]
    }


# ---- Snapshot routes ----
def _extract_explicit_date(text: str) -> str | None:
    """Extract 'YYYY-MM-DD' or “Month DD, YYYY” from free text, if present."""
    if not text:
        return None
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    try:
        m2 = re.search(
            r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
            text,
            re.I,
        )
        if m2:
            mon, dd, yyyy = m2.groups()
            from datetime import datetime as _d

            for fmt in ("%B %d %Y", "%b %d %Y"):
                try:
                    return (
                        _d.strptime(f"{mon} {int(dd)} {int(yyyy)}", fmt)
                        .date()
                        .isoformat()
                    )
                except Exception:
                    pass
    except Exception:
        pass
    return None


def route_lessons_overview(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route high-level “lessons” overview requests to 'calendar_snapshot'.

    Args
    - last: The latest user message indicating a lessons overview intent.

    Returns
    - Dict that triggers a 'calendar_snapshot' tool call; otherwise None.

    Raises
    - None
    """
    text = (last.content or "").lower().strip()
    raw_text = last.content or ""
    # Lightweight lessons-overview detection: look for lesson-related words
    # when the message is not clearly a booking request.
    has_lessons_word = any(
        k in text for k in ["lesson", "lessons", "my lessons", "lessons today"]
    )
    if not (has_lessons_word and not _is_booking_intent(text)):
        return None
    scope = "today"
    if "next week" in text:
        scope = "week"
    elif "this week" in text or "week" in text:
        scope = "week"
    elif "this month" in text or "month" in text:
        scope = "month"
    elif "today" in text:
        scope = "today"
    elif "tomorrow" in text:
        scope = "today"
    args: dict[str, Any] = {"scope": scope}
    if "next week" in text:
        args["anchor"] = "next_week"
    elif "tomorrow" in text:
        args["anchor"] = "tomorrow"
    elif "today" in text:
        args["anchor"] = "today"
    else:
        explicit = _extract_explicit_date(raw_text)
        if explicit:
            args["anchor"] = explicit
            args["scope"] = "today"
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[_tc("calendar_snapshot", args, id="call_lessons_snapshot")],
            )
        ]
    }


def route_schedule_overview(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route “schedule/agenda” overview requests to 'calendar_snapshot'.

    Args
    - last: The latest user message indicating a schedule overview intent.

    Returns
    - Dict that triggers a 'calendar_snapshot' tool call; otherwise None.

    Raises
    - None
    """
    text = (last.content or "").lower().strip()
    raw_text = last.content or ""
    # Lightweight schedule/agenda detection: look for common schedule words
    # when the message is not clearly a booking request.
    has_schedule_word = any(
        k in text
        for k in [
            "schedule",
            "agenda",
            "what's my day",
            "what is my day",
            "what am i doing",
        ]
    )
    if not (has_schedule_word and not _is_booking_intent(text)):
        return None
    scope = "today"
    if "next week" in text:
        scope = "week"
    elif "this week" in text or "week" in text:
        scope = "week"
    elif "this month" in text or "month" in text:
        scope = "month"
    elif "today" in text:
        scope = "today"
    elif "tomorrow" in text:
        scope = "today"
    args: dict[str, Any] = {"scope": scope}
    if "next week" in text:
        args["anchor"] = "next_week"
    elif "tomorrow" in text:
        args["anchor"] = "tomorrow"
    elif "today" in text:
        args["anchor"] = "today"
    else:
        explicit = _extract_explicit_date(raw_text)
        if explicit:
            args["anchor"] = explicit
            args["scope"] = "today"
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    _tc("calendar_snapshot", args, id="call_schedule_snapshot")
                ],
            )
        ]
    }


def route_availability_overview(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route pure availability questions (not booking) to 'calendar_snapshot'.

    Args
    - last: The latest user message indicating an availability overview intent.

    Returns
    - Dict that triggers a 'calendar_snapshot' tool call; otherwise None.

    Raises
    - None
    """
    text = (last.content or "").lower().strip()
    raw_text = last.content or ""
    # Treat any mention of availability/free/open slots that is not clearly a
    # booking request as an availability overview. This avoids heavy regex
    # dependence while keeping behavior similar.
    has_avail_word = any(
        k in text
        for k in [
            "availability",
            "available",
            "free time",
            "free slots",
            "when am i free",
            "when can i",
            "open slots",
            "open times",
        ]
    )
    if not (has_avail_word and not _is_booking_intent(text)):
        return None
    scope = "today"
    if "next week" in text:
        scope = "week"
    elif "this week" in text or "week" in text:
        scope = "week"
    elif "this month" in text or "month" in text:
        scope = "month"
    elif "today" in text:
        scope = "today"
    elif "tomorrow" in text:
        scope = "today"
    args: dict[str, Any] = {"scope": scope}
    if "next week" in text:
        args["anchor"] = "next_week"
    elif "tomorrow" in text:
        args["anchor"] = "tomorrow"
    elif "today" in text:
        args["anchor"] = "today"
    else:
        explicit = _extract_explicit_date(raw_text)
        if explicit:
            args["anchor"] = explicit
            args["scope"] = "today"
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    _tc("calendar_snapshot", args, id="call_availability_snapshot")
                ],
            )
        ]
    }


def route_cancel_appointment_intent(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route cancellation intents to 'cancel_appointment' with clarifications.

    Args
    - last: The latest user message possibly requesting a cancellation.

    Returns
    - Dict describing the precise 'list_appointments' lookup and the
      'cancel_appointment' call to perform; otherwise None.

    Raises
    - None
    """
    txt = last.content or ""
    low = txt.lower()
    # Lightweight keyword detection instead of regex: look for a cancel verb
    # together with an appointment/lesson noun.
    if not ("cancel" in low and ("appointment" in low or "lesson" in low)):
        return None
    who = None
    m_name = re.search(r"cancel\s+(.+?)'?s?\s+appointment", txt, re.I)
    if m_name:
        who = (m_name.group(1) or "").strip().strip('"')
    time_12 = None
    m_time = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", txt, re.I)
    if m_time:
        time_12 = _norm_hhmm_12to24(m_time.group(1))
    day_text = None
    for k in (
        "today",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ):
        if re.search(rf"\b{k}\b", txt, re.I):
            day_text = k
            break
    day_arg = day_text or "today"
    q = {"day": day_arg, "include_canceled": False}
    if who:
        q["client_query"] = who
    plan = (
        "User asked to cancel an appointment. "
        "First call 'list_appointments' with the exact arguments: "
        + json.dumps(q)
        + ". "
        "From its JSON result, pick one appointment id"
    )
    if time_12:
        plan += f" whose start_local contains 'T{time_12}'. "
    else:
        plan += " (e.g., the earliest one for that client/day). "
    plan += (
        "Then call 'cancel_appointment' EXACTLY ONCE with the argument 'appointment_id' set to that id, "
        "and include a short reason like 'Client requested'."
    )
    return {"messages": [SystemMessage(content=plan)]}


def route_client_phone_update(last: HumanMessage) -> dict[str, list[Any]] | None:
    """Route simple “update phone” requests to 'update_client' flow.

    Args
    - last: The latest user message indicating a phone update intent.

    Returns
    - Dict instructing to lookup the client and update phone; otherwise None.

    Raises
    - None
    """
    raw = last.content or ""
    low = raw.lower()
    if not re.search(r"\b(phone|phone number)\b", low):
        return None
    mpos = re.search(r"([A-Za-z][\w .'-]+?)'s\s+phone", raw)
    client_hint = mpos.group(1).strip() if mpos else None
    if client_hint:
        return {
            "messages": [
                SystemMessage(
                    content=(
                        f"User wants to change a client's phone. "
                        f"Call 'find_client' with 'selector'='{client_hint}'. "
                        "After it returns, if a valid 10-digit number is present, call 'update_client' with 'client_id' and the normalized phone (###-###-####). "
                        "If the number is invalid, ask for a valid 10-digit number."
                    )
                )
            ]
        }
    return {
        "messages": [
            AIMessage(content="Which client should I update the phone number for?")
        ]
    }


def route_duration_only_followup(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Handle replies that only specify a duration without a day.

    Args
    - last: The latest user message containing only a duration.
    - msgs: Conversation history for context and pending markers.

    Returns
    - Dict asking which day to check next when applicable; otherwise None.

    Raises
    - None
    """
    txt = (last.content or "").strip()
    # duration-only reply (e.g., "30", "30 min", "30 minutes", "45 mins")
    dur_only = bool(re.search(r"\b(\d{1,3})\s*(?:min|mins|minutes)?\b", txt))
    has_day = bool(
        TOMORROW_RE.search(txt)
        or WEEKDAY_WORD_RE.search(txt)
        or _extract_explicit_date(txt)
    )
    if not (dur_only and not has_day):
        return None
    prev_human = None
    for mm in reversed(msgs[:-1]):
        if isinstance(mm, HumanMessage):
            prev_human = mm
            break
    if prev_human:
        prev_txt = prev_human.content or ""
        prev_has_opening = bool(re.search(r"\bopening\b", prev_txt.lower()))
        prev_has_day = bool(
            TOMORROW_RE.search(prev_txt.lower())
            or WEEKDAY_WORD_RE.search(prev_txt.lower())
            or _extract_explicit_date(prev_txt)
        )
        prev_has_range = bool(
            TIME_FROM_TO_RE.search(prev_txt) or TIME_RANGE_RE.search(prev_txt)
        )
        if prev_has_opening and (prev_has_day or prev_has_range):
            return None
        # Do we have a pending resolved client? (from post_tools → PENDING_CLIENT)
        has_pending_client = False
        for mm in reversed(msgs[:-1]):
            if (
                isinstance(mm, AIMessage)
                and isinstance(mm.content, str)
                and mm.content.startswith("PENDING_CLIENT:")
            ):
                has_pending_client = True
                break
        if has_pending_client:
            return {
                "messages": [
                    AIMessage(
                        content="Great — which day should I check (today, tomorrow, or a specific day like ‘next Monday’)?"
                    )
                ]
            }
    return None


def route_pending_choices(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Resolve 'PENDING_CHOICES' follow-ups by picking day vs weekly scope.

    Args
    - last: The latest user message containing the choice.
    - msgs: Conversation history carrying the pending choices marker.

    Returns
    - Dict instructing the chosen tool call with its args; otherwise None.

    Raises
    - None
    """
    pending = None
    for m in reversed(msgs[:-1]):
        if (
            isinstance(m, AIMessage)
            and isinstance(m.content, str)
            and m.content.startswith("PENDING_CHOICES:")
        ):
            try:
                pending = json.loads(m.content[len("PENDING_CHOICES:") :])
            except Exception:
                pending = None
            break
    if not pending:
        return None
    text = last.content or ""
    low = (text or "").lower()
    # Decide whether the user chose "day only" vs "weekly" based on simple keywords.
    pick = None
    if any(
        k in low
        for k in [
            "this day",
            "just this day",
            "only this day",
            "today",
            "that day",
            "one-off",
            "single day",
            "date only",
        ]
    ):
        pick = "day"
    elif any(
        k in low
        for k in [
            "every",
            "weekly",
            "rule",
            "recurring",
            "for mon",
            "for tue",
            "for wed",
            "for thu",
            "for fri",
            "for sat",
            "for sun",
        ]
    ):
        pick = "weekly"
    if pick and pick in pending:
        choice = pending[pick]
        tool_name = choice.get("tool")
        args = choice.get("args") or {}
        return {
            "messages": [
                SystemMessage(
                    content=(
                        f"User chose '{pick}'. Call the tool '{tool_name}' exactly once with args: "
                        + json.dumps(args)
                    )
                )
            ]
        }
    return None


def route_person_after_ambiguity(
    prev_message: Any, last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Resolve who to book for when multiple people match a client.

    Args
    - prev_message: The previous AI clarification asking who the appointment is for.
    - last: The latest user message providing the exact name.
    - msgs: Conversation history containing the pending client marker.

    Returns
    - Dict with an updated 'PENDING_CLIENT' marker and a follow-up time question;
      otherwise None.

    Raises
    - None
    """
    pending_client = None
    if (
        isinstance(prev_message, AIMessage)
        and isinstance(prev_message.content, str)
        and "Who is this appointment for" in prev_message.content
    ):
        for mm in reversed(msgs[:-1]):
            if (
                isinstance(mm, (AIMessage, SystemMessage))
                and isinstance(mm.content, str)
                and mm.content.startswith("PENDING_CLIENT:")
            ):
                try:
                    pending_client = json.loads(mm.content[len("PENDING_CLIENT:") :])
                except Exception:
                    pending_client = None
                break
    else:
        return None
    if not pending_client:
        return None

    # Avoid pivoting during opening flow — detect recent opening instructions/questions
    def _recent_opening_flow(msgs_list: list) -> bool:
        try:
            for mm in reversed(msgs_list[:-1]):
                if isinstance(mm, AIMessage):
                    txt = (mm.content or "") if isinstance(mm.content, str) else ""
                    low = txt.lower()
                    if (
                        "slot length" in low
                        and "opening" in low
                        or "waiting for slot length for an opening" in low
                        or "create a one-off opening" in low
                        or "call 'add_special_opening'" in low
                        or low.startswith("pending_intent_after_holiday:")
                        and "add_special_opening" in low
                    ):
                        return True
                elif isinstance(mm, SystemMessage):
                    stxt = mm.content or ""
                    slow = stxt.lower()
                    if "add_special_opening" in slow or (
                        "opening" in slow and "slot length" in slow
                    ):
                        return True
            return False
        except Exception:
            return False

    if _recent_opening_flow(msgs):
        return None
    user_choice = (last.content or "").strip().lower()
    people = pending_client.get("people", [])
    chosen_person = None
    for person in people:
        if person.get("full_name", "").lower() == user_choice:
            chosen_person = person
            break
    if chosen_person:
        enhanced_client = dict(pending_client)
        enhanced_client["chosen_person"] = chosen_person
        enhanced_client["chosen_person_id"] = chosen_person.get("person_id")
        enhanced_client["chosen_person_name"] = chosen_person.get("full_name")
        enhanced_client["chosen_person_email"] = (
            chosen_person.get("email")
            or pending_client.get("primary_email")
            or pending_client.get("client_email")
            or pending_client.get("email")
        )
        marker = SystemMessage(content="PENDING_CLIENT:" + json.dumps(enhanced_client))
        ask = AIMessage(
            content=(
                f"Great! I'll book for {chosen_person.get('full_name')}. What time should I book the appointment? "
                "(e.g., 'tomorrow at 10am for 30 minutes')"
            )
        )
        return {"messages": [marker, ask]}
    else:
        people_names = [p.get("full_name", "") for p in people if p.get("full_name")]
        return {
            "messages": [
                AIMessage(
                    content=f"I didn't find '{(last.content or '').strip()}' in the list. Please choose from: {', '.join(people_names)}"
                )
            ]
        }


def route_pending_identity(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Handle identity follow-ups from 'PENDING_IDENTITY' markers.

    - Accepts an email or a full name, merges into pending tool args, and
      either asks for missing time/duration or instructs to book/update.
    """
    pending = None
    for m in reversed(msgs[:-1]):
        if (
            isinstance(m, AIMessage)
            and isinstance(m.content, str)
            and m.content.startswith("PENDING_IDENTITY:")
        ):
            try:
                pending = json.loads(m.content[len("PENDING_IDENTITY:") :])
            except Exception:
                pending = None
            break
    if not pending or not isinstance(last.content, str):
        return None
    user_text = last.content.strip()
    email_m = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", user_text, re.I)
    identity: dict[str, Any] = {}
    if email_m:
        identity["client_email"] = email_m.group(0)
    else:
        identity["client_name"] = user_text
    tool = pending.get("tool")
    base_args = pending.get("args") or {}
    args = {**base_args, **identity}
    if tool == "book_appointment":
        has_start = bool(args.get("start_local"))
        has_dur = bool(args.get("duration_min"))
        if not (has_start and has_dur):
            return {
                "messages": [
                    AIMessage(
                        content="Got it. What start time and duration should I book (e.g., '2025-09-26T10:00' for 60 minutes)?"
                    )
                ]
            }
        try:
            payload = {
                k: v
                for k, v in args.items()
                if k
                in (
                    "start_local",
                    "duration_min",
                    "client_name",
                    "client_email",
                    "person_id",
                    "client_query",
                )
            }
        except Exception:
            payload = args
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "Book now. Call 'book_appointment' exactly once with: "
                        + json.dumps(payload)
                    )
                )
            ]
        }
    if tool == "update_appointment_details":
        if not args.get("appointment_id"):
            return {"messages": []}
    return {
        "messages": [
            SystemMessage(
                content=f"Replay the previous tool '{tool}' exactly once with args: {json.dumps(args)}"
            )
        ]
    }


def route_ambiguous_client_before_booking(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Ask the user to pick a person when multiple people match before booking.

    Args
    - last: The latest user message indicating intent to book.
    - msgs: Conversation history containing the pending client marker.

    Returns
    - Dict with a clarification question listing matching people; otherwise None.

    Raises
    - None
    """
    pending_client = None
    for mm in reversed(msgs[:-1]):
        if (
            isinstance(mm, (AIMessage, SystemMessage))
            and isinstance(mm.content, str)
            and mm.content.startswith("PENDING_CLIENT:")
        ):
            try:
                pending_client = json.loads(mm.content[len("PENDING_CLIENT:") :])
            except Exception:
                pending_client = None
            break
    if not pending_client:
        return None
    people = pending_client.get("people", [])
    if isinstance(people, list) and len(people) > 1:
        if re.search(r"\b(book|schedule|set\s+up|make)\b", (last.content or ""), re.I):
            people_names = [
                p.get("full_name", "") for p in people if p.get("full_name")
            ]
            if people_names:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                f"I found multiple people for {pending_client.get('name', 'this client')}: "
                                f"{', '.join(people_names)}. Who is this appointment for? Please specify the exact name."
                            )
                        )
                    ]
                }
    return None


# ---- Book selected slot flows ----


def _identity_from_pending_client(
    pending_client: dict | None,
) -> tuple[str | None, str | None, object | None]:
    if not isinstance(pending_client, dict):
        return None, None, None
    name = (
        pending_client.get("chosen_person_name")
        or pending_client.get("person_name")
        or pending_client.get("client_name")
        or pending_client.get("name")
    )
    for val in (
        pending_client.get("chosen_person_email"),
        pending_client.get("primary_email"),
        pending_client.get("client_email"),
        pending_client.get("email"),
    ):
        if val:
            email = val
            break
    else:
        email = None
    person_id = pending_client.get("chosen_person_id") or pending_client.get(
        "person_id"
    )
    return name, email, person_id


def _load_pending_slots_and_client(msgs: list[Any]) -> tuple[dict | None, dict | None]:
    pending_slots = None
    pending_client = None
    for mm in reversed(msgs[:-1]):
        if isinstance(mm, AIMessage) and isinstance(mm.content, str):
            if mm.content.startswith("PENDING_SLOTS:") and not pending_slots:
                try:
                    pending_slots = json.loads(mm.content[len("PENDING_SLOTS:") :])
                except Exception:
                    pending_slots = None
            if mm.content.startswith("PENDING_CLIENT:") and not pending_client:
                try:
                    pending_client = json.loads(mm.content[len("PENDING_CLIENT:") :])
                except Exception:
                    pending_client = None
        if pending_slots and (pending_client is not None):
            break
    return pending_slots, pending_client


def _failed_overlap_time(msgs: list[Any]) -> str | None:
    for mm in reversed(msgs[:-1]):
        if (
            isinstance(mm, AIMessage)
            and isinstance(mm.content, str)
            and mm.content.startswith("PENDING_OVERLAP_AT:")
        ):
            return mm.content[len("PENDING_OVERLAP_AT:") :]
    return None


def route_book_range_selected(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Book a slot when the user replies with an exact start–end range.

    Args
    - last: The latest user message containing a time range.
    - msgs: Conversation history including a 'PENDING_SLOTS' marker and identity.

    Returns
    - Dict with a deterministic 'book_appointment' instruction; otherwise None.

    Raises
    - None
    """
    text_raw = last.content or ""
    m = TIME_RANGE_RE.search(text_raw)
    if not m:
        return None
    start_txt = m.group(1)
    end_txt = m.group(2)
    start_hhmm = _norm_hhmm_12to24(start_txt)
    end_hhmm = _norm_hhmm_12to24(end_txt)
    pending_slots, pending_client = _load_pending_slots_and_client(msgs)
    if not (
        pending_slots
        and isinstance(pending_slots.get("slots"), list)
        and start_hhmm
        and end_hhmm
    ):
        return None
    chosen = None
    for s in pending_slots["slots"]:
        try:
            sl = s.get("start_local")
            el = s.get("end_local")
            if not (sl and el and len(sl) >= 16 and len(el) >= 16):
                continue
            if sl[11:16] == start_hhmm and el[11:16] == end_hhmm:
                chosen = (sl, el)
                break
        except Exception:
            continue
    if not chosen:
        return None
    start_local, end_local = chosen
    ovl = _failed_overlap_time(msgs)
    if ovl and start_local == ovl:
        return {
            "messages": [
                AIMessage(
                    content="That exact time just failed due to a conflict. Please pick another slot from the list above."
                )
            ]
        }
    sh, sm = map(int, start_local[11:16].split(":"))
    eh, em = map(int, end_local[11:16].split(":"))
    duration_min = (eh * 60 + em) - (sh * 60 + sm)
    c_name = c_email = c_person_id = None
    if pending_client:
        c_name, c_email, c_person_id = _identity_from_pending_client(pending_client)
    person_id_hint = ""
    if c_person_id:
        person_id_hint = f"- person_id: {c_person_id}\n"
    return {
        "messages": [
            SystemMessage(
                content=(
                    "Book the selected slot now. Call 'book_appointment' exactly once with:\n"
                    f"- start_local: '{start_local}'\n"
                    f"- duration_min: {duration_min}\n"
                    f"- client_name: {json.dumps(c_name) if c_name else 'null'}\n"
                    f"- client_email: {json.dumps(c_email) if c_email else 'null'}\n"
                    f"{person_id_hint}"
                    "Do not ask for the duration again."
                )
            )
        ]
    }


def route_book_from_to_selected(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Book a slot when the user replies with “from HH:MM to HH:MM”.

    Args
    - last: The latest user message containing a from–to range.
    - msgs: Conversation history including a 'PENDING_SLOTS' marker and identity.

    Returns
    - Dict with a deterministic 'book_appointment' instruction; otherwise None.

    Raises
    - None
    """
    text_raw = last.content or ""
    m_ft = TIME_FROM_TO_RE.search(text_raw)
    if not m_ft:
        return None
    start_txt = m_ft.group(1)
    end_txt = m_ft.group(2)
    start_hhmm = _norm_hhmm_12to24(start_txt)
    end_hhmm = _norm_hhmm_12to24(end_txt)
    pending_slots, pending_client = _load_pending_slots_and_client(msgs)
    if not (
        pending_slots
        and isinstance(pending_slots.get("slots"), list)
        and start_hhmm
        and end_hhmm
    ):
        return None
    chosen = None
    for s in pending_slots["slots"]:
        sl = s.get("start_local")
        el = s.get("end_local")
        if not (sl and el and len(sl) >= 16 and len(el) >= 16):
            continue
        if sl[11:16] == start_hhmm and el[11:16] == end_hhmm:
            chosen = (sl, el)
            break
    if not chosen:
        return None
    start_local, end_local = chosen
    ovl = _failed_overlap_time(msgs)
    if ovl and start_local == ovl:
        return {
            "messages": [
                AIMessage(
                    content="That exact time just failed due to a conflict. Please pick another slot from the list above."
                )
            ]
        }
    sh, sm = map(int, start_local[11:16].split(":"))
    eh, em = map(int, end_local[11:16].split(":"))
    duration_min = (eh * 60 + em) - (sh * 60 + sm)
    c_name = c_email = c_person_id = None
    if pending_client:
        c_name, c_email, c_person_id = _identity_from_pending_client(pending_client)
    booking_args = {
        "start_local": start_local,
        "duration_min": int(duration_min),
        "client_name": c_name,
        "client_email": c_email,
    }
    if c_person_id:
        if isinstance(c_person_id, (int, str)):
            try:
                booking_args["person_id"] = int(c_person_id)
            except (TypeError, ValueError):
                booking_args["person_id"] = c_person_id
        else:
            booking_args["person_id"] = c_person_id
    return {
        "messages": [
            SystemMessage(
                content=(
                    "Book now. Call 'book_appointment' exactly once with: "
                    + json.dumps(booking_args)
                )
            )
        ]
    }


def route_book_at_selected(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Book a slot when the user replies with “at HH:MM” (and optional minutes).

    Args
    - last: The latest user message containing a start time (and optional duration).
    - msgs: Conversation history including a 'PENDING_SLOTS' marker and identity.

    Returns
    - Dict with a deterministic 'book_appointment' instruction; otherwise None.

    Raises
    - None
    """
    text_raw = last.content or ""
    m_at = TIME_AT_RE.search(text_raw)
    dur_from_text = None
    m_dur = re.search(r"\b(\d{1,3})\s*(?:min|mins|minutes)\b", text_raw, re.I)
    if m_dur:
        try:
            dur_from_text = int(m_dur.group(1))
        except Exception:
            dur_from_text = None
    if not m_at:
        return None
    start_txt = m_at.group(1)
    start_hhmm = _norm_hhmm_12to24(start_txt)
    pending_slots, pending_client = _load_pending_slots_and_client(msgs)
    if not (
        pending_slots and isinstance(pending_slots.get("slots"), list) and start_hhmm
    ):
        return None
    chosen = None
    for s in pending_slots["slots"]:
        sl = s.get("start_local")
        el = s.get("end_local")
        if not (sl and el and len(sl) >= 16 and len(el) >= 16):
            continue
        if sl[11:16] != start_hhmm:
            continue
        sh, sm = map(int, sl[11:16].split(":"))
        eh, em = map(int, el[11:16].split(":"))
        slot_len = (eh * 60 + em) - (sh * 60 + sm)
        if dur_from_text and int(dur_from_text) != slot_len:
            continue
        chosen = (sl, slot_len)
        break
    if not chosen:
        return None
    start_local, duration_min = chosen
    ovl = _failed_overlap_time(msgs)
    if ovl and start_local == ovl:
        return {
            "messages": [
                AIMessage(
                    content="That exact time just failed due to a conflict. Please pick another slot from the list above."
                )
            ]
        }
    c_name = c_email = c_person_id = None
    if pending_client:
        c_name, c_email, c_person_id = _identity_from_pending_client(pending_client)
    booking_args = {
        "start_local": start_local,
        "duration_min": int(duration_min),
        "client_name": c_name,
        "client_email": c_email,
    }
    if c_person_id:
        if isinstance(c_person_id, (int, str)):
            try:
                booking_args["person_id"] = int(c_person_id)
            except (TypeError, ValueError):
                booking_args["person_id"] = c_person_id
        else:
            booking_args["person_id"] = c_person_id
    return {
        "messages": [
            SystemMessage(
                content=(
                    "Book now. Call 'book_appointment' exactly once with: "
                    + json.dumps(booking_args)
                )
            )
        ]
    }


def _client_payload_from_pending(pending_client: dict | None) -> tuple[dict, bool]:
    payload: dict[str, Any] = {}
    if not isinstance(pending_client, dict):
        return payload, False
    chosen_person_id = pending_client.get("chosen_person_id")
    base_person_id = pending_client.get("person_id")
    people = pending_client.get("people") or []
    ambiguous = (
        isinstance(people, list)
        and len(people) > 1
        and not (chosen_person_id or base_person_id)
    )
    _, _, person_hint = _identity_from_pending_client(pending_client)
    person_id_value = chosen_person_id or base_person_id or person_hint
    if person_id_value is not None:
        if isinstance(person_id_value, (int, str)):
            try:
                payload["person_id"] = int(person_id_value)
            except Exception:
                payload["person_id"] = person_id_value
        else:
            payload["person_id"] = person_id_value
    name, email, _ = _identity_from_pending_client(pending_client)
    if name:
        payload["client_name"] = name
    if email:
        payload["client_email"] = email
    query = pending_client.get("client_query") or pending_client.get("name")
    if query:
        payload["client_query"] = query
    return payload, ambiguous


def route_pending_client_instruction(
    last_any: Any, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Emit booking instructions once client identity is resolved.

    Args
    - last_any: The last message which should be a 'PENDING_CLIENT' marker.
    - msgs: Conversation history to determine booking intent and cadence.

    Returns
    - Dict with a single or recurring booking instruction including identity
      fields; otherwise None.

    Raises
    - None
    """
    if not (
        isinstance(last_any, (AIMessage, SystemMessage))
        and isinstance(getattr(last_any, "content", None), str)
    ):
        return None
    text = last_any.content
    if not text.startswith("PENDING_CLIENT:"):
        return None
    try:
        pending_blob = json.loads(text[len("PENDING_CLIENT:") :])
    except Exception:
        pending_blob = None
    payload, ambiguous = _client_payload_from_pending(pending_blob)
    if ambiguous:
        people = (pending_blob or {}).get("people") or []
        names = ", ".join(
            p.get("full_name") or "(unnamed)" for p in people if isinstance(p, dict)
        )
        ask = "I found multiple matching people. Who should I book this for?" + (
            f" Choices: {names}." if names else ""
        )
        return {"messages": [AIMessage(content=ask)]}
    # If the most recent human asked to book, produce instructions with identity
    last_human = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if last_human and _is_booking_intent(last_human.content or ""):
        from agent.graph_parts.booking_messages import (
            build_booking_instruction_for_payload as _build,
            build_recurring_instruction_for_payload as _build_rec,
        )

        parsed_dur = _extract_duration_min(last_human.content or "")
        from agent.graph import _is_recurring_booking as _recurring

        if _recurring(last_human.content or ""):
            instruction = _build_rec(payload, parsed_dur)
        else:
            instruction = _build(payload, parsed_dur)
        return {"messages": [SystemMessage(content=instruction)]}
    return None
