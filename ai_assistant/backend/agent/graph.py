from __future__ import annotations
import json
import logging
import os
import re
from datetime import datetime as _dt
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import facts
from agent.llm import bound_llm_with_tools
from agent.memory import get_checkpointer
from agent.prompts import SYSTEM_PROMPT
from agent.tool_registry import ALL_TOOLS, get_tool_intents
from agent.graph_parts.tool_calls import _tc, _tc_name
from agent.graph_parts.types import AgentState
from agent.graph_parts.intent_patterns import (
    _is_booking_intent,
    _needs_duration_question,
    _extract_duration_min,
)
from agent.graph_parts.time_parse import (
    TIME_AT_RE,
    TIME_FROM_TO_RE,
    TIME_RANGE_RE,
    _parse_weekday_from_text,
    _extract_slot_minutes,
    _norm_hhmm_12to24,
)
from agent.graph_parts.booking_messages import (
    make_booking_llm_instruction as _make_booking_llm_instruction,
    build_booking_instruction_for_payload as _build_booking_instruction_for_payload,
    build_recurring_instruction_for_payload as _build_recurring_instruction_for_payload,
)
from agent.graph_parts.run_intercepts import (
    handle_pre_tool_intercepts as _pre_intercepts,
)

try:
    import agent.tool_ctx as _tool_ctx_mod

    tz_var = _tool_ctx_mod.tz_var
except Exception:

    class _TzVarStub:
        def get(self):
            return None

    tz_var = _TzVarStub()
from agent.graph_parts.post_tools import (
    guard_repeat_tool_messages as _pt_guard_repeat,
    handle_find_slots as _pt_handle_find_slots,
    handle_email_draft as _pt_handle_email_draft,
    handle_no_availability as _pt_handle_no_availability,
    handle_tool_error as _pt_handle_tool_error,
    handle_choice_required as _pt_handle_choice_required,
    handle_missing_identity as _pt_handle_missing_identity,
    handle_appt_conflict as _pt_handle_appt_conflict,
    handle_appt_overlap as _pt_handle_appt_overlap,
    handle_email_send_result as _pt_handle_email_send_result,
    handle_calendar_snapshot as _pt_handle_calendar_snapshot,
    handle_is_public_holiday as _pt_handle_is_public_holiday,
    handle_get_public_holidays as _pt_handle_get_public_holidays,
    handle_add_special_opening as _pt_handle_add_special_opening,
    handle_create_recurring_openings as _pt_handle_create_recurring_openings,
    handle_update_client as _pt_handle_update_client,
    handle_update_person as _pt_handle_update_person,
    handle_update_appointment as _pt_handle_update_appointment,
    handle_update_appointment_details_summary as _pt_handle_update_appointment_details_summary,
    handle_reschedule_appointment as _pt_handle_reschedule_appointment,
    handle_booking_success as _pt_handle_booking_success,
    handle_get_appointment_details as _pt_handle_get_appointment_details,
    handle_update_details_error_or_success as _pt_handle_update_details_error_or_success,
    handle_cancel_appointment_success as _pt_handle_cancel_appointment_success,
    handle_pending_email_human as _pt_handle_pending_email_human,
)
from agent.graph_parts.router import (
    route_holiday_intents as _rt_holiday,
    route_lessons_overview as _rt_lessons,
    route_schedule_overview as _rt_schedule,
    route_availability_overview as _rt_availability,
    route_cancel_appointment_intent as _rt_cancel_appt,
    route_client_phone_update as _rt_phone_update,
    route_duration_only_followup as _rt_duration_only,
    route_pending_choices as _rt_choices,
    route_person_after_ambiguity as _rt_person_after_ambiguity,
    route_pending_identity as _rt_pending_identity,
    route_ambiguous_client_before_booking as _rt_ambiguous_client,
    route_book_range_selected as _rt_book_range,
    route_book_from_to_selected as _rt_book_fromto,
    route_book_at_selected as _rt_book_at,
    route_add_client_refusal as _rt_add_client_refusal,
    route_email_message_intent as _rt_email_intent,
    route_truncate_after as _rt_truncate_after,
    route_opening_slot_length_followup as _rt_opening_slot_len_followup,
    route_pending_client_instruction as _rt_pending_client_instruction,
    route_time_off_status as _rt_time_off_status,
)

from typing import Any as _AnyType

tz_var: _AnyType
_toolcalls: _AnyType | None = None
_TOOL_INTENTS = get_tool_intents()
_ALLOWED_TOOL_NAMES = set((_TOOL_INTENTS or {}).keys())


def _tool_policy_guard(
    msgs: list[Any], config: Optional[RunnableConfig]
) -> dict[str, list[Any]] | None:
    """Block unexpected tool calls or missing owner context."""
    if not msgs:
        return None
    last = msgs[-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return None

    cfg = (
        config.get("configurable", {})
        if isinstance(config, dict)
        else getattr(config, "configurable", {}) or {}
    )
    owner_id = cfg.get("user_id") or cfg.get("owner_id") or None
    tool_calls = last.tool_calls or []
    bad_tools = []
    for tc in tool_calls:
        name = _tc_name(tc)
        if name not in _ALLOWED_TOOL_NAMES:
            bad_tools.append(name or "(unknown)")
    if bad_tools:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Sorry, I blocked an unexpected action. "
                        f"Invalid tool call(s): {', '.join(bad_tools)}."
                    )
                )
            ]
        }
    if not owner_id:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Sorry, I can't run that without an owner context. Please sign in again."
                    )
                )
            ]
        }
    return None


APPT_OVERLAP_RE = re.compile(r"APPT_OVERLAP:(\{.*\})", re.S)
NO_AVAILABILITY_RE = re.compile(r"NO_AVAILABILITY:(\{.*\})", re.S)
REMIND_WHO_RE = re.compile(r"(?:to|for)\s+(?:remind\s+)?([A-Za-z][\w .'-]+)", re.I)
SEND_EMAIL_RE = re.compile(r"^SEND_EMAIL:\s*(\{.*\})\s*$", re.S)
CANCEL_EMAIL_RE = re.compile(r"^CANCEL_EMAIL\s*$", re.I)

CONFIRM_RE = re.compile(r"CONFIRM_REQUIRED:(\{.*\})", re.S)
USER_CONFIRM_RE = re.compile(r"\b(confirm|proceed|yes|go ahead|do it)\b", re.I)
USER_CANCEL_RE = re.compile(r"\b(cancel|stop|nevermind|never mind)\b", re.I)
APPT_CONFLICT_RE = re.compile(r"APPT_CONFLICT:(\{.*\})", re.S)
CANCEL_APPT_RE = re.compile(r"\bcancel\b.*\b(appointment|lesson)\b", re.I)
EVERY_RE = re.compile(r"\b(every|each)\b", re.I)
TOMORROW_RE = re.compile(r"\b(tomorrow|today|next\s+\w+|\d{4}-\d{2}-\d{2})\b", re.I)
CLIENT_NEXT_RE = re.compile(
    r"when\s+is\s+(.+?)'s\s+next\s+(appointment|lesson)\??", re.I
)
CLIENT_DAY_RE = re.compile(
    r"does\s+(.+?)\s+have\s+(?:an\s+|any\s+)?(appointments?|lessons?)\s+(today|tomorrow|\d{4}-\d{2}-\d{2})\??",
    re.I,
)
MISSING_IDENTITY_RE = re.compile(
    r"(Appointment identity required|Need client_email|Need client_email or client_name|Need client_email to create a new person)",
    re.I,
)
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
BROADCAST_CLIENTS_RE = re.compile(r"\b(?:all|my)\s+clients\b|\bclients\b", re.I)
PHONE_10D_RE = re.compile(r"\b(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
ADD_PERSON_RE = re.compile(
    r"\badd\b.*\b(person|child|kid|member)\b(?:\s+named\s+)?([A-Za-z][\w .'-]+)?", re.I
)
HOLIDAY_RE = re.compile(r"\bholidays?\b", re.I)

CHOICE_RE = re.compile(r"CHOICE_REQUIRED:(\{.*\})", re.S)
BOOKING_SUCCESS_CLAIM_RE = re.compile(
    r"\b(appointment|appt|lesson)\b[\s\S]*\b(booked|scheduled|confirmed)\b|\b(booked|scheduled|confirmed)\b[\s\S]*\b(appointment|appt|lesson)\b",
    re.I,
)
_MONTH_WORD_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+\d{1,2}(?:,)?\s+\d{4}\b",
    re.I,
)


def _tool_hint_for_text(text: str) -> str | None:
    """Build a lightweight tool-intent hint block for the LLM.

    Uses simple keyword matches against TOOL_INTENTS metadata to highlight
    likely tools for the current user request. This is additive guidance only;
    it does not change tool schemas or hard-route to a specific tool.
    """
    if not text:
        return None
    low = text.lower()
    scores: list[tuple[str, int]] = []
    for tool_name, meta in (_TOOL_INTENTS or {}).items():
        kws = meta.get("keywords") or []
        if not isinstance(kws, (list, tuple)):
            continue
        score = 0
        for kw in kws:
            if not isinstance(kw, str):
                continue
            k = kw.strip().lower()
            if not k:
                continue
            if k in low:
                score += 1
        if score:
            scores.append((tool_name, score))
    if not scores:
        return None
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:3]
    lines: list[str] = []
    for name, _score in top:
        meta = _TOOL_INTENTS.get(name) or {}
        summary = meta.get("summary") or ""
        kws = meta.get("keywords") or []
        kw_str = ", ".join([k for k in kws if isinstance(k, str)]) if kws else ""
        line = f"- {name}: {summary}"
        if kw_str:
            line += f" (keywords: {kw_str})"
        lines.append(line)
    if not lines:
        return None
    header = (
        "Tool intent hints:\n"
        "The following tools may be relevant for this request. Prefer them when "
        "deciding tool calls if they match the user's intent:\n"
    )
    return header + "\n".join(lines)


def _extract_confirm_payload(text: str) -> dict | None:
    """Extract and parse a CONFIRM_REQUIRED JSON payload from an AI message.

    Args:
        text: Full message text that may begin with 'CONFIRM_REQUIRED:{...}'.

    Returns:
        The parsed JSON object if found and valid; otherwise None.
    """
    m = CONFIRM_RE.match((text or "").strip())
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _latest_pending_client(messages: list[Any]) -> dict | None:
    """Return the most recent 'PENDING_CLIENT' payload, if any.

    Args
    - messages: Conversation history to scan.

    Returns
    - The parsed JSON payload for the latest marker; otherwise None.

    Raises
    - None
    """
    for m in reversed(messages):
        if isinstance(m, (AIMessage, SystemMessage)) and isinstance(
            getattr(m, "content", None), str
        ):
            text = m.content
            if text.startswith("PENDING_CLIENT:"):
                try:
                    return json.loads(text[len("PENDING_CLIENT:") :])
                except Exception:
                    return None
    return None


def _identity_from_pending_client(
    pending_client: dict | None,
) -> tuple[str | None, str | None, object | None]:
    """Extract name, email, and person_id hints from a pending client blob.

    Args
    - pending_client: The pending client mapping to inspect.

    Returns
    - Tuple of (client_name, client_email, person_id-like) with best-effort
      values; None entries when unavailable.

    Raises
    - None
    """
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


def _client_payload_from_pending(pending_client: dict | None) -> tuple[dict, bool]:
    """Build deterministic identity payload from pending client context.

    Args
    - pending_client: Pending client information emitted earlier.

    Returns
    - Tuple of (payload, ambiguous). The payload includes any of
      'person_id', 'client_name', 'client_email', and 'client_query' when
      resolvable. 'ambiguous' indicates multiple people match without a chosen
      person.

    Raises
    - None
    """
    payload: dict[str, object] = {}
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


def _latest_pending_blob(messages: list[Any], prefix: str) -> dict | None:
    """Find and parse the latest AIMessage JSON payload for a given prefix.

    Args
    - messages: Conversation history to scan.
    - prefix: Message prefix such as 'PENDING_BOOKED:' to match.

    Returns
    - The parsed JSON mapping when found and valid; otherwise None.

    Raises
    - None
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(getattr(m, "content", None), str):
            text = m.content
            if text.startswith(prefix):
                try:
                    return json.loads(text[len(prefix) :])
                except Exception:
                    return None
    return None


def _latest_pending_booked(messages: list[Any]) -> dict | None:
    """Return the most recent 'PENDING_BOOKED' payload if present."""
    return _latest_pending_blob(messages, "PENDING_BOOKED:")


def _latest_pending_book_failed(messages: list[Any]) -> dict | None:
    """Return the most recent 'PENDING_BOOK_FAILED' payload if present."""
    return _latest_pending_blob(messages, "PENDING_BOOK_FAILED:")


def _is_recurring_booking(text: str) -> bool:
    """Detect phrases that imply a recurring booking request.

    Args
    - text: Free text to examine.

    Returns
    - True if recurring booking phrases are present; else False.

    Raises
    - None
    """
    if not text:
        return False
    low = text.lower()
    recurring_words = [
        "every week",
        "each week",
        "weekly",
        "biweekly",
        "recurring",
        "repeat",
        "every other",
    ]
    return any(k in low for k in recurring_words)


def _sanitize_tool_pairs(msgs: list[Any]) -> list[Any]:
    """Normalize tool call/reply pairs while preserving order.

    Args
    - msgs: Message history sequence.

    Returns
    - A list where assistant tool_calls are directly followed by a ToolMessage,
      and orphan ToolMessages are removed.

    Raises
    - None
    """
    cleaned = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            if i + 1 < len(msgs) and isinstance(msgs[i + 1], ToolMessage):
                cleaned.append(m)
                cleaned.append(msgs[i + 1])
                i += 2
            else:
                i += 1
        elif isinstance(m, ToolMessage):
            if (
                cleaned
                and isinstance(cleaned[-1], AIMessage)
                and getattr(cleaned[-1], "tool_calls", None)
            ):
                cleaned.append(m)
            i += 1
        else:
            cleaned.append(m)
            i += 1
    return cleaned


def _extract_explicit_date(text: str) -> str | None:
    """Extract an explicit date in ISO format if present.

    Supports forms: YYYY-MM-DD, MM/DD/YYYY, 'Month D, YYYY' or 'Mon D, YYYY'
    (comma optional), 'Month D' or 'Mon D' (assumes current year), and
    'Month D this|next year'. Returns the first valid match.

    Args:
        text: Free-form user text potentially containing a date.

    Returns:
        ISO date string (YYYY-MM-DD) if detected; otherwise None.
    """
    if not text:
        return None
    # 1) ISO yyyy-mm-dd
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        try:
            d = _dt.strptime(m.group(1), "%Y-%m-%d").date()
            return d.isoformat()
        except Exception:
            pass
    # 1b) Common numeric US style mm/dd/yyyy
    m_us = re.search(r"\b(\d{1,2})\/(\d{1,2})\/(\d{4})\b", text)
    if m_us:
        try:
            mm, dd, yyyy = map(int, m_us.groups())
            d = _dt(year=yyyy, month=mm, day=dd).date()
            return d.isoformat()
        except Exception:
            pass
    # 2) Month name day, year (with or without comma)
    m2 = _MONTH_WORD_RE.search(text)
    if m2:
        frag = m2.group(0)
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
            try:
                d = _dt.strptime(frag, fmt).date()
                return d.isoformat()
            except ValueError:
                continue
    # 2b) Month name + day (no year, assume current), optionally ordinal (e.g., "Oct 13th")
    m2c = re.search(
        r"\b("
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        text,
        re.I,
    )
    if m2c:
        mon, dd_s = m2c.groups()
        try:
            today = _dt.now().date()
            year = today.year
            d = _dt.strptime(f"{mon} {int(dd_s)} {year}", "%B %d %Y").date()
            return d.isoformat()
        except Exception:
            try:
                d = _dt.strptime(f"{mon} {int(dd_s)} {year}", "%b %d %Y").date()
                return d.isoformat()
            except Exception:
                pass
    # 2c) Month name + day + (this|next) year
    m3 = re.search(
        r"\b("
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")\s+(\d{1,2})(?:st|nd|rd|th)?\s+(this|next)\s+year\b",
        text,
        re.I,
    )
    if m3:
        mon, dd2_s, which = m3.groups()
        try:
            today = _dt.now().date()
            year = today.year + (1 if which.lower() == "next" else 0)
            d = _dt.strptime(f"{mon} {int(dd2_s)} {year}", "%B %d %Y").date()
            return d.isoformat()
        except Exception:
            try:
                d = _dt.strptime(f"{mon} {int(dd2_s)} {year}", "%b %d %Y").date()
                return d.isoformat()
            except Exception:
                pass
    return None


def _summarize_if_needed(
    msgs: list[Any],
    existing_summary: str | None,
    limit: int = 40,
) -> tuple[list[Any], str | None, str | None]:
    """Trim long histories and produce a running summary chunk.

    Args
    - msgs: Full conversation history.
    - existing_summary: Prior running summary text, if any.
    - limit: Number of human/ai turns to retain in the tail segment.

    Returns
    - Tuple of (messages_tail, combined_summary, new_summary_chunk). The tail
      preserves tool-call pairs and never starts on a ToolMessage. The combined
      summary is capped to roughly 4k chars.

    Raises
    - None
    """
    if not msgs:
        return msgs, existing_summary, None

    human_ai = 0
    cut = 0
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if getattr(m, "type", "") in ("human", "ai"):
            human_ai += 1
        if human_ai > limit:
            cut = i + 1
            break

    if human_ai <= limit:
        return msgs, existing_summary, None

    while cut < len(msgs) and getattr(msgs[cut], "type", "") == "tool":
        if (
            cut - 1 >= 0
            and isinstance(msgs[cut - 1], AIMessage)
            and getattr(msgs[cut - 1], "tool_calls", None)
        ):
            cut -= 1
            break
        cut += 1

    if (
        0 < cut < len(msgs)
        and isinstance(msgs[cut - 1], AIMessage)
        and getattr(msgs[cut - 1], "tool_calls", None)
    ):
        cut -= 1

    head = msgs[:cut]
    tail = msgs[cut:]

    head_chat_text = "\n".join(
        f"{m.type.upper()}: {getattr(m, 'content', '')}"
        for m in head
        if getattr(m, "type", "") in ("human", "ai") and bool(getattr(m, "content", ""))
    ).strip()

    if not head_chat_text:
        return tail, existing_summary, None

    combined = head_chat_text
    if existing_summary:
        combined = f"{existing_summary}\n{head_chat_text}"[-4000:]
    else:
        combined = head_chat_text[-4000:]

    return tail, combined, head_chat_text


def post_tools(
    state: AgentState, config: Optional[RunnableConfig] = None
) -> dict[str, list[Any]]:
    """Post-process ToolNode results before looping back to the LLM.

    Adds guardrails for repeated identical tool outputs, handles special UI
    flows (e.g., email approval), and may emit follow-up tool calls or short
    assistant messages.

    Args
    - state: Current graph state containing message history.
    - config: Optional runnable config (unused here but part of node signature).

    Returns
    - Partial state update with appended messages under the 'messages' key.

    Raises
    - None
    """
    msgs = state["messages"]
    last = msgs[-1] if msgs else None
    if isinstance(last, ToolMessage):
        try:
            sample = (
                last.content if isinstance(last.content, str) else str(last.content)
            )
            logging.getLogger(__name__).info(
                "post_tools: ToolMessage name=%s type=%s sample=%r",
                getattr(last, "name", None),
                type(last.content).__name__,
                sample[:200],
            )
        except Exception:
            pass

    # Guard against infinite retries of the same tool with identical output
    _guard = _pt_guard_repeat(msgs)
    if _guard:
        return _guard

    # Handle email draft markers early so UI hooks always receive them,
    # even when running post_tools in isolation (e.g., unit tests).
    if isinstance(last, ToolMessage):
        _res = _pt_handle_email_draft(last)
        if _res:
            return _res
        # Fallback: some ToolMessage variants may wrap content in non-dict
        # structures. If the tool name is 'create_email_draft', still emit
        # the standard brief + pending + JSON echo + UI markers so the UI
        # and tests have consistent hooks.
        if getattr(last, "name", "") == "create_email_draft":
            try:
                payload = None
                if isinstance(last.content, dict):
                    payload = last.content.get("payload") or last.content
                elif isinstance(last.content, str):
                    try:
                        j = json.loads(last.content)
                        if isinstance(j, dict):
                            payload = j.get("payload") or j
                    except Exception:
                        payload = None
                payload = payload or {}
                draft_id = payload.get("draft_id") or ""
                subject = payload.get("subject") or ""
                to = payload.get("to") or ""
                to_name = (payload.get("to_name") or "").strip()
                text = payload.get("text") or ""
            except Exception:
                draft_id = subject = to = to_name = text = ""

            brief = AIMessage(
                content=(
                    "I drafted an email based on your request. Review the card below and tell me to send or edit it."
                )
            )
            marker = AIMessage(
                content=(
                    "PENDING_EMAIL_SEND:"
                    + json.dumps(
                        {
                            "draft_id": draft_id,
                            "to": to,
                            "to_name": to_name,
                            "subject": subject,
                            "text": text,
                        }
                    )
                )
            )
            json_msg = AIMessage(
                content=json.dumps(
                    {
                        "marker": "email_draft",
                        "payload": {
                            "draft_id": draft_id,
                            "to": to,
                            "to_name": to_name,
                            "subject": subject,
                            "text": text,
                        },
                    },
                    ensure_ascii=False,
                )
            )
            ui_msg = AIMessage(
                content=(
                    "UI:EMAIL_DRAFT:"
                    + json.dumps(
                        {
                            "draft_id": draft_id,
                            "to": to,
                            "to_name": to_name,
                            "subject": subject,
                            "text": text,
                        }
                    )
                )
            )
            return {"messages": [brief, marker, json_msg, ui_msg]}

    # --- Parse tool output once (works for str or dict) ---
    if isinstance(last, ToolMessage):
        c = last.content
        if isinstance(c, dict):
            pass
        elif isinstance(c, str):
            try:
                json.loads(c)
            except Exception:
                pass

        # ---- stash find_client result for later deterministic booking ----
        if getattr(last, "name", "") == "find_client":
            client_blob = None
            if isinstance(last.content, dict):
                client_blob = last.content
            elif isinstance(last.content, str):
                try:
                    client_blob = json.loads(last.content)
                except Exception:
                    client_blob = None
            if client_blob:
                people = client_blob.get("people", [])
                if isinstance(people, list) and len(people) > 1:
                    recent_human = None
                    for mm in reversed(msgs[:-1]):
                        if isinstance(mm, HumanMessage):
                            recent_human = mm
                            break

                    if recent_human and _is_booking_intent(recent_human.content or ""):
                        txt = (recent_human.content or "").strip()
                        low = txt.lower()
                        import re as _re

                        matched: list[dict] = []
                        for p in people:
                            nm = (p.get("full_name") or "").strip()
                            if not nm:
                                continue
                            pat = rf"\b{_re.escape(nm.lower())}\b"
                            if _re.search(pat, low):
                                matched.append(p)
                        if len(matched) == 1:
                            chosen = matched[0]
                            enhanced_client = dict(client_blob)
                            enhanced_client["chosen_person"] = chosen
                            enhanced_client["chosen_person_id"] = chosen.get(
                                "person_id"
                            )
                            enhanced_client["chosen_person_name"] = chosen.get(
                                "full_name"
                            )
                            enhanced_client["chosen_person_email"] = (
                                chosen.get("email")
                                or client_blob.get("primary_email")
                                or client_blob.get("client_email")
                                or client_blob.get("email")
                            )
                            marker = SystemMessage(
                                content="PENDING_CLIENT:" + json.dumps(enhanced_client)
                            )
                            has_time = bool(
                                TIME_RANGE_RE.search(txt) or TIME_AT_RE.search(txt)
                            )
                            if has_time:
                                return {"messages": [marker]}
                            ask = AIMessage(
                                content=f"Great! I'll book for {chosen.get('full_name')}. "
                                f"What time should I book the appointment? "
                                f"(e.g., 'tomorrow at 10am for 30 minutes')"
                            )
                            return {"messages": [marker, ask]}
                        people_names = [
                            p.get("full_name", "") for p in people if p.get("full_name")
                        ]
                        if people_names:
                            marker = SystemMessage(
                                content="PENDING_CLIENT:" + json.dumps(client_blob)
                            )
                            ask = AIMessage(
                                content=f"I found multiple people for {client_blob.get('name', 'this client')}: "
                                f"{', '.join(people_names)}. "
                                f"Who is this appointment for? Please specify the exact name."
                            )
                            return {"messages": [marker, ask]}
                marker = SystemMessage(
                    content="PENDING_CLIENT:" + json.dumps(client_blob)
                )
                try:
                    acct_name = client_blob.get("name") or "client"
                    primary_email = client_blob.get("primary_email") or ""
                    primary_phone = client_blob.get("primary_phone") or ""
                    ppl = client_blob.get("people") or []

                    def _fmt_p(p):
                        nm = p.get("full_name") or "(unnamed)"
                        em = p.get("email") or ""
                        return f"{nm}{(' <' + em + '>') if em else ''}"

                    ppl_str = (
                        ", ".join(_fmt_p(p) for p in ppl)
                        if isinstance(ppl, list) and ppl
                        else "(no people)"
                    )
                    bits = []
                    bits.append(f"Found {acct_name}.")
                    if primary_email:
                        bits.append(f"Email: {primary_email}.")
                    if primary_phone:
                        bits.append(f"Phone: {primary_phone}.")
                    bits.append(f"People: {ppl_str}.")
                    summary = " ".join(bits)
                except Exception:
                    summary = "Found client."

                # If the user's last message was a booking request, immediately instruct booking
                recent_human = next(
                    (m for m in reversed(msgs[:-1]) if isinstance(m, HumanMessage)),
                    None,
                )
                rh_text = (recent_human.content if recent_human else "") or ""
                if recent_human and _is_booking_intent(recent_human.content or ""):
                    payload, ambiguous = _client_payload_from_pending(client_blob)
                    if not ambiguous:
                        parsed_dur = _extract_duration_min(recent_human.content or "")
                        if _is_recurring_booking(recent_human.content or ""):
                            instruction_text = _build_recurring_instruction_for_payload(
                                payload, parsed_dur
                            )
                        else:
                            instruction_text = _build_booking_instruction_for_payload(
                                payload, parsed_dur
                            )
                        return {
                            "messages": [
                                AIMessage(content=summary),
                                marker,
                                SystemMessage(content=instruction_text),
                            ]
                        }

                # 1) Phone update intent
                if recent_human and re.search(
                    r"\b(phone|phone number)\b", rh_text, re.I
                ):
                    mnum = PHONE_10D_RE.search(rh_text)
                    digits = None
                    if mnum:
                        digits = "".join(ch for ch in mnum.group(0) if ch.isdigit())
                    else:
                        alld = "".join(ch for ch in rh_text if ch.isdigit())
                        if len(alld) == 10:
                            digits = alld
                    if digits and len(digits) == 10:
                        formatted = f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
                        acct_id = client_blob.get("id")
                        if acct_id:
                            return {
                                "messages": [
                                    AIMessage(content=summary),
                                    marker,
                                    AIMessage(
                                        content="",
                                        tool_calls=[
                                            _tc(
                                                "update_client",
                                                {
                                                    "client_id": str(acct_id),
                                                    "phone": formatted,
                                                },
                                                id="call_update_phone",
                                            )
                                        ],
                                    ),
                                ]
                            }
                    ask = "Please provide a valid 10-digit phone number (e.g., 416-555-1212)."
                    return {
                        "messages": [
                            AIMessage(content=summary),
                            marker,
                            AIMessage(content=ask),
                        ]
                    }

                # 2) Add person intent
                apm = ADD_PERSON_RE.search(rh_text)
                if recent_human and apm:
                    name_guess = (apm.group(2) or "").strip() or None
                    email_guess = None
                    em = EMAIL_RE.search(rh_text)
                    if em:
                        email_guess = em.group(0)
                    acct_id = client_blob.get("id")
                    if acct_id and name_guess:
                        args = {
                            "client_id": str(acct_id),
                            "add_person_name": name_guess,
                        }
                        if email_guess:
                            args["add_person_email"] = email_guess
                        return {
                            "messages": [
                                AIMessage(content=summary),
                                marker,
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        _tc("update_client", args, id="call_add_person")
                                    ],
                                ),
                            ]
                        }
                    return {
                        "messages": [
                            AIMessage(content=summary),
                            marker,
                            AIMessage(content="What is the person's full name to add?"),
                        ]
                    }

                return {"messages": [AIMessage(content=summary), marker]}

        # ---- stash find_slots so "book 9-9:30" can map to real start_local ----
        if getattr(last, "name", "") == "find_slots":
            _res = _pt_handle_find_slots(last, msgs)
            if _res:
                return _res

    # --- Final guardrail: avoid hallucinating booking success ---
    try:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and isinstance(
            getattr(last_msg, "content", None), str
        ):
            text = (last_msg.content or "").strip()
            if BOOKING_SUCCESS_CLAIM_RE.search(text):
                booked_blob = _latest_pending_booked(state["messages"])
                if not booked_blob:
                    safe = AIMessage(
                        content=(
                            "I haven’t placed a booking yet. If you’d like me to proceed, "
                            "please confirm the dates and durations (or say ‘book now’), and I’ll book them using the tools."
                        )
                    )
                    return {"messages": [safe]}
    except Exception:
        pass

        _res = _pt_handle_email_send_result(last)
        if _res:
            return _res
        _res = _pt_handle_calendar_snapshot(last)
        if _res:
            return _res
        _res = _pt_handle_is_public_holiday(last, msgs)
        if _res:
            return _res

        _res = _pt_handle_get_public_holidays(last, msgs)
        if _res:
            return _res
        _res = _pt_handle_add_special_opening(last)
        if _res:
            return _res
        _res = _pt_handle_create_recurring_openings(last)
        if _res:
            return _res

        _res = _pt_handle_get_public_holidays(last, msgs)
        if _res:
            return _res
        _res = _pt_handle_update_client(last, msgs)
        if _res:
            return _res
        _res = _pt_handle_update_person(last, msgs)
        if _res:
            return _res
        _res = _pt_handle_update_appointment(last, msgs)
        if _res:
            return _res

        _res = _pt_handle_update_appointment_details_summary(last)
        if _res:
            return _res
        _res = _pt_handle_reschedule_appointment(last, msgs)
        if _res:
            return _res
        _res = _pt_handle_cancel_appointment_success(last)
        if _res:
            return _res

        # 3) Hard error normalization
        _res = _pt_handle_tool_error(last)
        if _res:
            return _res

        # 4) choice_required
        _res = _pt_handle_choice_required(last)
        if _res:
            return _res

        # 5) Missing identity → ask + stash
        _res = _pt_handle_missing_identity(last, msgs)
        if _res:
            return _res

        # 6) Appointment conflict pretty message
        _res = _pt_handle_appt_conflict(last)
        if _res:
            return _res

        _res = _pt_handle_appt_overlap(last, msgs)
        if _res:
            return _res

        if isinstance(last.content, str):
            _res = _pt_handle_no_availability(last, msgs)
            if _res:
                return _res
        _res = _pt_handle_booking_success(last, msgs)
        if _res:
            return _res
        _res = _pt_handle_get_appointment_details(last, msgs)
        if _res:
            return _res
        if getattr(last, "name", "") == "update_appointment_details":
            if (
                isinstance(last.content, str)
                and "update_appointment_details failed" in last.content
            ):
                booking_data = _latest_pending_booked(msgs[:-1]) or {}
                instruction = _make_booking_llm_instruction(
                    booking_data or {"status": "booked"}, "booked"
                )
                return {
                    "messages": [
                        AIMessage(
                            content="Booked and confirmed. I couldn’t link this to a specific person automatically."
                        ),
                        instruction,
                    ]
                }

        _res = _pt_handle_update_details_error_or_success(last, msgs)
        if _res:
            return _res
        if getattr(last, "name", "") == "cancel_appointment":
            payload_cancel: dict | None = None
            if isinstance(last.content, dict):
                payload_cancel = last.content
            elif isinstance(last.content, str):
                try:
                    payload_cancel = json.loads(last.content)
                except Exception:
                    payload_cancel = None
            if (
                isinstance(payload_cancel, dict)
                and (payload_cancel.get("status") or "").lower() == "canceled"
            ):
                return {"messages": [AIMessage(content="✅ Appointment canceled.")]}

    # 7) If the user replied and we have a pending draft, interpret it
    if isinstance(last, HumanMessage):
        _res = _pt_handle_pending_email_human(last, msgs)
        if _res:
            return _res

    return {"messages": []}


def build_graph() -> Any:
    """Compose and return the conversation LangGraph.

    Behavior
    - Routes time-off and cutoff intents to the appropriate tools.
    - Uses deterministic router helpers for openings, availability, and more.
    - Asks for duration only when user phrasing indicates booking intent.
    - Summarizes long histories before sending to the model.

    Returns
    - The compiled LangGraph instance ready to be executed.

    Raises
    - None
    """
    # 1) Leave LLM and tools *unscoped* here; we’ll pass user_id at call time.
    llm = bound_llm_with_tools()

    tool_node = ToolNode(tools=ALL_TOOLS)
    from agent.tool_ctx import owner_id_var, tz_var

    def run_tools(
        state: AgentState, config: Optional[RunnableConfig] = None
    ) -> dict[str, list[Any]]:
        """Execute tool calls with intercepts and context vars.

        Args
        - state: Current agent state containing messages.
        - config: Runnable configuration with 'configurable' metadata.

        Returns
        - Partial state update with appended messages under 'messages'.

        Raises
        - None
        """
        try:
            c = (
                config.get("configurable")
                if isinstance(config, dict)
                else getattr(config, "configurable", None)
            )
            logging.getLogger(__name__).info(
                "graph.run_tools config.configurable=%s", c
            )
        except Exception:
            c = None
        try:
            msgs = state["messages"]
            last = msgs[-1] if msgs else None
            policy_block = _tool_policy_guard(msgs, config)
            if policy_block:
                return policy_block
            res = _pre_intercepts(msgs, last, c, tz_var)
            if res:
                return res
        except Exception as e:
            logging.getLogger(__name__).warning("run_tools pre-intercept failed: %s", e)
        tok_owner = tok_tz = None
        try:
            if isinstance(c, dict):
                if (oid := c.get("user_id") or c.get("owner_id")) is not None:
                    tok_owner = owner_id_var.set(str(oid))
                if (tz := c.get("tz")) is not None:
                    tok_tz = tz_var.set(str(tz))
            return tool_node.invoke(state, config=config)
        finally:
            if tok_owner is not None:
                owner_id_var.reset(tok_owner)
            if tok_tz is not None:
                tz_var.reset(tok_tz)

    # ---------------- Router ----------------
    def route_or_llm(
        state: AgentState, config: Optional[RunnableConfig] = None
    ) -> dict[str, list[Any]]:
        """Route user intents or fall back to the LLM prompt.

        Args
        - state: Current agent state containing message history.
        - config: Runnable configuration (unused directly here).

        Returns
        - Partial state update with new messages to drive next steps.

        Raises
        - None
        """
        msgs = state["messages"]
        last = msgs[-1] if msgs else None
        if isinstance(last, (AIMessage, SystemMessage)):
            _res = _rt_pending_client_instruction(last, msgs)
            if _res:
                return _res

        # --- User replied after a confirm prompt ---
        if isinstance(last, HumanMessage):
            txt = last.content or ""
            _res = _rt_cancel_appt(last)
            if _res:
                return _res
            _res = _rt_time_off_status(last)
            if _res:
                return _res
            if USER_CANCEL_RE.search(txt):
                return {"messages": [AIMessage(content="Okay, I won’t book that.")]}

            if USER_CONFIRM_RE.search(txt):
                payload = None
                for m in reversed(msgs[:-1]):
                    if isinstance(m, (AIMessage, SystemMessage)) and isinstance(
                        m.content, str
                    ):
                        p = _extract_confirm_payload(m.content)
                        if p:
                            payload = p
                            break
                if payload:
                    tool_name = (
                        payload.get("tool") if isinstance(payload, dict) else None
                    )
                    args = payload.get("args") if isinstance(payload, dict) else None
                    if tool_name in (
                        "book_appointment",
                        "reschedule_appointment",
                        "add_special_opening",
                        "create_recurring_openings",
                    ) and isinstance(args, dict):
                        return {
                            "messages": [
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        _tc(
                                            tool_name,
                                            args,
                                            id=f"call_confirmed_{tool_name}",
                                        )
                                    ],
                                )
                            ]
                        }
                    return {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    _tc(
                                        "book_appointment",
                                        payload if isinstance(payload, dict) else {},
                                        id="call_confirmed_booking",
                                    )
                                ],
                            )
                        ]
                    }

                if isinstance(last, HumanMessage):
                    t = last.content or ""
                    low = t.lower()
                    has_opening_word = any(
                        k in low
                        for k in [
                            "opening",
                            "openings",
                            "availability",
                            "available",
                            "open up",
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
                    if (
                        any(k in low for k in ["add ", "create ", "set ", "make "])
                        and has_opening_word
                        and not has_timeoff_word
                    ):
                        slot = _extract_slot_minutes(t)
                        if not slot:
                            return {
                                "messages": [
                                    AIMessage(
                                        content="What slot length should I use for this opening (e.g., 30, 45, 60 minutes)?"
                                    )
                                ]
                            }
                        m = TIME_FROM_TO_RE.search(t) or TIME_RANGE_RE.search(t)
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

                        # Resolve the date anchor deterministically in owner's timezone
                        anchor_iso = _extract_explicit_date(t)
                        if not anchor_iso:
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
                                        _dt.now().date()
                                        + __import__("datetime").timedelta(days=1)
                                    ).isoformat()
                            elif re.search(r"\btoday\b", low):
                                try:
                                    from zoneinfo import ZoneInfo

                                    tzname = tz_var.get() or "America/Toronto"
                                    anchor_iso = (
                                        _dt.now(ZoneInfo(tzname)).date().isoformat()
                                    )
                                except Exception:
                                    anchor_iso = _dt.now().date().isoformat()
                            else:
                                wd = _parse_weekday_from_text(t)
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
                                        base
                                        + __import__("datetime").timedelta(days=delta)
                                    ).isoformat()

                        if not anchor_iso:
                            return {
                                "messages": [
                                    AIMessage(
                                        content="Which day should I add this opening for (today, tomorrow, or YYYY-MM-DD)?"
                                    )
                                ]
                            }

                        start_local = f"{anchor_iso}T{s24}"
                        end_local = f"{anchor_iso}T{e24}"
                        args = {
                            "start_local": start_local,
                            "end_local": end_local,
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
                            + json.dumps(
                                {
                                    "tool": "add_special_opening",
                                    "args": args,
                                    "date": day,
                                }
                            )
                        )
                        holiday_call = AIMessage(
                            content="",
                            tool_calls=[
                                _tc(
                                    "is_public_holiday",
                                    {
                                        "date": day,
                                        "country_code": cc,
                                        "region_code": rc,
                                    },
                                    id=call_id,
                                )
                            ],
                        )
                        return {"messages": [marker, holiday_call]}
            # Light intent hint for weekly openings without regex
            if isinstance(last, HumanMessage):
                t_weekly = last.content or ""
                low_weekly = t_weekly.lower()
                if "opening" in low_weekly and (
                    "weekly" in low_weekly
                    or any(
                        d in low_weekly
                        for d in [
                            "monday",
                            "tuesday",
                            "wednesday",
                            "thursday",
                            "friday",
                            "saturday",
                            "sunday",
                        ]
                    )
                ):
                    wd = _parse_weekday_from_text(t_weekly)
                    return {
                        "messages": [
                            SystemMessage(
                                content=(
                                    "User asked to create a weekly opening. "
                                    "Call 'create_recurring_openings' once with arguments named exactly: "
                                    f"'weekday'={wd if wd is not None else 'detect from user text (0=Mon..6=Sun)'}, "
                                    "'start_hhmm', 'end_hhmm', 'slot_minutes' (default 60 if omitted), 'buffer_minutes' (0), and 'weeks' (e.g. 8)."
                                )
                            )
                        ]
                    }
        if isinstance(last, HumanMessage):
            _res = _rt_duration_only(last, msgs)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_choices(last, msgs)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            prev_message = msgs[-2] if len(msgs) >= 2 else None
            _res = _rt_person_after_ambiguity(prev_message, last, msgs)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_pending_identity(last, msgs)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_ambiguous_client(last, msgs)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_book_range(last, msgs)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_book_fromto(last, msgs)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_book_at(last, msgs)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_add_client_refusal(last)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_email_intent(last)
            if _res:
                return _res
        if isinstance(last, HumanMessage):
            _res = _rt_truncate_after(last)
            if _res:
                return _res

        # ---- Client listing / lookup → force tool usage over free-form ----
        if isinstance(last, HumanMessage):
            text_low = (last.content or "").lower().strip()
            if any(
                k in text_low
                for k in [
                    "who are my clients",
                    "list clients",
                    "show my clients",
                    "find client",
                    "lookup client",
                    "search clients",
                ]
            ):
                return {
                    "messages": [
                        SystemMessage(
                            content=(
                                "For client listing/lookup requests, call 'list_clients' (optionally with 'query') "
                                "or 'find_client' (with 'selector'). Do not answer without using the appropriate tool."
                            )
                        )
                    ]
                }

        # ---- Booking intent (prefer using parsed duration; never double-ask) ----
        if isinstance(last, HumanMessage):
            raw_text = last.content or ""
            text_low = raw_text.lower()
            has_timeoff_word = any(
                k in text_low
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
            booking_like = (
                _is_booking_intent(text_low)
                and ("@" not in text_low)
                and not has_timeoff_word
            )

            if booking_like:
                pending_client = _latest_pending_client(msgs[:-1])
                parsed_dur = _extract_duration_min(raw_text)

                if pending_client:
                    payload, ambiguous = _client_payload_from_pending(pending_client)
                    if ambiguous:
                        people = pending_client.get("people") or []
                        names = ", ".join(
                            p.get("full_name") or "(unnamed)"
                            for p in people
                            if isinstance(p, dict)
                        )
                        ask = (
                            "I found multiple matching people. Who should I book this for?"
                            + (f" Choices: {names}." if names else "")
                        )
                        return {"messages": [AIMessage(content=ask)]}

                    if payload and parsed_dur:
                        if _is_recurring_booking(raw_text):
                            instruction = _build_recurring_instruction_for_payload(
                                payload, parsed_dur
                            )
                        else:
                            instruction = _build_booking_instruction_for_payload(
                                payload, parsed_dur
                            )
                        return {"messages": [SystemMessage(content=instruction)]}
                if parsed_dur:
                    return {
                        "messages": [
                            SystemMessage(
                                content=(
                                    "User asked to book an appointment and already included a duration.\n"
                                    f"- First, resolve the client: call 'find_client' with 'selector' set to the name mentioned in the user's message.\n"
                                    f"- Then call 'book_appointment' exactly once with:\n"
                                    "    - start_local: parse from the user's message (YYYY-MM-DDTHH:MM in OWNER local time)\n"
                                    f"    - duration_min: {parsed_dur}\n"
                                    "    - client_name/client_email from the resolved client (if available). If multiple matches,\n"
                                    "      pick the exact-name match; if still ambiguous, include client_name/email from context.\n"
                                    "Do NOT ask the user for duration again."
                                )
                            )
                        ]
                    }

                if _is_booking_intent(text_low):
                    client_hint = None
                    for_match = re.search(
                        r"\bfor\s+([^,.!?]+?)(?:\s+(?:at|on|tomorrow|today|next|this)|\s*$)",
                        text_low,
                    )
                    if for_match:
                        client_hint = for_match.group(1).strip()

                    if client_hint:
                        return {
                            "messages": [
                                SystemMessage(
                                    content=f"User wants to book for '{client_hint}'. Call 'find_client' with 'selector'='{client_hint}' to resolve the client first."
                                )
                            ]
                        }

                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    "I can look up the client’s contact details. "
                                    "What duration should I use (30, 45, 60, or 120 minutes)? "
                                    "If there are multiple clients with similar names, I’ll confirm the right one."
                                )
                            )
                        ]
                    }

        # ---- Client profile edit intent (phone) → find_client → update_client ----
        if isinstance(last, HumanMessage):
            _res = _rt_phone_update(last)
            if _res:
                return _res

        # ---- Client “next / day” questions are now handled via LLM + tool metadata ----

        # ---- Day "schedule" / "agenda" → show EVERYTHING (appts + time off + openings) ----
        if isinstance(last, HumanMessage):
            _res = _rt_lessons(last)
            if _res:
                return _res

            _res = _rt_schedule(last)
            if _res:
                return _res

        # ---- Pure availability query → openings only (keep your current behavior) ----
        if isinstance(last, HumanMessage):
            raw_text = last.content or ""

            _res = _rt_availability(last)
            if _res:
                return _res

        # ---- Fallback: if booking-like but missing duration, ask for it ----
        if _needs_duration_question(msgs):
            return {
                "messages": [
                    AIMessage(
                        content="What duration should I look for (30, 45, 60, or 120 minutes)?"
                    )
                ]
            }
        # Fallback: no deterministic route; provide a lightweight tool-intent
        # hint so the LLM can choose tools using metadata rather than regexes
        # alone. This preserves existing behavior while improving orchestration.
        if isinstance(last, HumanMessage):
            hint = _tool_hint_for_text(last.content or "")
            if hint:
                return {"messages": [SystemMessage(content=hint)]}
        return {"messages": []}

    # ---------------- LLM node ----------------
    def call_model(
        state: AgentState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Call the LLM with trimmed context, facts, and system prompts.

        Args
        - state: Current agent state with messages and optional summary.
        - config: Runnable configuration including 'configurable' for owner/thread.

        Returns
        - Partial state update including the new AIMessage, and an updated
          running 'summary' when changed.

        Raises
        - None
        """
        messages = state["messages"]
        summary_so_far = state.get("summary")

        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            logging.getLogger(__name__).warning(
                "call_model saw pending tool_calls; skipping model until tools run."
            )
            return {"messages": []}

        trimmed, combined_summary, summary_chunk = _summarize_if_needed(
            messages,
            summary_so_far,
            limit=40,
        )

        trimmed = _sanitize_tool_pairs(trimmed)

        summary_text = (
            combined_summary if combined_summary is not None else summary_so_far
        )
        summary_text = (summary_text or "").strip()

        cfg_conf: dict[str, Any] = {}
        if isinstance(config, dict):
            cfg_conf = dict(config.get("configurable") or {})
        else:
            cfg_conf = dict(getattr(config, "configurable", {}) or {})

        owner_id = str(cfg_conf.get("owner_id") or cfg_conf.get("user_id") or "")
        thread_id = str(cfg_conf.get("thread_id") or "")

        last_human = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "human"),
            None,
        )
        fact_snippets = []
        if owner_id and last_human and getattr(last_human, "content", ""):
            fact_snippets = facts.fetch_facts(owner_id, last_human.content)

        prompt_messages = []
        system_prompt_msg = SystemMessage(SYSTEM_PROMPT)
        prompt_messages.append(system_prompt_msg)

        if summary_text:
            prompt_messages.append(
                SystemMessage(content=f"Conversation summary so far:\n{summary_text}")
            )

        if fact_snippets:
            fact_block = "\n".join(f"- {snippet}" for snippet in fact_snippets)
            prompt_messages.append(
                SystemMessage(content=f"Known owner facts:\n{fact_block}")
            )

        for m in trimmed:
            if isinstance(m, SystemMessage) and m.content == SYSTEM_PROMPT:
                continue
            prompt_messages.append(m)

        try:
            ai = llm.invoke(prompt_messages, config=config)
        except Exception as exc:
            logging.getLogger(__name__).exception("LLM invoke failed")
            err_text = "Sorry, I can't reach the model right now. Please retry in a few moments."
            ai = AIMessage(
                content=err_text,
                additional_kwargs={"event": "error", "reason": "llm_unavailable"},
            )
        logging.getLogger(__name__).info(
            "LLM response tool_calls=%s",
            getattr(ai, "tool_calls", None)
            or getattr(ai, "additional_kwargs", {}).get("tool_calls"),
        )

        updates: dict[str, Any] = {"messages": [ai]}
        if combined_summary is not None and combined_summary != summary_so_far:
            updates["summary"] = combined_summary
            if summary_chunk:
                if owner_id and thread_id:
                    facts.store_fact(
                        owner_id,
                        thread_id,
                        summary_chunk,
                        metadata={"kind": "conversation_summary"},
                    )
                logging.getLogger(__name__).debug(
                    "Updated running summary (len=%d)", len(combined_summary)
                )
        return updates

    # ---------------- Graph wiring ----------------
    from agent.graph_parts.build import wire_graph

    return wire_graph(
        route_or_llm=route_or_llm,
        call_model=call_model,
        run_tools=run_tools,
        post_tools=post_tools,
        checkpointer=get_checkpointer(),
    )


# --- Rebind shared tool-call helpers to modular implementations (non-invasive) ---
try:
    from agent.graph_parts import tool_calls as _toolcalls

    _tc = _toolcalls._tc
    _tc_name = _toolcalls._tc_name
    _tc_args = _toolcalls._tc_args
    _last_tool_call_and_args = _toolcalls._last_tool_call_and_args
except Exception:
    pass
