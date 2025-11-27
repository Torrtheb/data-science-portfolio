from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from datetime import datetime as _dt, timedelta as _td
import re
from agent.graph_parts.tool_calls import _tc, _last_tool_call_and_args
from agent.graph_parts.booking_messages import (
    make_pending_book_failed_message as _make_pending_book_failed_message,
    make_booking_llm_instruction as _make_booking_llm_instruction,
    make_pending_booked_message as _make_pending_booked_message,
)

try:
    import agent.tool_ctx as _tool_ctx_mod

    tz_var = _tool_ctx_mod.tz_var
except Exception:

    class _TzVarStub:
        def get(self):
            return None

    tz_var = _TzVarStub()

NO_AVAILABILITY_RE = re.compile(r"NO_AVAILABILITY:(\{.*\})", re.S)
APPT_CONFLICT_RE = re.compile(r"APPT_CONFLICT:(\{.*\})", re.S)
APPT_OVERLAP_RE = re.compile(r"APPT_OVERLAP:(\{.*\})", re.S)
MISSING_IDENTITY_RE = re.compile(
    r"(Appointment identity required|Need client_email|Need client_email or client_name|Need client_email to create a new person)",
    re.I,
)
CHOICE_RE = re.compile(r"CHOICE_REQUIRED:(\{.*\})", re.S)


def _today_owner_date() -> "_date":
    """Return today's date in the owner's timezone (falls back to UTC)."""
    try:
        from zoneinfo import ZoneInfo

        tz = tz_var.get() or "UTC"
        return _dt.now(ZoneInfo(tz)).date()
    except Exception:
        return _dt.now().date()


def guard_repeat_tool_messages(msgs: list[Any]) -> dict[str, list[Any]] | None:
    """Guard against infinite loops when identical tool outputs repeat.

    Args
    - msgs: Current message history. Compares the last two 'ToolMessage'
      entries by tool name and a small, normalized content signature.

    Returns
    - Dict with a single 'messages' key when a short-circuit reply should
      replace a repeat tool invocation; otherwise 'None'.

    Raises
    - None
    """
    last = msgs[-1] if msgs else None
    if isinstance(last, ToolMessage):
        try:
            last_name = getattr(last, "name", "") or ""
            # Normalize content to a small signature
            sig = None
            if isinstance(last.content, str):
                c = last.content.strip()
                sig = c if len(c) <= 64 else c[:64]
            elif isinstance(last.content, (list, dict)):
                sig = json.dumps(last.content, sort_keys=True)
            # If the previous ToolMessage matches name+signature, bail with a helpful reply
            prev_tools = [m for m in reversed(msgs[:-1]) if isinstance(m, ToolMessage)]
            if prev_tools:
                prev = prev_tools[0]
                prev_name = getattr(prev, "name", "") or ""
                prev_sig = None
                if isinstance(prev.content, str):
                    pc = prev.content.strip()
                    prev_sig = pc if len(pc) <= 64 else pc[:64]
                elif isinstance(prev.content, (list, dict)):
                    prev_sig = json.dumps(prev.content, sort_keys=True)
                if last_name == prev_name and sig == prev_sig:
                    # Special-case: repeated people lookup → stop looping
                    if last_name == "list_people":
                        return {
                            "messages": [
                                AIMessage(
                                    content="No people are listed under that account. I won't repeat that lookup."
                                )
                            ]
                        }
        except Exception:
            logging.getLogger(__name__).exception("guard_repeat_tool_messages failed")
    return None


def handle_find_slots(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Handle 'find_slots' results: suggest, autoselect, or ask.

    Args
    - last: The 'ToolMessage' from 'find_slots'.
    - msgs: Conversation history for context (e.g., recent 'PENDING_OVERLAP_AT').

    Returns
    - Dict with messages to present suggestions, auto-book an alternative, or
      ask for another day; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "find_slots":
            return None
        slots = None
        if isinstance(last.content, dict) and "slots" in last.content:
            slots = last.content.get("slots")
        elif isinstance(last.content, str):
            s = last.content.strip()
            # Try JSON first
            try:
                j = json.loads(s)
                if isinstance(j, dict) and "slots" in j:
                    slots = j["slots"]
            except Exception:
                slots = None
            if slots is None:
                m = re.search(r"slots\s*=\s*(\[[\s\S]*\])", s, re.S)
                if m:
                    blob = m.group(1)
                    try:
                        slots = json.loads(blob.replace("'", '"'))
                    except Exception:
                        slots = None

        if isinstance(slots, list) and slots:
            # If we just failed to book a specific start time, proactively pick the next available slot
            # and call book_appointment again with the same identity/duration.
            failing_start = None
            for m2 in reversed(msgs[:-1]):
                if isinstance(m2, AIMessage) and isinstance(
                    getattr(m2, "content", None), str
                ):
                    t = m2.content
                    if t.startswith("PENDING_OVERLAP_AT:"):
                        failing_start = t[len("PENDING_OVERLAP_AT:") :]
                        break

            tool_name, tool_args = _last_tool_call_and_args(msgs)
            can_autobook = False
            args: dict = {}
            if tool_name == "book_appointment":
                for k in ("person_id", "client_email", "client_name"):
                    if (tool_args or {}).get(k):
                        args[k] = (tool_args or {}).get(k)
                        can_autobook = True
                if (tool_args or {}).get("duration_min"):
                    args["duration_min"] = (tool_args or {}).get("duration_min")

            # choose an alternative slot (prefer the next start after the failing start; otherwise first slot)
            chosen = None
            try:
                if failing_start:
                    alts = [s for s in slots if s.get("start_local") != failing_start]
                    greater = [
                        s for s in alts if s.get("start_local", "") > failing_start
                    ]
                    chosen = greater[0] if greater else (alts[0] if alts else None)
                else:
                    chosen = slots[0]
            except Exception:
                chosen = slots[0]

            if can_autobook and chosen and chosen.get("start_local"):
                args["start_local"] = chosen.get("start_local")
                # Emit a direct tool call to re-book the closest alternative
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                _tc(
                                    "book_appointment",
                                    args,
                                    id="call_rebook_after_overlap",
                                )
                            ],
                        )
                    ]
                }

            # Otherwise, stash slots for the LLM to ask/decide
            marker = SystemMessage(
                content="PENDING_SLOTS:" + json.dumps({"slots": slots})
            )
            return {"messages": [marker]}

        elif isinstance(slots, list) and not slots:
            tool_name, tool_args = _last_tool_call_and_args(msgs)
            dur = None
            if tool_name == "find_slots":
                dur = (tool_args or {}).get("duration_minutes")
            msg = "No openings for that request"
            if dur:
                try:
                    msg = f"No {int(dur)}-minute openings for that request"
                except Exception:
                    msg = "No openings for that request"
            ask = (
                msg
                + ". Would you like me to check another day (e.g., tomorrow or next Monday)?"
            )
            return {"messages": [AIMessage(content=ask)]}
    except Exception:
        logging.getLogger(__name__).exception("handle_find_slots failed")
    return None


def handle_email_draft(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Detect email draft results and emit preview and actionable markers.

    Args
    - last: The 'ToolMessage' containing a draft payload or a JSON string.

    Returns
    - Dict with messages: brief confirmation, 'PENDING_EMAIL_SEND:{...}' marker,
      raw JSON echo, and 'UI:EMAIL_DRAFT:{...}' preview; otherwise 'None'.

    Raises
    - None
    """
    try:
        draft_payload = None
        c = last.content if isinstance(last, ToolMessage) else None
        if isinstance(c, dict) and c.get("marker") == "email_draft":
            draft_payload = c.get("payload") or {}
        elif isinstance(c, str):
            s = c.strip()
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and parsed.get("marker") == "email_draft":
                    draft_payload = parsed.get("payload") or {}
            except Exception:
                parsed = None
            if draft_payload is None and '"marker"' in s and "email_draft" in s:
                for m in re.finditer(r"\{.*?\}", s, re.S):
                    blob = m.group(0)
                    try:
                        j = json.loads(blob)
                        if isinstance(j, dict) and j.get("marker") == "email_draft":
                            draft_payload = j.get("payload") or {}
                            break
                    except Exception:
                        continue

        if draft_payload is None:
            return None

        draft_id = draft_payload.get("draft_id")
        subject = draft_payload.get("subject", "")
        to = draft_payload.get("to", "")
        to_name = (draft_payload.get("to_name") or "").strip()
        text = draft_payload.get("text", "")

        try:
            logging.getLogger(__name__).info(
                "post_tools: preparing email draft markers draft_id=%s to=%s subj=%r text_len=%d",
                draft_id,
                to,
                subject,
                len(text or ""),
            )
        except Exception:
            pass

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
        ui_msg = AIMessage(
            content="UI:EMAIL_DRAFT:"
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
        json_msg = AIMessage(
            content=json.dumps(
                {
                    "marker": "email_draft",
                    "payload": draft_payload,
                },
                ensure_ascii=False,
            )
        )
        return {"messages": [brief, marker, json_msg, ui_msg]}
    except Exception:
        logging.getLogger(__name__).exception("handle_email_draft failed")
        return None


def handle_no_availability(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Normalize 'NO_AVAILABILITY:{...}' errors into helpful follow-ups.

    Args
    - last: The 'ToolMessage' that may contain a 'NO_AVAILABILITY' payload.
    - msgs: Conversation history to recover prior request context.

    Returns
    - Dict with a human summary, a 'PENDING_BOOK_FAILED:{...}' marker, and
      optional 'find_slots' suggestion; otherwise 'None'.

    Raises
    - None
    """
    try:
        if not isinstance(last.content, str):
            return None
        m = NO_AVAILABILITY_RE.search(last.content.strip())
        if not m:
            return None

        payload = None
        try:
            payload = json.loads(m.group(1))
        except Exception:
            payload = None

        human = (payload or {}).get("human") or "That time isn't available."
        start_local = (payload or {}).get("start_local")
        end_local = (payload or {}).get("end_local")
        reason = (payload or {}).get("reason")

        details = []
        if start_local and end_local:
            details.append(f"Requested window: {start_local}–{end_local}.")
        if reason:
            details.append(f"Reason: {reason}.")
        info = " " + " ".join(details) if details else ""

        failure_data = {
            "reason": reason,
            "start_local": start_local,
            "end_local": end_local,
            "duration_minutes": (payload or {}).get("duration_min"),
            "human": human,
        }
        fail_marker = _make_pending_book_failed_message(failure_data)
        fail_instruction = _make_booking_llm_instruction(failure_data, "failed")

        tool_name, tool_args = _last_tool_call_and_args(msgs)
        day = None
        dur = None
        if tool_name == "book_appointment":
            sl = (tool_args or {}).get("start_local")
            dur = (tool_args or {}).get("duration_min")
            if isinstance(sl, str) and len(sl) >= 10:
                day = sl[:10]

        suggest_alternatives = reason in {
            "no opening covers the requested time",
            "time off",
            "conflicts with another appointment",
        }

        if day and dur and suggest_alternatives:
            msgs_out = [
                AIMessage(
                    content=human
                    + info
                    + " Here are other available times for that day:"
                ),
                fail_marker,
            ]
            msgs_out.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        _tc(
                            "find_slots",
                            {"day": day, "duration_minutes": int(dur)},
                            id="call_suggest_after_no_avail",
                        )
                    ],
                )
            )
            return {"messages": msgs_out}

        return {
            "messages": [AIMessage(content=human + info), fail_marker, fail_instruction]
        }
    except Exception:
        logging.getLogger(__name__).exception("handle_no_availability failed")
        return None


def handle_tool_error(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Intercept low-level tool invocation errors and stop blind retries.

    Args
    - last: The 'ToolMessage' content to examine for invocation errors.

    Returns
    - Dict with a system instruction to explain the error; otherwise 'None'.

    Raises
    - None
    """
    try:
        if isinstance(last.content, str) and (
            "TypeError" in last.content or "invalid keyword argument" in last.content
        ):
            return {
                "messages": [
                    SystemMessage(
                        content="Stop retrying the same tool call. Explain the error to the user succinctly and ask for corrected input only if needed."
                    )
                ]
            }
    except Exception:
        logging.getLogger(__name__).exception("handle_tool_error failed")
    return None


def handle_choice_required(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Handle 'CHOICE_REQUIRED:{...}' prompts from tools.

    Args
    - last: The 'ToolMessage' that may contain a choice requirement payload.

    Returns
    - Dict with a 'PENDING_CHOICES:{...}' marker and a clarifying question; or
      'None' if not applicable.

    Raises
    - None
    """
    try:
        if isinstance(last.content, str):
            m = CHOICE_RE.match(last.content.strip())
            if m:
                try:
                    blob = json.loads(m.group(1))
                except Exception:
                    blob = {}
                choice = blob.get("choice") or {}
                marker = AIMessage(content="PENDING_CHOICES:" + json.dumps(choice))
                ask = AIMessage(
                    content="Should I apply this change **only for that day** or **every week**?"
                )
                return {"messages": [marker, ask]}
    except Exception:
        logging.getLogger(__name__).exception("handle_choice_required failed")
    return None


def handle_missing_identity(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Handle missing appointment identity errors.

    Args
    - last: The 'ToolMessage' containing an identity-related error message.
    - msgs: Conversation history used to recover the last tool call and args.

    Returns
    - Dict with a 'PENDING_IDENTITY:{...}' marker and a request to supply
      identity; otherwise 'None'.

    Raises
    - None
    """
    try:
        if isinstance(last.content, str) and MISSING_IDENTITY_RE.search(last.content):
            tool_name, tool_args = _last_tool_call_and_args(msgs)
            if tool_name:
                tool_args = dict(tool_args or {})
                for k in ("person_id", "client_email", "client_name"):
                    tool_args.pop(k, None)
                payload = {
                    "tool": tool_name,
                    "args": tool_args,
                    "reason": "missing_identity",
                }
                marker = AIMessage(content="PENDING_IDENTITY:" + json.dumps(payload))
                ask = AIMessage(
                    content="Who is this appointment for? Please provide their **email** (preferred) or the **exact full name**."
                )
                return {"messages": [marker, ask]}
    except Exception:
        logging.getLogger(__name__).exception("handle_missing_identity failed")
    return None


def handle_appt_conflict(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Render 'APPT_CONFLICT:{...}' payloads as readable bullets.

    Args
    - last: The 'ToolMessage' potentially containing a conflict payload.

    Returns
    - Dict with a single message listing blocking appointments; otherwise 'None'.

    Raises
    - None
    """
    try:
        if isinstance(last.content, str):
            m = APPT_CONFLICT_RE.search(last.content.strip())
            if m:
                try:
                    p = json.loads(m.group(1))
                except Exception:
                    p = {}
                human = (
                    p.get("human")
                    or "Time off cannot be added because an appointment overlaps."
                )
                lines = p.get("blocked_appointments") or []
                bullets = "\n".join(
                    f"- {x['id']} ({x['start_local']} – {x['end_local']})"
                    for x in lines[:5]
                )
                msg = human + ("\n" + bullets if bullets else "")
                return {"messages": [AIMessage(content=msg)]}
    except Exception:
        logging.getLogger(__name__).exception("handle_appt_conflict failed")
    return None


def handle_appt_overlap(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Handle appointment overlap (unique/constraint) failures.

    Args
    - last: The 'ToolMessage' with overlap details or raw errors.
    - msgs: Conversation history to infer the original request context.

    Returns
    - Dict with a human message, optional marker, and suggestions; otherwise 'None'.

    Raises
    - None
    """
    try:
        if isinstance(last.content, str):
            m = APPT_OVERLAP_RE.search(last.content.strip())
            payload = None
            if m:
                try:
                    payload = json.loads(m.group(1))
                except Exception:
                    payload = None
            raw_overlap = ("ExclusionViolation" in last.content) or (
                "appointments_no_overlap" in last.content
            )
            if payload or raw_overlap:
                human = (payload or {}).get(
                    "human"
                ) or "That time is no longer available."
                failing_start = (payload or {}).get("start_local")
                marker = (
                    AIMessage(content="PENDING_OVERLAP_AT:" + failing_start)
                    if failing_start
                    else None
                )
                tool_name, tool_args = _last_tool_call_and_args(msgs)
                day = None
                dur = None
                if tool_name == "book_appointment":
                    sl = (tool_args or {}).get("start_local")
                    dur = (tool_args or {}).get("duration_min")
                    if isinstance(sl, str) and len(sl) >= 10:
                        day = sl[:10]
                if day and dur:
                    msgs_out = []
                    if marker:
                        msgs_out.append(marker)
                    msgs_out.append(
                        AIMessage(
                            content=human
                            + " Here are other available times for that day:"
                        )
                    )
                    msgs_out.append(
                        AIMessage(
                            content="",
                            tool_calls=[
                                _tc(
                                    "find_slots",
                                    {"day": day, "duration_minutes": int(dur)},
                                    id="call_suggest_slots",
                                )
                            ],
                        )
                    )
                    return {"messages": msgs_out}
                msgs_out = []
                if marker:
                    msgs_out.append(marker)
                msgs_out.append(AIMessage(content=human + " Please pick another time."))
                return {"messages": msgs_out}
    except Exception:
        logging.getLogger(__name__).exception("handle_appt_overlap failed")
    return None


def handle_email_send_result(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Summarize the result of 'send_approved_email'.

    Args
    - last: The 'ToolMessage' containing the send result.

    Returns
    - Dict with a success/failure summary message; otherwise 'None'.

    Raises
    - None
    """
    try:
        parsed = None
        if isinstance(last.content, dict):
            parsed = last.content
        elif isinstance(last.content, str):
            try:
                parsed = json.loads(last.content)
            except Exception:
                parsed = None
        if isinstance(parsed, dict) and {"ok", "status"} <= set(parsed.keys()):
            status = str(parsed.get("status", ""))
            if parsed.get("ok") and status.lower() == "sent":
                return {
                    "messages": [AIMessage(content="✅ Message sent successfully.")]
                }
            return {
                "messages": [
                    AIMessage(
                        content=f"Email not sent (status: {status}). You can say 'send' again or adjust the draft."
                    )
                ]
            }
    except Exception:
        logging.getLogger(__name__).exception("handle_email_send_result failed")
    return None


def handle_calendar_snapshot(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Convert a calendar snapshot into a readable schedule list.

    Args
    - last: The 'ToolMessage' with snapshot content.

    Returns
    - Dict with a single message containing a compact schedule; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "calendar_snapshot":
            return None
        snap = None
        if isinstance(last.content, dict):
            snap = last.content
        elif isinstance(last.content, str):
            try:
                snap = json.loads(last.content)
            except Exception:
                snap = None
        lines: list[str] = []
        if isinstance(snap, dict):
            pl = snap.get("pretty_lines") or []
            if isinstance(pl, list) and pl:
                lines = [str(x) for x in pl]
            else:
                evs = snap.get("events") or []
                for ev in evs:
                    kind = ev.get("type")
                    s = ev.get("start_local_pretty") or ev.get("start_local") or ""
                    e = ev.get("end_local_pretty") or ev.get("end_local") or ""
                    title = ev.get("title") or (
                        "Appointment" if kind == "appointment" else kind or "Event"
                    )
                    status = ev.get("status") or ""
                    if status == "canceled" and kind == "appointment":
                        title = f"Canceled {title}"
                    if s and e:
                        lines.append(f"{s} – {e}: {title}")
        else:
            c = last.content
            try:
                if isinstance(c, str):
                    m = re.search(r"pretty_lines=\[(.*)\]", c, re.S)
                    if m:
                        blob = m.group(1)
                        items = re.findall(r"'([^']+)'", blob)
                        if items:
                            lines = [str(x) for x in items]
                if not lines:
                    pairs = re.findall(
                        r"start_local_pretty='([^']+)'[\s\S]*?end_local_pretty='([^']+)'[\s\S]*?(?:title='([^']+)')?",
                        c,
                    )
                    for s_pretty, e_pretty, title in pairs:
                        t = title or "Event"
                        lines.append(f"{s_pretty} – {e_pretty}: {t}")
            except Exception:
                pass
        if lines:
            header = "Here is your schedule:"
            body = "\n".join(lines[:100])
            return {"messages": [AIMessage(content=f"{header}\n{body}")]}
        return None
    except Exception:
        logging.getLogger(__name__).exception("handle_calendar_snapshot failed")
        return None


def handle_booking_success(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Handle success from booking tools: book_appointment/create_appointment.

    Args
    - last: The 'ToolMessage' containing booking result data.
    - msgs: Conversation history to stitch pending markers and enrich details.

    Returns
    - Dict with confirmation, deterministic markers, and follow-up actions; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") not in ("book_appointment", "create_appointment"):
            return None
        payload = None
        if isinstance(last.content, dict):
            payload = last.content
        elif isinstance(last.content, str):
            s = last.content.strip()
            try:
                payload = json.loads(s)
            except Exception:
                payload = None
            if payload is None:
                m = re.search(r"appointment_id=['\"]?([0-9a-fA-F-]+)", s)
                if m:
                    payload = {"appointment_id": m.group(1)}
        if not (isinstance(payload, dict) and payload.get("appointment_id")):
            return None
        appt_id = payload.get("appointment_id")
        tool_name, tool_args = _last_tool_call_and_args(msgs)
        start_local = (tool_args or {}).get("start_local")
        dur = (tool_args or {}).get("duration_min")
        who = (tool_args or {}).get("client_name") or "the client"
        eml = (tool_args or {}).get("client_email")
        who_str = f"{who} <{eml}>" if eml else who
        when_str = None
        if isinstance(start_local, str) and len(start_local) >= 16:
            when_str = start_local.replace("T", " ").rsplit(":", 1)[0]
        dur_str = f" for {int(dur)} minutes" if dur else ""
        ack_text = "✅ Appointment booked. Pulling the full details now."
        if when_str:
            ack_text = f"✅ Booked {who_str} at {when_str}{dur_str}. I'll pull the full details now."
        elif who_str:
            ack_text = f"✅ Booked {who_str}. I'll pull the full details now."
        booking_data = {
            "appointment_id": appt_id,
            "client_name": who,
            "client_email": eml,
            "start_local": start_local,
            "duration_minutes": int(dur) if dur else None,
            "person_id": (tool_args or {}).get("person_id"),
            "status": "booked",
        }
        try:
            from agent.tool_registry import ALL_TOOLS as _TOOLS

            _TOOL_NAMES = {getattr(t, "name", None) or "" for t in _TOOLS}
            has_details = "get_appointment_details" in _TOOL_NAMES
        except Exception:
            has_details = False
        if has_details:
            marker = _make_pending_booked_message(booking_data)
            call = AIMessage(
                content="",
                tool_calls=[
                    _tc(
                        "get_appointment_details",
                        {"appointment_id": appt_id},
                        id="call_details_after_book",
                    )
                ],
            )
            return {"messages": [AIMessage(content=ack_text), marker, call]}
        if when_str:
            marker = _make_pending_booked_message(booking_data)
            instruction = _make_booking_llm_instruction(
                {**booking_data, "start_local": start_local}, "booked"
            )
            return {
                "messages": [
                    AIMessage(content=f"✅ Booked {who_str} at {when_str}{dur_str}."),
                    marker,
                    instruction,
                ]
            }
        marker = _make_pending_booked_message(booking_data)
        instruction = _make_booking_llm_instruction(booking_data, "booked")
        return {
            "messages": [
                AIMessage(content=f"✅ Booked {who_str}. Appointment ID: {appt_id}."),
                marker,
                instruction,
            ]
        }
    except Exception:
        logging.getLogger(__name__).exception("handle_booking_success failed")
        return None


def handle_get_appointment_details(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Enrich a booking confirmation using 'get_appointment_details' result.

    Args
    - last: The 'ToolMessage' containing an 'appointment' payload.
    - msgs: Conversation history with pending markers and hints.

    Returns
    - Dict with confirmation text, refreshed markers, and optional identity-attach
      tool call; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "get_appointment_details":
            return None
        details = None
        if isinstance(last.content, dict):
            details = last.content.get("appointment")
        elif isinstance(last.content, str):
            try:
                j = json.loads(last.content)
                details = j.get("appointment")
            except Exception:
                details = None
        if not (details and details.get("id")):
            return None
        appt_id = details["id"]
        start_local = details.get("start_local")
        end_local = details.get("end_local")
        who_name = details.get("client_name") or "client"
        who_email = details.get("client_email") or ""
        dur_min = None
        try:
            from datetime import datetime as _dt

            _a = _dt.fromisoformat(start_local)
            _b = _dt.fromisoformat(end_local)
            dur_min = int(round((_b - _a).total_seconds() / 60.0))
        except Exception:
            pass
        when_str = (
            start_local.replace("T", " ").rsplit(":", 1)[0]
            if isinstance(start_local, str)
            else "the requested time"
        )
        email_str = f" <{who_email}>" if who_email else ""
        had_pending_cancel = any(
            isinstance(m, AIMessage)
            and isinstance(m.content, str)
            and m.content.startswith("PENDING_CANCELED:")
            for m in msgs[:-1]
        )
        if had_pending_cancel:
            return {
                "messages": [
                    AIMessage(
                        content=f"✅ Canceled {who_name}{email_str} at {when_str}."
                    )
                ]
            }
        dur_str = f" for {dur_min} minutes" if dur_min else ""
        confirm_text = f"✅ Booked {who_name}{email_str} at {when_str}{dur_str}."
        # Rehydrate booking_data from pending marker if present
        booking_data = {}
        for mm in reversed(msgs[:-1]):
            if (
                isinstance(mm, AIMessage)
                and isinstance(mm.content, str)
                and mm.content.startswith("PENDING_BOOKED:")
            ):
                try:
                    booking_data = json.loads(mm.content[len("PENDING_BOOKED:") :])
                except Exception:
                    booking_data = {}
                break
        booking_data.update(
            {
                "appointment_id": appt_id,
                "client_name": who_name,
                "client_email": who_email,
                "start_local": start_local,
                "end_local": end_local,
                "duration_minutes": dur_min,
                "person_id": details.get("person_id"),
                "status": "booked",
            }
        )
        marker = _make_pending_booked_message(booking_data)
        instruction = _make_booking_llm_instruction(booking_data, "booked")
        # Optionally auto-link identity if we have hints and details missing
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
        people = (pending_client or {}).get("people") or []
        single_person_id = None
        if (
            not details.get("person_id")
            and isinstance(people, list)
            and len(people) == 1
        ):
            try:
                single_person_id = int(people[0]["person_id"])
            except Exception:
                single_person_id = None
        identity_payload: dict[str, object] = {}
        if single_person_id is not None:
            identity_payload["person_id"] = single_person_id
        name_hint = booking_data.get("client_name")
        if name_hint:
            identity_payload["client_name"] = name_hint
        email_hint = booking_data.get("client_email")
        if email_hint:
            identity_payload["client_email"] = email_hint
        needs_person = bool(identity_payload.get("person_id")) and not details.get(
            "person_id"
        )
        needs_name = bool(identity_payload.get("client_name")) and not details.get(
            "client_name"
        )
        needs_email = bool(identity_payload.get("client_email")) and not details.get(
            "client_email"
        )
        if identity_payload and (needs_person or needs_name or needs_email):
            fix_args = {"appointment_id": appt_id}
            fix_args.update(identity_payload)
            fix_call = AIMessage(
                content="",
                tool_calls=[
                    _tc(
                        "update_appointment_details",
                        fix_args,
                        id="call_attach_identity",
                    )
                ],
            )
            return {"messages": [AIMessage(content=confirm_text), marker, fix_call]}
        return {"messages": [AIMessage(content=confirm_text), marker, instruction]}
    except Exception:
        logging.getLogger(__name__).exception("handle_get_appointment_details failed")
        return None


def handle_update_details_error_or_success(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Summarize the outcome of 'update_appointment_details'.

    Args
    - last: The 'ToolMessage' containing success/error details.
    - msgs: Conversation history with pending booking markers.

    Returns
    - Dict acknowledging booking and either explaining link failure or
      confirming the identity link; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "update_appointment_details":
            return None
        if (
            isinstance(last.content, str)
            and "update_appointment_details failed" in last.content
        ):
            booking_data = {}
            for mm in reversed(msgs[:-1]):
                if (
                    isinstance(mm, AIMessage)
                    and isinstance(mm.content, str)
                    and mm.content.startswith("PENDING_BOOKED:")
                ):
                    try:
                        booking_data = json.loads(mm.content[len("PENDING_BOOKED:") :])
                    except Exception:
                        booking_data = {}
                    break
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
        payload = None
        if isinstance(last.content, dict):
            payload = last.content
        elif isinstance(last.content, str):
            try:
                payload = json.loads(last.content)
            except Exception:
                payload = None
        if isinstance(payload, dict) and payload.get("ok"):
            who = payload.get("client_name") or "the client"
            booking_data = {}
            for mm in reversed(msgs[:-1]):
                if (
                    isinstance(mm, AIMessage)
                    and isinstance(mm.content, str)
                    and mm.content.startswith("PENDING_BOOKED:")
                ):
                    try:
                        booking_data = json.loads(mm.content[len("PENDING_BOOKED:") :])
                    except Exception:
                        booking_data = {}
                    break
            if booking_data:
                booking_data["client_name"] = payload.get(
                    "client_name"
                ) or booking_data.get("client_name")
                booking_data["client_email"] = payload.get(
                    "client_email"
                ) or booking_data.get("client_email")
                booking_data["person_id"] = payload.get(
                    "person_id"
                ) or booking_data.get("person_id")
                marker = _make_pending_booked_message(booking_data)
                instruction = _make_booking_llm_instruction(booking_data, "booked")
                return {
                    "messages": [
                        AIMessage(content=f"🔗 Linked this appointment to {who}."),
                        marker,
                        instruction,
                    ]
                }
            instruction = _make_booking_llm_instruction(
                {"client_name": who, "status": "booked"}, "booked"
            )
            return {
                "messages": [
                    AIMessage(content=f"🔗 Linked this appointment to {who}."),
                    instruction,
                ]
            }
    except Exception:
        logging.getLogger(__name__).exception(
            "handle_update_details_error_or_success failed"
        )
    return None


def handle_cancel_appointment_success(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Acknowledge successful cancellation from 'cancel_appointment'.

    Args
    - last: The 'ToolMessage' containing cancellation status.

    Returns
    - Dict with a simple checkmark confirmation; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "cancel_appointment":
            return None
        payload = None
        if isinstance(last.content, dict):
            payload = last.content
        elif isinstance(last.content, str):
            try:
                payload = json.loads(last.content)
            except Exception:
                payload = None
        if (
            isinstance(payload, dict)
            and (payload.get("status") or "").lower() == "canceled"
        ):
            return {"messages": [AIMessage(content="✅ Appointment canceled.")]}
    except Exception:
        logging.getLogger(__name__).exception(
            "handle_cancel_appointment_success failed"
        )
    return None


def handle_is_public_holiday(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Confirm or proceed with actions on a public holiday.

    Args
    - last: The 'ToolMessage' with holiday evaluation result.
    - msgs: Conversation history providing the queued intent marker.

    Returns
    - Dict prompting confirmation on a holiday or proceeding with the queued
      action; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "is_public_holiday":
            return None
        parsed = None
        if isinstance(last.content, dict):
            parsed = last.content
        elif isinstance(last.content, str):
            try:
                parsed = json.loads(last.content)
            except Exception:
                parsed = None
        is_holiday = False
        holiday_name = None
        try:
            if isinstance(parsed, dict):
                is_holiday = bool(parsed.get("is_holiday"))
                holiday_name = parsed.get("name")
        except Exception:
            is_holiday = False
            holiday_name = None

        pending = None
        for m in reversed(msgs[:-1]):
            if (
                isinstance(m, (AIMessage, SystemMessage))
                and isinstance(m.content, str)
                and m.content.startswith("PENDING_INTENT_AFTER_HOLIDAY:")
            ):
                try:
                    pending = json.loads(
                        m.content[len("PENDING_INTENT_AFTER_HOLIDAY:") :]
                    )
                except Exception:
                    pending = None
                break
        tool_name = (pending or {}).get("tool")
        args = (pending or {}).get("args") or {}
        date_str = (pending or {}).get("date")
        if not tool_name or not isinstance(args, dict):
            return {"messages": []}
        if is_holiday:
            nm = str(holiday_name) if holiday_name else "a public holiday"
            try:
                from datetime import datetime as _d

                lbl = (
                    _d.fromisoformat(date_str).strftime("%A, %b %d, %Y")
                    if date_str
                    else date_str
                )
            except Exception:
                lbl = date_str
            marker = SystemMessage(
                content="CONFIRM_REQUIRED:"
                + json.dumps(
                    {
                        "tool": tool_name,
                        "args": args,
                        "date": date_str,
                        "holiday": holiday_name,
                    }
                )
            )
            ask = AIMessage(
                content=f"Heads up — {lbl} is {nm}. Do you want me to proceed?"
            )
            return {"messages": [marker, ask]}
        if tool_name == "book_appointment":
            kept = {
                "start_local": args.get("start_local"),
                "duration_min": args.get("duration_min"),
                "client_name": args.get("client_name"),
                "client_email": args.get("client_email"),
                "client_query": args.get("client_query"),
                "notes": args.get("notes"),
            }
            who = kept.get("client_name") or kept.get("client_query") or "the client"
            eml = kept.get("client_email")
            who_str = f"{who} <{eml}>" if eml else who
            human_when = (kept.get("start_local") or "").replace("T", " ")
            dur = kept.get("duration_min")
            confirm_marker = AIMessage(
                content="CONFIRM_REQUIRED:"
                + json.dumps({"tool": tool_name, "args": args})
            )
            ask = AIMessage(
                content=f"Please confirm: book {who_str} at {human_when} for {dur} minutes? (yes / no)"
            )
            return {"messages": [confirm_marker, ask]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        _tc(tool_name, args, id=f"call_after_holiday_{tool_name}")
                    ],
                )
            ]
        }
    except Exception:
        logging.getLogger(__name__).exception("handle_is_public_holiday failed")
        return None


def _parse_holiday_list(last: ToolMessage, parsed: Any) -> list:
    """Extract a 'holidays' list from a tool payload or string content.

    Supports dict payloads and loose string dumps. Returns a list (possibly
    empty).
    """
    holidays = []
    if isinstance(parsed, list):
        holidays = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("holidays"), list):
        holidays = parsed["holidays"]
    elif hasattr(parsed, "holidays"):
        try:
            holidays = list(getattr(parsed, "holidays") or [])
        except Exception:
            holidays = []
    elif isinstance(last.content, str):
        try:
            j = json.loads(last.content)
            if isinstance(j, list):
                holidays = j
            elif isinstance(j, dict) and isinstance(j.get("holidays"), list):
                holidays = j["holidays"]
        except Exception:
            pass
    elif isinstance(last.content, list):
        holidays = last.content
    elif hasattr(last.content, "holidays"):
        try:
            holidays = list(getattr(last.content, "holidays") or [])
        except Exception:
            holidays = []
    if not holidays and isinstance(last.content, str):
        import re as _re
        import ast as _ast

        # Handle stringified dicts (repr-style) that aren't valid JSON
        try:
            blob = _ast.literal_eval(last.content)
            if isinstance(blob, dict) and isinstance(blob.get("holidays"), list):
                holidays = blob["holidays"]
        except Exception:
            pass
        if not holidays:
            m = _re.search(r"holidays\s*[:=]\s*(\[[\s\S]*\])", last.content)
            if m:
                try:
                    arr = _ast.literal_eval(m.group(1))
                    if isinstance(arr, list):
                        holidays = arr
                except Exception:
                    pass
    return holidays


def handle_get_public_holidays(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Render public holidays within a requested window.

    Args
    - last: The 'ToolMessage' with holidays content.
    - msgs: Conversation history providing the scope marker.

    Returns
    - Dict with a concise list or a friendly message when none found; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "get_public_holidays":
            return None
        holidays = _parse_holiday_list(last, last.content)
        scope_info = None
        for m in reversed(msgs[:-1]):
            if (
                isinstance(m, AIMessage)
                and isinstance(m.content, str)
                and m.content.startswith("PENDING_HOLIDAY_SCOPE:")
            ):
                try:
                    scope_info = json.loads(m.content[len("PENDING_HOLIDAY_SCOPE:") :])
                except Exception:
                    scope_info = None
                break
        today = _today_owner_date()
        which = (scope_info or {}).get("which") or "this"
        scope = (scope_info or {}).get("scope") or "week"
        mode = (scope_info or {}).get("mode") or None
        if mode == "next":
            upcoming: list[tuple[_date, dict]] = []
            for h in holidays:
                try:
                    hd = _dt.fromisoformat(str(h.get("date"))).date()
                except Exception:
                    try:
                        y, m, d = [int(x) for x in str(h.get("date")).split("-")]
                        hd = _dt(y, m, d).date()
                    except Exception:
                        continue
                if hd >= today:
                    upcoming.append((hd, h))
            next_h = None
            if upcoming:
                next_h = sorted(upcoming, key=lambda x: x[0])[0][1]
            if not next_h:
                # No upcoming holidays this year; try the next year automatically once.
                attempted_next_year = any(
                    isinstance(mm, AIMessage)
                    and isinstance(mm.content, str)
                    and mm.content == "PENDING_HOLIDAY_NEXT_YEAR"
                    for mm in msgs[:-1]
                )
                if not attempted_next_year:
                    ny = today.year + 1
                    return {
                        "messages": [
                            AIMessage(content="PENDING_HOLIDAY_NEXT_YEAR"),
                            AIMessage(
                                content="",
                                tool_calls=[
                                    _tc(
                                        "get_public_holidays",
                                        {"year": ny},
                                        id="call_holidays_next",
                                    )
                                ],
                            ),
                        ]
                    }
                return {
                    "messages": [
                        AIMessage(content="No upcoming public holidays found.")
                    ]
                }
            nm = next_h.get("localName") or next_h.get("name") or "Public Holiday"
            dt_s = str(next_h.get("date"))
            return {"messages": [AIMessage(content=f"Next holiday: {dt_s}: {nm}")]}
        start = today
        end = today
        if scope == "today":
            if which == "tomorrow":
                start = today + _td(days=1)
                end = start
            else:
                start = today
                end = today
        else:
            wd = today.weekday()
            this_monday = today - _td(days=wd)
            if which == "next":
                this_monday = this_monday + _td(days=7)
            start = this_monday
            end = this_monday + _td(days=7)
        in_window = []
        for h in holidays:
            try:
                y, m, d = [int(x) for x in str(h.get("date")).split("-")]
                hd = _dt(y, m, d).date()
                if start <= hd < end:
                    in_window.append(h)
            except Exception:
                continue
        if not in_window:
            return {
                "messages": [
                    AIMessage(content="No public holidays in the requested window.")
                ]
            }
        lines = [
            f"{h['date']}: {h.get('localName') or h.get('name')}"
            for h in sorted(in_window, key=lambda x: x.get("date", ""))
        ]
        header = "Holidays:"
        body = "\n".join(lines[:50])
        return {"messages": [AIMessage(content=f"{header}\n{body}")]}
    except Exception:
        logging.getLogger(__name__).exception("handle_get_public_holidays failed")
        return None


def handle_add_special_opening(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Confirm creation of a one-off opening with local time formatting.

    Args
    - last: The 'ToolMessage' containing opening creation result.

    Returns
    - Dict with a short, localized confirmation line; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "add_special_opening":
            return None
        payload: dict | None = last.content if isinstance(last.content, dict) else None
        if payload is None and isinstance(last.content, str):
            try:
                payload = json.loads(last.content)
            except Exception:
                payload = None
        su = (payload or {}).get("start_utc")
        eu = (payload or {}).get("end_utc")
        slot_min = (payload or {}).get("slot_minutes")
        from datetime import datetime as _d

        def _parse_dt(val):
            if val is None:
                return None
            try:
                if isinstance(val, str):
                    s = val.replace("Z", "+00:00")
                    return _d.fromisoformat(s)
                return val if hasattr(val, "isoformat") else None
            except Exception:
                return None

        su_dt = _parse_dt(su)
        eu_dt = _parse_dt(eu)
        try:
            from zoneinfo import ZoneInfo as _ZI

            tzname = tz_var.get() or "America/Toronto"
            if su_dt and su_dt.tzinfo:
                su_loc = su_dt.astimezone(_ZI(tzname))
            elif su_dt:
                su_loc = su_dt.replace(tzinfo=_ZI("UTC")).astimezone(_ZI(tzname))
            else:
                su_loc = None
            if eu_dt and eu_dt.tzinfo:
                eu_loc = eu_dt.astimezone(_ZI(tzname))
            elif eu_dt:
                eu_loc = eu_dt.replace(tzinfo=_ZI("UTC")).astimezone(_ZI(tzname))
            else:
                eu_loc = None
        except Exception:
            su_loc = su_dt
            eu_loc = eu_dt
        if su_loc and eu_loc:
            s_pretty = (
                su_loc.strftime("%a %b %d, %I:%M %p").lstrip("0").replace(" 0", " ")
            )
            e_pretty = eu_loc.strftime("%I:%M %p").lstrip("0")
            msg = f"Added opening: {s_pretty} – {e_pretty} ({int(slot_min) if slot_min else 'unknown'}-minute slots)."
            return {"messages": [AIMessage(content=msg)]}
        return {"messages": [AIMessage(content="Opening added.")]}
    except Exception:
        logging.getLogger(__name__).exception("handle_add_special_opening failed")
        return None


def handle_create_recurring_openings(last: ToolMessage) -> dict[str, list[Any]] | None:
    """Confirm creation of recurring openings and show the first occurrence.

    Args
    - last: The 'ToolMessage' containing creation results.

    Returns
    - Dict with a concise confirmation message referencing the first slot; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "create_recurring_openings":
            return None
        payload: dict | None = last.content if isinstance(last.content, dict) else None
        if payload is None and isinstance(last.content, str):
            try:
                payload = json.loads(last.content)
            except Exception:
                payload = None
        created: list[dict] = []
        if isinstance(payload, dict):
            created = payload.get("created") or []
        if created:
            first = created[0]
            s = first.get("start_local") or first.get("start_utc")
            e = first.get("end_local") or first.get("end_utc")
            try:
                from zoneinfo import ZoneInfo as _ZI

                tzname = tz_var.get() or "America/Toronto"
                from datetime import datetime as _d

                def _fix(v):
                    if not v:
                        return None
                    if isinstance(v, str) and "T" in v:
                        vv = v.replace("Z", "+00:00")
                        dt = _d.fromisoformat(vv)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=_ZI("UTC"))
                        return dt.astimezone(_ZI(tzname)).strftime("%Y-%m-%dT%H:%M")
                    return v

                if s and "Z" in str(s):
                    s = _fix(s)
                if e and "Z" in str(e):
                    e = _fix(e)
            except Exception:
                pass
            return {
                "messages": [
                    AIMessage(content=f"Created weekly openings. First: {s} – {e}.")
                ]
            }
        return {"messages": [AIMessage(content="Created weekly openings.")]}
    except Exception:
        logging.getLogger(__name__).exception("handle_create_recurring_openings failed")
        return None


def handle_update_client(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Summarize client updates in a friendly, compact sentence.

    Args
    - last: The 'ToolMessage' with canonical updated values.
    - msgs: Conversation history to include requested changes.

    Returns
    - Dict with a succinct confirmation sentence; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "update_client":
            return None
        upd = last.content if isinstance(last.content, dict) else None
        if upd is None and isinstance(last.content, str):
            try:
                upd = json.loads(last.content)
            except Exception:
                upd = None
        if not isinstance(upd, dict):
            return None
        pieces = []
        seen: set[str] = set()
        tool_name, tool_args = _last_tool_call_and_args(msgs)
        if tool_name == "update_client" and isinstance(tool_args, dict):
            if tool_args.get("account_name"):
                pieces.append(f"account name set to {tool_args['account_name']}")
                seen.add("account_name")
            if tool_args.get("phone"):
                pieces.append(f"phone set to {tool_args['phone']}")
                seen.add("phone")
            if tool_args.get("emergency_contact"):
                pieces.append(
                    f"emergency contact set to {tool_args['emergency_contact']}"
                )
                seen.add("emergency_contact")
            if tool_args.get("primary_email"):
                pieces.append(f"primary email set to {tool_args['primary_email']}")
                seen.add("primary_email")
            if tool_args.get("secondary_email"):
                pieces.append(f"secondary email set to {tool_args['secondary_email']}")
                seen.add("secondary_email")
            if tool_args.get("name"):
                pieces.append(f"user name set to {tool_args['name']}")
                seen.add("name")
            if tool_args.get("email"):
                pieces.append(f"user email set to {tool_args['email']}")
                seen.add("email")
            if tool_args.get("add_person_name"):
                nm = tool_args.get("add_person_name")
                em = tool_args.get("add_person_email")
                pieces.append(f"added {nm}{(' <' + em + '>' if em else '')}")
                seen.add("add_person")
        ph = upd.get("primary_phone")
        if ph and "phone" not in seen:
            pieces.append(f"phone set to {ph}")
            seen.add("phone")
        pe = upd.get("primary_email")
        if pe and "primary_email" not in seen:
            pieces.append(f"primary email set to {pe}")
            seen.add("primary_email")
        se = upd.get("secondary_email")
        if se and "secondary_email" not in seen:
            pieces.append(f"secondary email set to {se}")
            seen.add("secondary_email")
        nm_now = upd.get("name")
        if nm_now and "account_name" not in seen:
            pieces.append(f"account name set to {nm_now}")
            seen.add("account_name")
        ec_now = upd.get("emergency_contact")
        if ec_now and "emergency_contact" not in seen:
            pieces.append(f"emergency contact set to {ec_now}")
            seen.add("emergency_contact")
        ap = upd.get("added_person") or {}
        ap_name = ap.get("full_name")
        ap_email = ap.get("email")
        if ap_name and "add_person" not in seen:
            pieces.append(
                f"added {ap_name}{(' <' + ap_email + '>' if ap_email else '')}"
            )
            seen.add("add_person")
        msg = "Updated client" + (": " + ", ".join(pieces) if pieces else ".")
        return {"messages": [AIMessage(content=f"✅ {msg}")]}
    except Exception:
        logging.getLogger(__name__).exception("handle_update_client failed")
        return None


def handle_update_person(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Summarize person updates with name/email when applicable.

    Args
    - last: The 'ToolMessage' containing the update result.
    - msgs: Conversation history to incorporate requested changes.

    Returns
    - Dict with a friendly confirmation line; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "update_person":
            return None
        upd = last.content if isinstance(last.content, dict) else None
        if upd is None and isinstance(last.content, str):
            try:
                upd = json.loads(last.content)
            except Exception:
                upd = None
        if not isinstance(upd, dict):
            return None
        pieces = []
        tool_name, tool_args = _last_tool_call_and_args(msgs)
        if tool_name == "update_person" and isinstance(tool_args, dict):
            if tool_args.get("full_name"):
                pieces.append(f"name set to {tool_args['full_name']}")
            if tool_args.get("email"):
                pieces.append(f"email set to {tool_args['email']}")
        nm = upd.get("full_name")
        em = upd.get("email")
        head = f"Updated person{(' ' + nm) if nm else ''}"
        tail = (
            (": " + ", ".join(pieces))
            if pieces
            else (
                f": {nm} <{em}>"
                if (nm and em)
                else (f": {nm}" if nm else (f": <{em}>" if em else "."))
            )
        )
        return {"messages": [AIMessage(content=f"✅ {head}{tail}")]}
    except Exception:
        logging.getLogger(__name__).exception("handle_update_person failed")
        return None


def handle_update_appointment(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Summarize appointment metadata updates (payment, attendance, price, etc.).

    Args
    - last: The 'ToolMessage' containing the update result.
    - msgs: Conversation history to read requested fields from the prior call.

    Returns
    - Dict with a compact summary line; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "update_appointment":
            return None
        tool_name, tool_args = _last_tool_call_and_args(msgs)
        pieces = []

        def _fmt_cents(v):
            try:
                n = int(v)
            except Exception:
                return None
            return f"${n / 100:.2f}"

        if tool_name == "update_appointment" and isinstance(tool_args, dict):
            if tool_args.get("payment_status"):
                pieces.append(f"payment set to {tool_args['payment_status']}")
            if tool_args.get("amount_paid_cents") is not None:
                usd = _fmt_cents(tool_args.get("amount_paid_cents"))
                if usd:
                    pieces.append(f"cash paid set to {usd}")
            if tool_args.get("price_override_cents") is not None:
                usd = _fmt_cents(tool_args.get("price_override_cents"))
                if usd:
                    pieces.append(f"price set to {usd}")
            if tool_args.get("bundle_id") is not None:
                bid = tool_args.get("bundle_id")
                if bid and int(bid) > 0:
                    pieces.append(f"wallet attached (bundle {bid})")
                else:
                    pieces.append("wallet detached")
            if tool_args.get("attendance"):
                pieces.append(f"attendance set to {tool_args['attendance']}")
            if tool_args.get("late_minutes") is not None:
                pieces.append(f"late minutes set to {tool_args['late_minutes']}")
            if tool_args.get("private_note"):
                pieces.append("note updated")
        msg = "Updated appointment" + (": " + ", ".join(pieces) if pieces else ".")
        return {"messages": [AIMessage(content=f"✅ {msg}")]}
    except Exception:
        logging.getLogger(__name__).exception("handle_update_appointment failed")
        return None


def handle_update_appointment_details_summary(
    last: ToolMessage,
) -> dict[str, list[Any]] | None:
    """Summarize updates from 'update_appointment_details' (identity edits).

    Args
    - last: The 'ToolMessage' with the details update result.

    Returns
    - Dict with a brief confirmation of identity changes; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "update_appointment_details":
            return None
        upd = last.content if isinstance(last.content, dict) else None
        if upd is None and isinstance(last.content, str):
            try:
                upd = json.loads(last.content)
            except Exception:
                upd = None
        pieces = []
        if isinstance(upd, dict):
            cn = upd.get("client_name")
            ce = upd.get("client_email")
            if cn or ce:
                pieces.append(
                    "client set to " + ((cn or "client") + (f" <{ce}>" if ce else ""))
                )
            pid = upd.get("person_id")
            if pid and not cn:
                pieces.append(f"person set (id {pid})")
        msg = "Updated appointment details" + (
            ": " + ", ".join(pieces) if pieces else "."
        )
        return {"messages": [AIMessage(content=f"✅ {msg}")]}
    except Exception:
        logging.getLogger(__name__).exception(
            "handle_update_appointment_details_summary failed"
        )
        return None


def handle_reschedule_appointment(
    last: ToolMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Summarize rescheduling outcome from 'reschedule_appointment'.

    Args
    - last: The 'ToolMessage' describing the new start time.
    - msgs: Conversation history for context and fallbacks.

    Returns
    - Dict with a simple checkmark confirmation; otherwise 'None'.

    Raises
    - None
    """
    try:
        if getattr(last, "name", "") != "reschedule_appointment":
            return None
        upd = last.content if isinstance(last.content, dict) else None
        if upd is None and isinstance(last.content, str):
            try:
                upd = json.loads(last.content)
            except Exception:
                upd = None
        when_str = None
        if isinstance(upd, dict):
            tool_name, tool_args = _last_tool_call_and_args(msgs)
            if tool_name == "reschedule_appointment" and isinstance(tool_args, dict):
                sl = tool_args.get("start_local")
                if isinstance(sl, str) and sl:
                    when_str = sl.replace("T", " ")[:16]
        if not when_str and isinstance(upd, dict) and upd.get("start_utc"):
            when_str = str(upd.get("start_utc"))
        text = f"Rescheduled to {when_str}." if when_str else "Rescheduled."
        return {"messages": [AIMessage(content=f"✅ {text}")]}
    except Exception:
        logging.getLogger(__name__).exception("handle_reschedule_appointment failed")
        return None


def handle_pending_email_human(
    last: HumanMessage, msgs: list[Any]
) -> dict[str, list[Any]] | None:
    """Interpret human responses to a pending email draft.

    Args
    - last: The user's 'HumanMessage' reply.
    - msgs: Conversation history to locate the pending draft marker.

    Returns
    - Dict issuing a 'send_approved_email' tool call, a cancel/confirm reply,
      or minor subject/body edits; otherwise 'None' when no pending context.

    Raises
    - None
    """
    try:
        pending = None
        for m in reversed(msgs[:-1]):
            if (
                isinstance(m, AIMessage)
                and isinstance(m.content, str)
                and m.content.startswith("PENDING_EMAIL_SEND:")
            ):
                try:
                    pending = json.loads(m.content[len("PENDING_EMAIL_SEND:") :])
                except Exception:
                    pending = None
                break
        if not pending:
            return None
        reply = (last.content or "").strip()
        m_send = re.match(r"^SEND_EMAIL:\s*(\{.*\})\s*$", reply, re.S)
        if m_send:
            try:
                overrides = json.loads(m_send.group(1))
            except Exception:
                return {
                    "messages": [
                        AIMessage(
                            content="That Send action had invalid JSON. Try again."
                        )
                    ]
                }
            args = {
                "draft_id": overrides.get("draft_id") or pending["draft_id"],
                "approve": bool(overrides.get("approve", True)),
                "to": overrides.get("to", pending.get("to")),
                "to_name": overrides.get("to_name", pending.get("to_name")),
                "subject": overrides.get("subject", pending.get("subject")),
                "text": overrides.get("text", pending.get("text")),
            }
            tool_call = AIMessage(
                content="",
                tool_calls=[_tc("send_approved_email", args, id="call_send_email_ui")],
            )
            return {"messages": [tool_call]}
        if re.search(r"\b(confirm|proceed|yes|go ahead|do it)\b", reply, re.I):
            tool_call = AIMessage(
                content="",
                tool_calls=[
                    _tc(
                        "send_approved_email",
                        {
                            "draft_id": pending["draft_id"],
                            "approve": True,
                            "to": pending.get("to"),
                            "to_name": pending.get("to_name"),
                            "subject": pending.get("subject"),
                            "text": pending.get("text"),
                        },
                        id="call_send_email",
                    )
                ],
            )
            return {"messages": [tool_call]}
        if re.search(r"\b(cancel|stop|nevermind|never mind)\b", reply, re.I):
            return {
                "messages": [
                    AIMessage(
                        content="Okay, I won’t send it. You can tell me new edits or start a new message anytime."
                    )
                ]
            }
        new_subject = None
        new_text = None
        m_subj = re.search(
            r"(?:change|set)\s+subject\s+(?:to|as)\s+\"?(.+?)\"?$", reply, re.I
        )
        if m_subj:
            new_subject = m_subj.group(1).strip()
        m_body = re.search(
            r"(?:replace|set)\s+(?:body|message|text)\s+(?:to|with)\s+\"?(.+?)\"?$",
            reply,
            re.I | re.S,
        )
        if m_body:
            new_text = m_body.group(1).strip()
        if new_subject or new_text:
            tool_call = AIMessage(
                content="",
                tool_calls=[
                    _tc(
                        "send_approved_email",
                        {
                            "draft_id": pending["draft_id"],
                            "approve": True,
                            "to": pending.get("to"),
                            "to_name": pending.get("to_name"),
                            "subject": new_subject or pending.get("subject"),
                            "text": new_text or pending.get("text"),
                        },
                        id="call_send_email_edits",
                    )
                ],
            )
            return {"messages": [tool_call]}
        return {"messages": []}
    except Exception:
        logging.getLogger(__name__).exception("handle_pending_email_human failed")
        return None
