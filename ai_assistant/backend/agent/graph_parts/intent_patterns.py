from __future__ import annotations
import re
from typing import Sequence

from agent.graph_parts.time_parse import (
    TIME_FROM_TO_RE,
    TIME_RANGE_RE,
    END_AT_RE,
    TIME_AT_RE,
    _norm_hhmm_12to24,
)
from langchain_core.messages import BaseMessage, HumanMessage

# Explicit duration like "for 30 minutes"
DUR_RE = re.compile(r"\bfor\s+(\d{1,3})\s*(?:minutes?|mins?|min)\b", re.I)

# Numeric hints for durations (used to decide whether to ask for minutes)
DURATION_NUM_RE = re.compile(r"\b(1[5-9]|[2-9]\d|1\d\d|2[0-3]\d|240|30|45|60|120)\b")


def _extract_duration_min(text: str) -> int | None:
    """Parse a duration in minutes from free text.

    Args
    - text: Input message content.

    Returns
    - Integer minutes in the range 5..300 when detected; otherwise None.

    Raises
    - None

    Behavior
    - Supports "for 30 minutes" and phrases like "half an hour"/"a quarter hour".
    """
    if not text:
        return None
    m = DUR_RE.search(text)
    if m:
        try:
            v = int(m.group(1))
            return v if 5 <= v <= 300 else None
        except ValueError:
            return None

    low = text.lower()
    if re.search(r"\bhalf\s+(?:an?\s+)?hour\b", low):
        return 30
    if re.search(r"\ban?\s+hour\b", low):
        return 60
    if re.search(r"\ban?\s+quarter\s+hour\b", low):
        return 15

    # Derive duration from an explicit time range if present (e.g., 9am-10am).
    m_range = TIME_FROM_TO_RE.search(text) or TIME_RANGE_RE.search(text)
    if m_range:
        try:
            start_txt, end_txt = m_range.group(1), m_range.group(2)
            s24, e24 = _norm_hhmm_12to24(start_txt), _norm_hhmm_12to24(end_txt)
            if s24 and e24:
                sh, sm = [int(x) for x in s24.split(":")]
                eh, em = [int(x) for x in e24.split(":")]
                start_min = sh * 60 + sm
                end_min = eh * 60 + em
                if end_min > start_min:
                    diff = end_min - start_min
                    if 5 <= diff <= 300:
                        return diff
        except Exception:
            pass

    # Phrases like "start at 9am, end at 10am"
    m_end = END_AT_RE.search(text)
    m_start = TIME_AT_RE.search(text)
    if m_start and m_end:
        try:
            s24 = _norm_hhmm_12to24(m_start.group(1))
            e24 = _norm_hhmm_12to24(m_end.group(1))
            if s24 and e24:
                sh, sm = [int(x) for x in s24.split(":")]
                eh, em = [int(x) for x in e24.split(":")]
                start_min = sh * 60 + sm
                end_min = eh * 60 + em
                if end_min > start_min:
                    diff = end_min - start_min
                    if 5 <= diff <= 300:
                        return diff
        except Exception:
            pass
    return None


def _is_booking_intent(text: str) -> bool:
    """Determine if text is an actionable booking request.

    Args
    - text: Input message content.

    Returns
    - True when actionable booking phrases are present and openings/availability
      phrasing is not; otherwise False.

    Raises
    - None
    """
    if not text:
        return False
    low = text.lower()
    # Treat messages as booking-intent when they include common booking words
    # and are not clearly about openings/availability only.
    if any(
        k in low
        for k in [
            "opening",
            "availability",
            "available",
            "open slots",
            "open slot",
            "open a slot",
            "open up a slot",
            "create an opening",
            "create a opening",
            "make an opening",
            "make a opening",
            "add an opening",
            "add a opening",
        ]
    ):
        return False
    booking_words = [
        "book",
        "reschedule",
        "slot",
        "slots",
        "find a time",
        "schedule with",
        "schedule for",
        "schedule a",
        "schedule an",
        "make an appointment",
        "make a appointment",
        "set up an appointment",
        "set up a appointment",
        "create an appointment",
        "create a appointment",
    ]
    return any(k in low for k in booking_words)


def _needs_duration_question(messages: Sequence[BaseMessage]) -> bool:
    """Decide whether to ask the user for a duration clarification.

    Args
    - messages: Conversation history as a sequence of messages.

    Returns
    - True if the last user message intends to book/find a slot and does not
      specify a duration; otherwise False.

    Raises
    - None
    """
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return False
    text = (last.content or "").lower()
    # Email/message intents should not trigger a duration clarification.
    if any(k in text for k in ["email", "message", "notify", "remind", "reach out"]):
        return False

    # Pure availability/overview questions (not booking) should also skip the
    # duration follow-up.
    if any(
        k in text
        for k in [
            "availability",
            "available",
            "free time",
            "free slots",
            "when am i free",
            "when can i",
            "schedule",
            "agenda",
            "lessons today",
            "my lessons",
        ]
    ) and not _is_booking_intent(text):
        return False

    if _is_booking_intent(text):
        has_minutes = (DURATION_NUM_RE.search(text) is not None) or (
            _extract_duration_min(text) is not None
        )
        return not has_minutes

    return False
