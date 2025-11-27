from __future__ import annotations
import re

# Examples: "at 9am", "at 9:30 pm"
TIME_AT_RE = re.compile(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", re.I)

# Examples: "from 9am to 9:30am"
TIME_FROM_TO_RE = re.compile(
    r"\bfrom\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\s+(?:to|until|till|’til|til)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
    re.I,
)

# Examples: "9am-9:30am", "09:00–09:30"
TIME_RANGE_RE = re.compile(
    r"\b(?:book\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*[-–]\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
    re.I,
)

# Examples: "end at 3pm", "until 15:00"
END_AT_RE = re.compile(
    r"\b(?:end|until|till|’til|til)\s+(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
    re.I,
)

# Weekday words
WEEKDAY_WORD_RE = re.compile(
    r"\b(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)s?\b",
    re.I,
)

DOW_WORDS = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def _parse_weekday_from_text(txt: str) -> int | None:
    """Parse a weekday from free text.

    Args
    - txt: Input string possibly containing a weekday word.

    Returns
    - Integer 0..6 (Mon..Sun) when a weekday token is found; otherwise None.

    Raises
    - None
    """
    for k, v in DOW_WORDS.items():
        if re.search(rf"\b{k}\b", txt, re.I):
            return v
    return None


SLOT_MIN_RE = re.compile(
    r"(?:every|each)\s*(\d{1,3})\s*(?:mins?|minutes?)\b|"
    r"\b(\d{1,3})\s*-?\s*minute(?:s)?\s*(?:slot|slots)?\b|"
    r"\bslot(?:s)?\s*of\s*(\d{1,3})\s*(?:mins?|minutes?)\b",
    re.I,
)


def _extract_slot_minutes(text: str) -> int | None:
    """Extract a plausible slot length in minutes from free text.

    Args
    - text: Input containing phrases like "every 30 minutes", "30-minute slots",
      or "slots of 45 minutes".

    Returns
    - Integer minutes within 5..240 when detected; otherwise None.

    Raises
    - None
    """
    if not text:
        return None
    m = SLOT_MIN_RE.search(text)
    if not m:
        return None
    for i in range(1, 4):
        g = m.group(i)
        if g:
            try:
                v = int(g)
                return v if 5 <= v <= 240 else None
            except Exception:
                return None
    return None


def _norm_hhmm_12to24(txt: str) -> str | None:
    """Normalize time text to 'HH:MM' 24-hour format.

    Args
    - txt: Time text such as "9am", "9:30pm", or a valid 24h form like "13:45".

    Returns
    - Normalized string like "09:00" or "09:30"; returns the validated input for
      already-24h times; otherwise None on invalid input.

    Raises
    - None
    """
    if not txt:
        return None
    s = txt.strip().lower().replace(" ", "")
    # Already in HH:MM 24h format
    if re.fullmatch(r"\d{2}:\d{2}", s):
        return s
    # HH:MMam/pm or HHam/pm
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)", s)
    if not m:
        # accept H:MM or H without am/pm as-is only if 00:00..23:59
        if re.fullmatch(r"\d{1,2}:\d{2}", s):
            h, mm = s.split(":", 1)
            try:
                hi = int(h)
                mi = int(mm)
                if 0 <= hi <= 23 and 0 <= mi <= 59:
                    return f"{hi:02d}:{mi:02d}"
            except Exception:
                return None
        return None
    h_i = int(m.group(1))
    m_i = int(m.group(2) or 0)
    ap = m.group(3)
    if h_i == 12:
        h_i = 0
    if ap == "pm":
        h_i += 12
    if not (0 <= h_i <= 23 and 0 <= m_i <= 59):
        return None
    return f"{h_i:02d}:{m_i:02d}"
