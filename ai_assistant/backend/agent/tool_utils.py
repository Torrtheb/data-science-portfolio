from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException
from agent.tool_ctx import owner_id_var, tz_var

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _owner_now_date(tz: str) -> date:
    """Return today's date in the specified timezone.

    Args:
        tz: IANA timezone name.

    Returns:
        A 'date' representing the current day in that timezone.
    """
    return datetime.now(ZoneInfo(tz)).date()


def _parse_owner_day(text: str, owner_tz: str) -> date:
    """Parse an owner‑local day string into a date.

    Accepts ISO 'YYYY-MM-DD', month names, and keywords like
    'today'/'tomorrow'/'next <weekday>'. Uses the provided timezone to
    anchor relative dates.

    Args:
        text: Free‑form day description.
        owner_tz: IANA timezone name for interpretation.

    Returns:
        A 'date' corresponding to the described day.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    s = text.strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return date.fromisoformat(s)
    for fmt in ("%B %d %Y", "%b %d %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(s.title(), fmt).date()
        except ValueError:
            pass

    for fmt in ("%B %d", "%b %d"):
        try:
            d0 = datetime.strptime(s.title(), fmt).date()
            today = _owner_now_date(owner_tz)
            guess = date(year=today.year, month=d0.month, day=d0.day)
            if guess < today:
                guess = date(year=today.year + 1, month=d0.month, day=d0.day)
            return guess
        except ValueError:
            pass

    today = _owner_now_date(owner_tz)
    if s == "today":
        return today
    if s in ("tomorrow", "tmrw"):
        return today + timedelta(days=1)

    m = re.fullmatch(
        r"(next|this)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s
    )
    if m:
        which, wname = m.groups()
        widx = WEEKDAYS[wname]
        delta = (widx - today.weekday()) % 7
        if which == "next" or delta == 0:
            delta = delta or 7
        return today + timedelta(days=delta)

    raise ValueError("Unrecognized day")


def _parse_owner_local_dt(iso_local: str, owner_tz: str) -> datetime:
    """Parse a local datetime string into an aware datetime in owner tz.

    Accepts:
      - 'YYYY-MM-DDTHH:MM' (no tz, interpreted in owner's tz)
      - ISO8601 with optional seconds
      - ISO8601 with 'Z' or explicit offset (e.g., '-04:00')
      - Lenient natural forms like "Oct 1 10am" or "10:00 Oct 1 2025"

    Args:
        iso_local: Free‑form datetime string.
        owner_tz: IANA timezone name to interpret local times.

    Returns:
        Timezone‑aware 'datetime' localized to 'owner_tz'.

    Raises:
        ValueError: If the string cannot be parsed into a datetime.
    """
    s = iso_local.strip()
    tz = ZoneInfo(owner_tz)

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=tz)
        return dt.astimezone(tz)
    except ValueError:
        pass
    raw = s
    s = s.replace(",", " ")
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+at\s+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+on\s+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()

    def _convert_am_pm(txt: str) -> str:
        def repl(m: re.Match) -> str:
            hh = int(m.group(1))
            mm = m.group(2)
            ap = m.group(3).lower()
            if ap == "am":
                hour = 0 if hh == 12 else hh
            else:
                hour = 12 if hh == 12 else hh + 12
            return f"{hour:02d}:{mm or '00'}"

        return re.sub(
            r"\b(1[0-2]|0?[1-9])(?::([0-5][0-9]))?\s*(am|pm)\b",
            repl,
            txt,
            flags=re.IGNORECASE,
        )

    s = _convert_am_pm(s)

    for fmt in ("%Y-%m-%d %H:%M", "%B %d %Y %H:%M", "%b %d %Y %H:%M"):
        try:
            dt = datetime.strptime(s.title(), fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            pass
    for fmt in ("%H:%M %B %d %Y", "%H:%M %b %d %Y"):
        try:
            dt = datetime.strptime(s.title(), fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            pass
    for fmt in (
        "%Y-%m-%d %H",
        "%B %d %Y %H",
        "%b %d %Y %H",
        "%H %B %d %Y",
        "%H %b %d %Y",
    ):
        try:
            dt = datetime.strptime(s.title(), fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            pass
    for fmt in ("%B %d %H:%M", "%b %d %H:%M"):
        try:
            today = _owner_now_date(owner_tz)
            dt0 = datetime.strptime(s.title(), fmt)
            dt = dt0.replace(year=today.year)
            if dt.replace(tzinfo=tz) < datetime.now(tz):
                dt = dt.replace(year=today.year + 1)
            return dt.replace(tzinfo=tz)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized datetime format: {raw!r}")


def _to_utc(dt: datetime) -> datetime:
    """Convert a tz‑aware datetime to UTC (no‑op if already UTC)."""
    return dt.astimezone(ZoneInfo("UTC"))


def _owner_tz_from_config(
    config: RunnableConfig, default: str = "America/Toronto"
) -> str:
    """Extract the owner's timezone from config or contextvars.

    Args:
        config: Runnable configuration with optional 'configurable.tz'.
        default: Fallback timezone when none is provided.

    Returns:
        Timezone name string.
    """
    cfg = (
        (config or {}).get("configurable", {})
        if isinstance(config, dict)
        else (getattr(config, "configurable", None) or {})
    )
    return cfg.get("tz") or tz_var.get() or default


def _owner_id_from_config(config: RunnableConfig) -> str:
    """Extract the owner id from config, contextvars, or environment.

    Priority:
      1) 'config.configurable.user_id' or 'config.configurable.owner_id'
      2) contextvar 'owner_id_var'
      3) env 'OWNER_ID_DEFAULT' / 'OWNER_ID'

    Args:
        config: Runnable configuration object or dict.

    Returns:
        Owner id as a string.

    Raises:
        ToolException: If no owner id can be determined.
    """
    cfg = (
        (config or {}).get("configurable", {})
        if isinstance(config, dict)
        else (getattr(config, "configurable", None) or {})
    )
    owner_id = cfg.get("user_id") or cfg.get("owner_id") or owner_id_var.get()
    if not owner_id:
        owner_id = os.getenv("OWNER_ID_DEFAULT") or os.getenv("OWNER_ID")
    if not owner_id:
        raise ToolException("Missing owner id in tool config")
    return str(owner_id)


def _set_if_column(
    appt: object, column_set: set[str], name: str, value: object
) -> bool:
    """Conditionally set an attribute on an appointment ORM object.

    Sets 'name' if present in the known column set or in alias mappings.

    Args:
        appt: Appointment‑like ORM object.
        column_set: Columns present on the mapped table.
        name: Canonical field name intended to update.
        value: Value to set (ignored if None).

    Returns:
        True if any field was set; otherwise False.
    """
    if value is None:
        return False
    if name in column_set:
        setattr(appt, name, value)
        return True
    aliases = {
        "private_note": ["owner_private_note", "note_private"],
        "attendance": ["visit_attendance", "attended_status"],
        "late_minutes": ["lateness_minutes", "late_mins"],
        "payment_status": ["pay_status", "payment_state"],
        "amount_paid_cents": ["amount_paid", "paid_cents"],
        "price_override_cents": ["price_cents_override", "override_price_cents"],
        "bundle_id": ["pack_id", "bundle_uuid"],
    }
    for alt in aliases.get(name, []):
        if alt in column_set:
            setattr(appt, alt, value)
            return True
    return False
