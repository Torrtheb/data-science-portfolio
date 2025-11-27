from __future__ import annotations

from typing import Optional, List, Dict
from datetime import datetime
import json
import re
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from langchain_core.tools import tool, ToolException
from langchain_core.runnables import RunnableConfig

from agent.tool_ctx import tz_var
from agent.schemas import (
    ToolGetPublicHolidaysIn,
    ToolGetPublicHolidaysOut,
    ToolIsPublicHolidayIn,
    ToolIsPublicHolidayOut,
)
import os


_HOLIDAY_CACHE: dict[str, List[Dict]] = {}
DEFAULT_COUNTRY = os.getenv("HOLIDAYS_DEFAULT_COUNTRY", "CA").strip().upper() or "CA"
DEFAULT_REGION = (
    os.getenv("HOLIDAYS_DEFAULT_REGION", "CA-NB").strip().upper() or "CA-NB"
)


def _validate_country(code: str) -> str:
    """Ensure a country code looks like ISO-3166 alpha-2 (e.g., 'US', 'GB')."""
    code = code.strip().upper()
    if not re.fullmatch(r"[A-Z]{2}", code):
        raise ToolException(
            "BAD_COUNTRY_CODE: provide ISO-3166 alpha-2 like 'US', 'GB', 'DE'"
        )
    return code


def _current_year_owner_tz() -> int:
    """Return the current year in the owner's timezone (from tool context)."""
    tz = tz_var.get() or "UTC"
    now = datetime.now().astimezone()
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo(tz))
    except Exception:
        now = datetime.now().astimezone()
    return now.year


def _fetch_holidays(year: int, country_code: str, timeout: float = 3.0) -> List[Dict]:
    """Fetch and normalize public holidays from Nager.Date for year/country.

    Returns a list of dicts with at least: date, name/localName, countryCode, global, types, counties.
    Results are cached in-process for the (year,country) key.
    """
    key = f"{year}|{country_code}"
    if key in _HOLIDAY_CACHE:
        return _HOLIDAY_CACHE[key]

    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    req = Request(url, headers={"User-Agent": "aieng-3-agent/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise ToolException(f"HOLIDAY_LOOKUP_FAILED: HTTP {resp.status}")
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise ToolException(f"HOLIDAY_LOOKUP_FAILED: HTTP {e.code}")
    except URLError:
        raise ToolException("HOLIDAY_LOOKUP_FAILED: network error")
    except Exception:
        raise ToolException("HOLIDAY_LOOKUP_FAILED: unexpected error")
    out: List[Dict] = []
    for row in data or []:
        try:
            out.append(
                {
                    "date": row.get("date"),
                    "name": row.get("name"),
                    "localName": row.get("localName"),
                    "countryCode": row.get("countryCode", country_code),
                    "global": bool(row.get("global", False)),
                    "types": row.get("types"),
                    "counties": row.get("counties"),
                }
            )
        except Exception:
            continue

    _HOLIDAY_CACHE[key] = out
    return out


@tool("get_public_holidays", args_schema=ToolGetPublicHolidaysIn, return_direct=False)
def get_public_holidays_tool(
    country_code: Optional[str] = None,
    year: Optional[int] = None,
    region_code: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolGetPublicHolidaysOut:
    """List public holidays for a given country and year.

    Args:
        country_code: ISO‑3166 alpha‑2 code (e.g., "US", "GB", "DE"). Defaults
            to 'HOLIDAYS_DEFAULT_COUNTRY' or "CA".
        year: Four‑digit year; defaults to the owner's current year (via
            timezone from tool context).
        region_code: Optional region (e.g., "CA-NB") to filter provincial/state
            holidays; defaults to 'HOLIDAYS_DEFAULT_REGION' if set.
        config: Unused; present for consistency with other tools.

    Returns:
        'ToolGetPublicHolidaysOut' with a list of normalized holiday dicts
        containing fields like 'date', 'name', 'localName',
        'countryCode', 'global', 'types', and 'counties'.

    Raises:
        ToolException: For invalid country codes or network/API errors.
    """
    cc = _validate_country((country_code or DEFAULT_COUNTRY))
    y = int(year) if year else _current_year_owner_tz()
    holidays = _fetch_holidays(y, cc)
    rc = (
        (region_code or DEFAULT_REGION).strip().upper()
        if (region_code or DEFAULT_REGION)
        else None
    )
    if rc:
        filtered = []
        for h in holidays:
            counties = h.get("counties")
            if not counties:
                filtered.append(h)
            elif rc in counties:
                filtered.append(h)
        holidays = filtered
    return ToolGetPublicHolidaysOut(holidays=holidays)


@tool("is_public_holiday", args_schema=ToolIsPublicHolidayIn, return_direct=False)
def is_public_holiday_tool(
    date: str,
    country_code: Optional[str] = None,
    region_code: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolIsPublicHolidayOut:
    """Check if a given owner‑local date is a public holiday.

    Args:
        date: Date string in 'YYYY-MM-DD' format (owner‑local).
        country_code: ISO‑3166 alpha‑2 code (defaults as in
            'get_public_holidays_tool').
        region_code: Optional region code to scope holidays (e.g., "CA-NB").
        config: Unused; present for consistency with other tools.

    Returns:
        'ToolIsPublicHolidayOut' with 'is_holiday' and an optional
        'name' when the date is a holiday.

    Raises:
        ToolException: For bad date format or underlying lookup failures.
    """
    cc = _validate_country((country_code or DEFAULT_COUNTRY))
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ToolException("BAD_DATE: use 'YYYY-MM-DD'")

    holidays = _fetch_holidays(dt.year, cc)
    rc = (
        (region_code or DEFAULT_REGION).strip().upper()
        if (region_code or DEFAULT_REGION)
        else None
    )
    for h in holidays:
        if h.get("date") == date:
            counties = h.get("counties")
            applies = False
            if not counties:
                applies = True
            elif rc and rc in counties:
                applies = True
            if applies:
                name = h.get("localName") or h.get("name")
                return ToolIsPublicHolidayOut(
                    is_holiday=True, name=str(name) if name else None
                )
    return ToolIsPublicHolidayOut(is_holiday=False, name=None)
