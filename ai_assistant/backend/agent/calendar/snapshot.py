from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, ToolException
from agent.schemas import ToolCalendarSnapshotIn, ToolCalendarSnapshotOut
from app.db import SessionLocal
from app.models import User, Appointment, SpecialOpening, Person, ClientAccount
from sqlalchemy import func
from agent.constants import ACTIVE_APPT_STATUSES

from services.services_scheduling import owner_calendar_snapshot
from agent.tools import _parse_owner_day
from agent.calendar.openings import _owner_id_from_config, _iso_local
import re
import uuid
import logging

log = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _ensure_dt(v):
    """Best-effort parse of a datetime-like value.

    - Accepts a 'datetime' instance or an ISO-8601 string (trailing 'Z'
      allowed).
    - Returns a 'datetime' on success, or 'None' if parsing fails.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        s = v.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def _to_utc_dt(d, fallback_tz: str) -> datetime:
    """Convert a datetime or ISO-8601 string to a timezone-aware UTC datetime.

    - If the input is naive (no tzinfo), applies 'fallback_tz' first.
    - Supports 'datetime' and ISO-8601 strings; raises 'ValueError' otherwise.
    """
    if isinstance(d, str):
        d = datetime.fromisoformat(d.replace("Z", "+00:00"))
    if not isinstance(d, datetime):
        raise ValueError(f"Expected datetime/iso, got: {type(d)}")
    if d.tzinfo is None:
        d = d.replace(tzinfo=ZoneInfo(fallback_tz))
    return d.astimezone(ZoneInfo("UTC"))


def _extract_uuid(value) -> str | None:
    """Extract a UUID string from 'value' or embedded within its text.

    - Returns a canonical UUID string on success, otherwise 'None'.
    - Accepts raw UUIDs or values containing a UUID substring.
    """
    if value is None:
        return None
    s = str(value)
    try:
        return str(uuid.UUID(s))
    except Exception:
        m = _UUID_RE.search(s)
        if not m:
            return None
        try:
            return str(uuid.UUID(m.group(0)))
        except Exception:
            return None


def _pretty_local(dt: datetime, tz_name: str) -> str:
    """Format a datetime in 'tz_name' as a human-friendly string.

    Example: "Mon Jan 1, 3:00 PM". Falls back to the given datetime if
    timezone conversion fails.
    """
    try:
        local_dt = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        local_dt = dt
    return local_dt.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


def _iso_utc(dt: datetime) -> str:
    """Return an ISO-8601 string (minute precision) in UTC."""
    return dt.astimezone(ZoneInfo("UTC")).isoformat(timespec="minutes")


@tool("calendar_snapshot", args_schema=ToolCalendarSnapshotIn, return_direct=False)
def calendar_snapshot_tool(
    scope: str,
    anchor: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolCalendarSnapshotOut:
    """Return an enriched calendar snapshot for the owner.

    Arguments
    - 'scope': One of 'today', 'week', or 'month'. Aliases: 'day' is treated as
      'today'; 'tomorrow' is treated as 'today' with 'anchor="tomorrow"'.
    - 'anchor': Optional owner-local reference (e.g., 'YYYY-MM-DD', 'today',
      'tomorrow', 'next monday'). When omitted, uses the current owner-local day.
    - 'config': LangChain Runnable config carrying owner identity and optional
      timezone override via 'configurable.tz'/'owner_tz'/'client_tz'.

    Behavior
    - Calls 'owner_calendar_snapshot' to fetch events and a window ('start'/'end').
    - Normalizes each event to include:
      'start_local', 'end_local', 'start_local_pretty', 'end_local_pretty',
      'start_utc', 'end_utc', and coerces 'person_id' to string when present.
    - For openings, tags 'meta.source' as 'special' when overlapping a
      'SpecialOpening' in the window, otherwise 'weekly'.
    - When configured, may opportunistically backfill appointment identity
      fields from the DB ('BACKFILL_APPT_IDENTITY_ON_SNAPSHOT').

    Returns
    - 'ToolCalendarSnapshotOut' with:
      - 'tz': Effective timezone name.
      - 'start'/'end': Window bounds.
      - 'events': List of enriched events (dicts with normalized fields).
      - 'pretty_lines': Concise human-friendly descriptions for display.

    Errors
    - 'ToolException("Owner not found")' if owner resolution fails.
    - 'ToolException("calendar_snapshot failed: ...")' for unexpected errors.
    """
    try:
        owner_id = _owner_id_from_config(config)
        if scope == "day":
            scope = "today"
        if scope == "tomorrow":
            scope = "today"
            anchor = anchor or "tomorrow"
        if scope == "week":
            a_str = (anchor or "").strip().lower()
            if a_str in ("today", "tomorrow"):
                scope = "today"
        log.info(
            "calendar_snapshot_tool(scope=%r, anchor=%r) owner_id=%s",
            scope,
            anchor,
            owner_id,
        )

        cfg = (
            (config or {}).get("configurable", {})
            if isinstance(config, dict)
            else (getattr(config, "configurable", None) or {})
        )
        tz_override: Optional[str] = (
            cfg.get("tz") or cfg.get("owner_tz") or cfg.get("client_tz")
        )

        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            owner_tz = tz_override or owner.timezone or "America/Toronto"
            anchor_date = None
            if anchor:
                a = str(anchor).strip().lower()
                try:
                    if a in ("today", "tomorrow"):
                        anchor_date = _parse_owner_day(a, owner_tz)
                    elif a in ("this week", "week", "current week"):
                        anchor_date = _parse_owner_day("today", owner_tz)
                    elif a in ("next week", "next_week"):
                        now = datetime.now(ZoneInfo(owner_tz))
                        today_local = now.date()
                        wd = today_local.weekday()
                        days_until_next_monday = (7 - wd) or 7
                        next_monday = today_local + timedelta(
                            days=days_until_next_monday
                        )
                        anchor_date = next_monday
                    else:
                        anchor_date = _parse_owner_day(a, owner_tz)
                except Exception:
                    anchor_date = None

            snap = owner_calendar_snapshot(
                db, owner_id=owner.id, scope=scope, anchor=anchor_date, tz_str=owner_tz
            )
            tz_name = snap.get("tz") or owner_tz
            events_in = list(snap.get("events", []))

            s_utc_win = _to_utc_dt(snap["start"], tz_name)
            e_utc_win = _to_utc_dt(snap["end"], tz_name)

            specials_in_window: List[SpecialOpening] = (
                db.query(SpecialOpening)
                .filter(
                    SpecialOpening.owner_id == owner.id,
                    SpecialOpening.start_utc < e_utc_win,
                    SpecialOpening.end_utc > s_utc_win,
                )
                .all()
            )

            events_out: List[Dict[str, Any]] = []
            for ev in events_in:
                e = dict(ev)
                start_dt = _ensure_dt(e.get("start"))
                end_dt = _ensure_dt(e.get("end"))

                if e.get("type") == "appointment":
                    try:
                        appt_id_raw = e.get("id")
                        meta = dict(e.get("meta") or {})
                        person_id = meta.get("person_id")
                        client_name = meta.get("client_name") or meta.get("person_name")
                        client_email = meta.get("client_email")

                        appt_db_id = _extract_uuid(appt_id_raw)
                        appt_row = None
                        if appt_db_id:
                            appt_row = (
                                db.query(Appointment, Person)
                                .outerjoin(Person, Appointment.person_id == Person.id)
                                .filter(
                                    Appointment.owner_id == owner.id,
                                    Appointment.id == appt_db_id,
                                )
                                .first()
                            )

                        if (
                            appt_row
                            and getattr(appt_row[0], "status", None)
                            in ACTIVE_APPT_STATUSES
                        ):
                            appt, p = appt_row
                            resolved_person_id = getattr(appt, "person_id", None)
                            resolved_name = (
                                p.full_name
                                if p and p.full_name
                                else getattr(appt, "client_name", None)
                            )
                            resolved_email = (
                                p.email
                                if p and p.email
                                else getattr(appt, "client_email", None)
                            )

                            e["person_id"] = (
                                str(resolved_person_id)
                                if resolved_person_id is not None
                                else None
                            )
                            e["client_name"] = resolved_name
                            e["client_email"] = resolved_email

                            BACKFILL_APPT_IDENTITY_ON_SNAPSHOT = False
                            try:
                                from agent.tools_calendar import (
                                    BACKFILL_APPT_IDENTITY_ON_SNAPSHOT as _BF,
                                )

                                BACKFILL_APPT_IDENTITY_ON_SNAPSHOT = bool(_BF)
                            except Exception:
                                pass
                            if BACKFILL_APPT_IDENTITY_ON_SNAPSHOT:
                                needs_update = False
                                if not resolved_person_id and resolved_name:
                                    match = (
                                        db.query(Person)
                                        .join(
                                            ClientAccount,
                                            ClientAccount.id == Person.account_id,
                                        )
                                        .filter(
                                            ClientAccount.owner_user_id == owner.id,
                                            func.lower(Person.full_name)
                                            == resolved_name.strip().lower(),
                                        )
                                        .all()
                                    )
                                    if len(match) == 1:
                                        appt.person_id = match[0].id
                                        resolved_person_id = match[0].id
                                        needs_update = True
                                        if not resolved_email and match[0].email:
                                            appt.client_email = match[0].email
                                            resolved_email = match[0].email
                                if not resolved_email and resolved_person_id:
                                    p2 = (
                                        db.query(Person)
                                        .filter(Person.id == resolved_person_id)
                                        .first()
                                    )
                                    if p2 and p2.email:
                                        appt.client_email = p2.email
                                        resolved_email = p2.email
                                        needs_update = True
                                if needs_update:
                                    try:
                                        db.add(appt)
                                        db.commit()
                                        db.refresh(appt)
                                    except Exception:
                                        db.rollback()
                                e["person_id"] = (
                                    str(resolved_person_id)
                                    if resolved_person_id is not None
                                    else None
                                )
                                e["client_name"] = resolved_name
                                e["client_email"] = resolved_email
                        else:
                            e["person_id"] = (
                                str(person_id) if person_id is not None else None
                            )
                            e["client_name"] = client_name
                            e["client_email"] = client_email
                    except Exception as appt_enrich_err:
                        log.warning(
                            "Appointment enrich failed for id=%r: %s",
                            e.get("id"),
                            appt_enrich_err,
                        )

                if (
                    e.get("type") == "opening"
                    and isinstance(start_dt, datetime)
                    and isinstance(end_dt, datetime)
                ):
                    open_s_utc = (
                        start_dt
                        if start_dt.tzinfo
                        else start_dt.replace(tzinfo=ZoneInfo("UTC"))
                    )
                    open_e_utc = (
                        end_dt
                        if end_dt.tzinfo
                        else end_dt.replace(tzinfo=ZoneInfo("UTC"))
                    )
                    open_s_utc = open_s_utc.astimezone(ZoneInfo("UTC"))
                    open_e_utc = open_e_utc.astimezone(ZoneInfo("UTC"))

                    meta = dict(e.get("meta") or {})
                    tagged = False
                    for sp in specials_in_window:
                        if open_s_utc < sp.end_utc and sp.start_utc < open_e_utc:
                            meta["source"] = "special"
                            meta["special_id"] = str(sp.id)
                            tagged = True
                            break
                    if not tagged:
                        meta["source"] = "weekly"
                    e["meta"] = meta

                if isinstance(start_dt, datetime) and start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=ZoneInfo("UTC"))
                if isinstance(end_dt, datetime) and end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=ZoneInfo("UTC"))

                if isinstance(start_dt, datetime) and isinstance(end_dt, datetime):
                    e["start_local"] = _iso_local(start_dt, tz_name)
                    e["end_local"] = _iso_local(end_dt, tz_name)
                    e["start_local_pretty"] = _pretty_local(start_dt, tz_name)
                    e["end_local_pretty"] = _pretty_local(end_dt, tz_name)
                    e["start_utc"] = _iso_utc(start_dt)
                    e["end_utc"] = _iso_utc(end_dt)
                    e["start"] = start_dt
                    e["end"] = end_dt

                events_out.append(e)

            for ev in events_out:
                pid = ev.get("person_id")
                if pid is not None and not isinstance(pid, str):
                    try:
                        ev["person_id"] = str(pid)
                    except Exception:
                        ev["person_id"] = None

            log.info(
                "calendar_snapshot_tool -> %d events (tz=%s)", len(events_out), tz_name
            )

            def _line(
                kind: str,
                s_pretty: str,
                e_pretty: str,
                title: Optional[str] = None,
                status: Optional[str] = None,
            ):
                """Build a concise, human-friendly line for an event."""
                if kind == "appointment":
                    base = f"{s_pretty} – {e_pretty}: Appointment"
                    if status == "canceled":
                        base = f"{s_pretty} – {e_pretty}: Canceled Appointment"
                    if title and title.lower() != "appointment":
                        base = (
                            f"{s_pretty} – {e_pretty}: {title}"
                            if status != "canceled"
                            else f"{s_pretty} – {e_pretty}: Canceled {title}"
                        )
                    return base
                if kind == "time_off":
                    return f"{s_pretty} – {e_pretty}: Time Off"
                return f"{s_pretty} – {e_pretty}: Opening"

            events_sorted = sorted(
                events_out,
                key=lambda e: (
                    e.get("start_local") or "",
                    e.get("type") != "appointment",
                ),
            )

            pretty_lines: list[str] = []
            for e in events_sorted:
                kind = e.get("type")
                s_pretty = e.get("start_local_pretty") or ""
                e_pretty = e.get("end_local_pretty") or ""
                title = e.get("title")
                status = e.get("status")
                pretty_lines.append(
                    _line(kind, s_pretty, e_pretty, title=title, status=status)
                )

            return ToolCalendarSnapshotOut(
                tz=tz_name,
                start=snap["start"],
                end=snap["end"],
                events=events_out,
                pretty_lines=pretty_lines,
            )

    except ToolException:
        raise
    except Exception as e:
        log.error(
            "calendar_snapshot_tool failed (scope=%r, anchor=%r, owner_id=%s)",
            scope,
            anchor,
            owner_id,
            exc_info=True,
        )
        raise ToolException(f"calendar_snapshot failed: {e}")
