from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, time as _time, timedelta
from zoneinfo import ZoneInfo
import json

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.db import SessionLocal
from app.models import (
    User,
    TimeOff,
    SpecialOpening,
    Appointment,
    AvailabilityRule,
)
from services.services_scheduling import owner_calendar_snapshot
from agent.schemas import (
    ToolCalendarSnapshotIn,
    ToolAddSpecialOpeningIn,
    ToolAddSpecialOpeningOut,
    ToolCalendarSnapshotOut,
)
from langchain_core.tools import ToolException
from agent.constants import ACTIVE_APPT_STATUSES
from sqlalchemy.exc import IntegrityError
from services.services_scheduling import (
    carve_opening_through_timeoff,
    merge_or_get_special_opening,
)
from agent.tools import _parse_owner_day, _parse_owner_local_dt, _to_utc
import re

import os
import sqlalchemy as sa
import uuid
from datetime import date as _date
from typing import Union
from agent.tool_ctx import owner_id_var
from sqlalchemy import func
from app.models import Person, ClientAccount
import logging

# Re-export time off status helper for convenience
from agent.calendar.timeoff import next_time_off_tool  # noqa: E402

log = logging.getLogger(__name__)

BACKFILL_APPT_IDENTITY_ON_SNAPSHOT = os.getenv(
    "BACKFILL_APPT_IDENTITY_ON_SNAPSHOT", "false"
).lower() in ("1", "true", "yes")

NONCANCELLED_STATUSES = {"booked", "confirmed", "pending"}
DISPLAY_OPENING_RE = re.compile(r"^open-(\d+)$")
CHOICE_RE = re.compile(r"CHOICE_REQUIRED:(\{.*\})", re.S)
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    """Return True if intervals [a_start,a_end) and [b_start,b_end) overlap."""
    return a_start < b_end and a_end > b_start


def _appt_cols() -> tuple[str, str]:
    """Return the Appointment start/end UTC column names present on the model.

    Supports schemas that use either 'start_utc/end_utc' or legacy
    'start_time_utc/end_time_utc' columns.
    """
    cols = {c.name for c in Appointment.__table__.columns}
    start_col = "start_utc" if "start_utc" in cols else "start_time_utc"
    end_col = "end_utc" if "end_utc" in cols else "end_time_utc"
    return start_col, end_col


class ToolTruncateAfterIn(BaseModel):
    """Input to truncate openings after a local time for a day or weekly rule."""

    local_hhmm: str = Field(..., description="e.g. '15:40' for owner-local cutoff")
    day: Optional[str] = Field(
        default="today",
        description="Anchor day: 'today' | 'tomorrow' | 'YYYY-MM-DD' | 'next friday'",
    )
    scope: Optional[Literal["day", "weekly"]] = Field(
        default=None,
        description="If omitted and a weekly rule is affected, the tool will ask the user to choose.",
    )


class ToolTruncateAfterOut(BaseModel):
    """Result of truncating openings, including affected rule ids and any blocks."""

    ok: bool
    updated: List[str] = []
    deleted: List[str] = []
    cutoff_local: str
    requires_cancellation: bool = False
    blocked_appointments: List[Dict[str, Any]] = []


# Removed unused helpers: _minutes_since_midnight, _fmt_hhmm, _weekday_of_date, _touching_or_overlapping


def _merge_overlapping_or_touching_openings(
    db, owner: User, new_row: SpecialOpening
) -> SpecialOpening:
    """
    Merge the newly-created SpecialOpening with any other openings that overlap or touch it.
    Keeps slot_minutes/buffer_minutes of the earliest row (or choose a rule you prefer).
    Returns the merged row.
    """
    q = (
        db.query(SpecialOpening)
        .filter(
            SpecialOpening.owner_id == owner.id,
            SpecialOpening.id != new_row.id,
            SpecialOpening.start_utc <= new_row.end_utc,
            SpecialOpening.end_utc >= new_row.start_utc,
        )
        .order_by(SpecialOpening.start_utc.asc())
    )
    rows = q.all()

    if not rows:
        rows = (
            db.query(SpecialOpening)
            .filter(
                SpecialOpening.owner_id == owner.id,
                SpecialOpening.id != new_row.id,
                sa.or_(
                    SpecialOpening.end_utc == new_row.start_utc,
                    SpecialOpening.start_utc == new_row.end_utc,
                ),
            )
            .order_by(SpecialOpening.start_utc.asc())
            .all()
        )

    if not rows:
        return new_row
    all_rows = rows + [new_row]
    merged_start = min(r.start_utc for r in all_rows)
    merged_end = max(r.end_utc for r in all_rows)
    keep = min(all_rows, key=lambda r: (r.start_utc, r.id))
    keep.start_utc = merged_start
    keep.end_utc = merged_end
    for r in all_rows:
        if r.id != keep.id:
            db.delete(r)
    db.commit()
    db.refresh(keep)
    return keep


def _list_conflicts(
    db, owner: User, s_utc: datetime, e_utc: datetime, owner_tz: str
) -> Dict[str, Any]:
    """Return overlapping time off, appointments, and openings for a UTC range.

    Returns a dict with lists keyed by 'timeoffs', 'appointments', and 'openings'.
    """
    timeoffs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner.id,
            TimeOff.start_utc < e_utc,
            TimeOff.end_utc > s_utc,
        )
        .order_by(TimeOff.start_utc.asc())
        .all()
    )

    # Appointment conflicts (active only)
    scol, ecol = _appt_cols()
    appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            getattr(Appointment, scol) < e_utc,
            getattr(Appointment, ecol) > s_utc,
            (Appointment.status.in_(ACTIVE_APPT_STATUSES)),
        )
        .order_by(getattr(Appointment, scol).asc())
        .all()
    )

    # Existing availability (BOTH weekly rules & special openings) using your snapshot generator
    tz = ZoneInfo(owner_tz)
    day = s_utc.astimezone(tz).date()
    snap = owner_calendar_snapshot(
        db,
        owner_id=owner.id,
        scope="today",
        anchor=_date(day.year, day.month, day.day),
        tz_str=owner_tz,
    )

    existing_openings = []
    for ev in snap.get("events", []):
        if ev.get("type") == "opening":
            st = ev["start"]
            en = ev["end"]
            if isinstance(st, str):
                st = datetime.fromisoformat(st.replace("Z", "+00:00"))
            if isinstance(en, str):
                en = datetime.fromisoformat(en.replace("Z", "+00:00"))
            if st.tzinfo is None:
                st = st.replace(tzinfo=ZoneInfo("UTC"))
            if en.tzinfo is None:
                en = en.replace(tzinfo=ZoneInfo("UTC"))
            if _overlaps(s_utc, e_utc, st, en):
                existing_openings.append(
                    {"start_utc": st, "end_utc": en, "id": ev.get("id")}
                )

    return {
        "timeoffs": timeoffs,
        "appointments": appts,
        "openings": existing_openings,
    }


def _fmt_local_range(s_utc: datetime, e_utc: datetime, tz_name: str) -> str:
    return f"{_pretty_local(s_utc, tz_name)} → {_pretty_local(e_utc, tz_name)}"


# Removed unused helpers: _owner_local_day_window, _conflicting_appointments, _parse_opening_id_any


def _owner_id_from_config(config: RunnableConfig) -> str:
    """
    Resolve the owner id from LCEL config or environment.
    Priority:
      1) config['configurable']['user_id'] or ['owner_id']
     2) contextvars (owner_id_var set by graph ToolNode wrapper)
     3) env OWNER_ID_DEFAULT
     4) env OWNER_ID
    Raises ToolException if none found.
    """
    cfg = (
        (config or {}).get("configurable", {})
        if isinstance(config, dict)
        else (getattr(config, "configurable", None) or {})
    )
    owner_id = cfg.get("user_id") or cfg.get("owner_id")
    if not owner_id:
        owner_id = owner_id_var.get()
    if not owner_id:
        owner_id = os.getenv("OWNER_ID_DEFAULT") or os.getenv("OWNER_ID")

    if not owner_id:
        raise ToolException("Missing owner id in tool config")
    return str(owner_id)


def _resolve_opening_row(
    db,
    owner: User,
    opening_identifier: Union[int, str],
    day: Optional[str] = None,
) -> Optional[SpecialOpening]:
    """
    Accepts:
      - UUID string (exact SpecialOpening.id)  -> returns that row
      - numeric string or int (legacy)         -> tries exact id match as string
      - 'open-<n>' (display id from snapshot)  -> maps to the nth opening on the given owner-local 'day'
                                                  (default = today), ordered by start_utc ascending
    """
    s = str(opening_identifier).strip()

    # 1) Try direct UUID / exact string id match
    row = (
        db.query(SpecialOpening)
        .filter(SpecialOpening.owner_id == owner.id, SpecialOpening.id == s)
        .first()
    )
    if row:
        return row

    # 2) If it's 'open-<n>', resolve against openings for the requested owner-local day
    m = DISPLAY_OPENING_RE.match(s)
    if m:
        idx = int(m.group(1))
        tz = ZoneInfo(owner.timezone)
        if day:
            d = _parse_owner_day(day, owner.timezone)
        else:
            d = datetime.now(tz).date()

        day_start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
        day_end_local = datetime(d.year, d.month, d.day, 23, 59, tzinfo=tz)
        u_start = _to_utc(day_start_local)
        u_end = _to_utc(day_end_local)

        rows: List[SpecialOpening] = (
            db.query(SpecialOpening)
            .filter(
                SpecialOpening.owner_id == owner.id,
                SpecialOpening.start_utc < u_end,
                SpecialOpening.end_utc > u_start,
            )
            .order_by(SpecialOpening.start_utc.asc())
            .all()
        )
        if 0 <= idx < len(rows):
            return rows[idx]
        return None

    # 3) Fallback: numeric-as-string exact match
    row = (
        db.query(SpecialOpening)
        .filter(SpecialOpening.owner_id == owner.id, SpecialOpening.id == s)
        .first()
    )
    return row


class ToolListOpeningsIn(BaseModel):
    day: str = Field(
        ..., description="Owner-local day like 'YYYY-MM-DD' or 'today'/'tomorrow'"
    )


class ToolListOpeningsOut(BaseModel):
    openings: List[Dict]


class ToolUpdateOpeningIn(BaseModel):
    opening_id: Union[int, str] = Field(
        ..., description="DB id or display id like 'open-3'"
    )
    start_local: Optional[str] = None
    end_local: Optional[str] = None
    slot_minutes: Optional[int] = Field(None, ge=5, le=360)
    buffer_minutes: Optional[int] = Field(None, ge=0)
    note: Optional[str] = None
    day: Optional[str] = Field(
        None,
        description="Owner-local day for resolving 'open-<n>' (e.g. 'today', 'tomorrow', 'YYYY-MM-DD')",
    )


class ToolDeleteOpeningIn(BaseModel):
    opening_id: Union[int, str] = Field(
        ..., description="DB id or display id like 'open-3'"
    )
    day: Optional[str] = Field(
        None,
        description="Owner-local day for resolving 'open-<n>' (e.g. 'today', 'tomorrow', 'YYYY-MM-DD')",
    )


class ToolDeleteOpeningOut(BaseModel):
    deleted: bool
    opening_id: str


# ---------- local/format helpers ----------


def _ensure_dt(v: Any) -> Optional[datetime]:
    """Accept datetime or ISO string. Returns tz-aware datetime if possible."""
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


def _pretty_local(dt: datetime, tz_name: str) -> str:
    """e.g., 'Fri Sep 26, 9:00 AM' (owner-local)."""
    try:
        local_dt = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        local_dt = dt
    return local_dt.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


def _iso_local(dt: datetime, tz_name: str) -> str:
    try:
        local_dt = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        local_dt = dt
    return local_dt.isoformat(timespec="minutes")


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).isoformat(timespec="minutes")


# Removed unused helper: _coerce_window_to_utc


def _extract_uuid(value) -> Optional[str]:
    """
    Return a canonical UUID string if one can be extracted from 'value'.
    Accepts bare UUIDs or ids with prefixes like 'appt-<uuid>'.
    """
    if value is None:
        return None
    s = str(value)
    try:
        return str(uuid.UUID(s))
    except Exception:
        pass
    m = _UUID_RE.search(s)
    if not m:
        return None
    try:
        return str(uuid.UUID(m.group(0)))
    except Exception:
        return None


def _to_utc_dt(d, fallback_tz: str) -> datetime:
    """
    Coerce 'd' (datetime or ISO string) to an aware UTC datetime.
    If 'd' is naive, treat it as 'fallback_tz'.
    """
    if isinstance(d, str):
        d = datetime.fromisoformat(d.replace("Z", "+00:00"))
    if not isinstance(d, datetime):
        raise ValueError(f"Expected datetime/iso, got: {type(d)}")
    if d.tzinfo is None:
        d = d.replace(tzinfo=ZoneInfo(fallback_tz))
    return d.astimezone(ZoneInfo("UTC"))


# Removed unused helper: _ensure_dt_aware_utc


@tool("calendar_snapshot", args_schema=ToolCalendarSnapshotIn, return_direct=False)
def calendar_snapshot_tool(
    scope: str,
    anchor: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolCalendarSnapshotOut:
    """Return an enriched calendar snapshot for the owner.

    Args:
        scope: One of "today", "week", or "month" ("day" coerces to "today").
        anchor: Optional natural language or date anchor (e.g., "tomorrow").
        config: Runnable configuration providing the owner id and optional tz.

    Returns:
        'ToolCalendarSnapshotOut' including normalized events and
        'pretty_lines' for display.

    Raises:
        ToolException: If the owner cannot be resolved or an internal error
        occurs while building the snapshot.
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

            # Base snapshot (may contain 'appointment' and 'opening' events)
            snap = owner_calendar_snapshot(
                db, owner_id=owner.id, scope=scope, anchor=anchor_date, tz_str=owner_tz
            )
            tz_name = snap.get("tz") or owner_tz
            events_in = list(snap.get("events", []))
            s_utc_win = _to_utc_dt(snap["start"], tz_name)
            e_utc_win = _to_utc_dt(snap["end"], tz_name)

            # Pull one-off openings that overlap the snapshot window
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

                # --- Tag openings by source (weekly vs special) ---
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

                # --- Normalize / compute presentational fields ---
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

            # Ensure person_id is a string for all events to satisfy the schema
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
                if kind == "appointment":
                    base = f"{s_pretty} – {e_pretty}: Appointment (Lesson)"
                    if status == "canceled":
                        base = f"{s_pretty} – {e_pretty}: Canceled Appointment (Lesson)"
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


@tool("add_special_opening", args_schema=ToolAddSpecialOpeningIn, return_direct=False)
def add_special_opening_tool(
    start_local: str,
    end_local: str,
    slot_minutes: int,
    buffer_minutes: int,
    note: Optional[str],
    config: RunnableConfig,
) -> ToolAddSpecialOpeningOut:
    """Create or merge a one-off opening without modifying weekly rules.

    Carves through overlapping time off, then merges with touching/overlapping
    specials to keep the calendar tidy.

    Args:
        start_local: Owner‑local start time (YYYY‑MM‑DDTHH:MM).
        end_local: Owner‑local end time (YYYY‑MM‑DDTHH:MM).
        slot_minutes: Slot length in minutes.
        buffer_minutes: Buffer between slots in minutes.
        note: Optional note.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolAddSpecialOpeningOut' describing the stored opening.

    Raises:
        ToolException: For missing owner, invalid times, or other errors.
    """
    owner_id = _owner_id_from_config(config)
    try:
        with SessionLocal.begin() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            s_loc = _parse_owner_local_dt(start_local, owner.timezone)
            e_loc = _parse_owner_local_dt(end_local, owner.timezone)
            if e_loc <= s_loc:
                raise ToolException("end_local must be after start_local")

            s_utc, e_utc = _to_utc(s_loc), _to_utc(e_loc)

            # 1) Carve through TimeOff so specials actually apply
            carve_opening_through_timeoff(db, owner.id, s_utc, e_utc)

            # 2) Merge/reuse special openings (no conflicting insert)
            sp = merge_or_get_special_opening(
                db,
                owner.id,
                s_utc,
                e_utc,
                slot_minutes=int(slot_minutes),
                buffer_minutes=int(buffer_minutes),
                note=note,
            )

        with SessionLocal() as db_ro:
            sp = db_ro.query(SpecialOpening).filter(SpecialOpening.id == sp.id).first()

            return ToolAddSpecialOpeningOut(
                id=str(sp.id),
                start_utc=sp.start_utc,
                end_utc=sp.end_utc,
                slot_minutes=int(sp.slot_minutes or slot_minutes),
                buffer_minutes=int(sp.buffer_minutes or buffer_minutes),
                note=sp.note or note,
            )

    except IntegrityError:
        with SessionLocal.begin() as db2:
            owner = db2.query(User).filter(User.id == owner_id).first()
            tz = owner.timezone if owner else "America/Toronto"

            s_loc = _parse_owner_local_dt(start_local, tz)
            e_loc = _parse_owner_local_dt(end_local, tz)
            s_utc, e_utc = _to_utc(s_loc), _to_utc(e_loc)

            sp2 = merge_or_get_special_opening(
                db2,
                owner_id,
                s_utc,
                e_utc,
                slot_minutes=int(slot_minutes),
                buffer_minutes=int(buffer_minutes),
                note=note,
            )
        with SessionLocal() as db_ro:
            sp2 = (
                db_ro.query(SpecialOpening).filter(SpecialOpening.id == sp2.id).first()
            )
            return ToolAddSpecialOpeningOut(
                id=str(sp2.id),
                start_utc=sp2.start_utc,
                end_utc=sp2.end_utc,
                slot_minutes=int(sp2.slot_minutes or slot_minutes),
                buffer_minutes=int(sp2.buffer_minutes or buffer_minutes),
                note=sp2.note or note,
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"add_special_opening failed: {e}")


@tool("list_openings", args_schema=ToolListOpeningsIn, return_direct=False)
def list_openings_tool(day: str, config: RunnableConfig) -> ToolListOpeningsOut:
    """
    List one-off (special) openings on an owner-local day.

    Args:
        day: "today" | "tomorrow" | "YYYY-MM-DD".
        config: Must include configurable.user_id (or owner_id).

    Returns:
        ToolListOpeningsOut with openings enriched with local times.

    Raises:
        ToolException on owner/config errors.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            d = _parse_owner_day(day, owner.timezone)
            tz = ZoneInfo(owner.timezone)

            start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
            end_local = datetime(d.year, d.month, d.day, 23, 59, tzinfo=tz)

            start_utc = _to_utc(start_local)
            end_utc = _to_utc(end_local)

            rows = (
                db.query(SpecialOpening)
                .filter(
                    SpecialOpening.owner_id == owner.id,
                    SpecialOpening.start_utc < end_utc,
                    SpecialOpening.end_utc > start_utc,
                )
                .order_by(SpecialOpening.start_utc.asc())
                .all()
            )

            out = []
            for r in rows:
                out.append(
                    {
                        "opening_id": r.id,
                        "start_utc": r.start_utc.isoformat(timespec="minutes"),
                        "end_utc": r.end_utc.isoformat(timespec="minutes"),
                        "start_local": _iso_local(r.start_utc, owner.timezone),
                        "end_local": _iso_local(r.end_utc, owner.timezone),
                        "slot_minutes": r.slot_minutes,
                        "buffer_minutes": r.buffer_minutes,
                        "note": r.note,
                    }
                )
            return ToolListOpeningsOut(openings=out)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"list_openings failed: {e}")


@tool("update_opening", args_schema=ToolUpdateOpeningIn, return_direct=False)
def update_opening_tool(
    opening_id: Union[int, str],
    start_local: Optional[str] = None,
    end_local: Optional[str] = None,
    slot_minutes: Optional[int] = None,
    buffer_minutes: Optional[int] = None,
    note: Optional[str] = None,
    day: Optional[str] = None,
    config: RunnableConfig = None,
) -> Dict:
    """
    Update a one-off opening in place. Omitted fields are left unchanged.

    Args:
        opening_id: DB id or 'open-<n>' display id (resolved using 'day' when needed).
        start_local, end_local: Optional new times (owner-local ISO-ish).
        slot_minutes, buffer_minutes, note: Optional updates.
        day: Owner-local day used to resolve 'open-<n>'.
        config: Must include configurable.user_id (or owner_id).

    Returns:
        Dict with updated opening fields (local times, slot/buffer, note).

    Raises:
        ToolException if opening not found or invalid times.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            row: Optional[SpecialOpening] = _resolve_opening_row(
                db, owner, opening_id, day=day
            )
            if not row:
                raise ToolException(f"Opening not found for identifier: {opening_id!r}")

            if start_local:
                s_loc = _parse_owner_local_dt(start_local, owner.timezone)
                row.start_utc = _to_utc(s_loc)
            if end_local:
                e_loc = _parse_owner_local_dt(end_local, owner.timezone)
                row.end_utc = _to_utc(e_loc)
            if start_local or end_local:
                if row.end_utc <= row.start_utc:
                    raise ToolException("end must be after start")

            if slot_minutes is not None:
                row.slot_minutes = int(slot_minutes)
            if buffer_minutes is not None:
                row.buffer_minutes = int(buffer_minutes)
            if note is not None:
                row.note = note

            db.commit()
            db.refresh(row)
            row = _merge_overlapping_or_touching_openings(db, owner, row)
            db.refresh(row)
            return {
                "opening_id": str(row.id),
                "start_local": _iso_local(row.start_utc, owner.timezone),
                "end_local": _iso_local(row.end_utc, owner.timezone),
                "slot_minutes": row.slot_minutes,
                "buffer_minutes": row.buffer_minutes,
                "note": row.note,
            }
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"update_opening failed: {e}")


@tool("delete_opening", args_schema=ToolDeleteOpeningIn, return_direct=False)
def delete_opening_tool(
    opening_id: Union[int, str],
    day: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolDeleteOpeningOut:
    """
    Delete a one-off opening by id (DB id or display id 'open-<n>').

    Args:
        opening_id: The identifier to delete.
        day: Owner-local day used when resolving a display id.
        config: Must include configurable.user_id (or owner_id).

    Returns:
        ToolDeleteOpeningOut with deleted flag and opening_id.

    Raises:
        ToolException on owner/config errors. Missing opening is treated as success (idempotent).
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            row: Optional[SpecialOpening] = _resolve_opening_row(
                db, owner, opening_id, day=day
            )
            if not row:
                return ToolDeleteOpeningOut(deleted=True, opening_id=str(opening_id))

            row_id = str(row.id)
            db.delete(row)
            db.commit()
            return ToolDeleteOpeningOut(deleted=True, opening_id=row_id)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"delete_opening failed: {e}")


class ToolAddAvailabilityIn(BaseModel):
    start_local: str
    end_local: str
    slot_minutes: int = Field(..., ge=5, le=360)
    buffer_minutes: int = Field(..., ge=0)
    note: Optional[str] = None
    confirm_if_conflicts: bool = False


class ToolAddAvailabilityOut(BaseModel):
    id: str
    start_utc: datetime
    end_utc: datetime
    slot_minutes: int
    buffer_minutes: int
    note: Optional[str] = None


@tool("add_availability", args_schema=ToolAddAvailabilityIn, return_direct=False)
def add_availability_tool(
    start_local: str,
    end_local: str,
    slot_minutes: int,
    buffer_minutes: int,
    note: Optional[str],
    confirm_if_conflicts: bool,
    config: RunnableConfig,
) -> ToolAddAvailabilityOut:
    """
    Create a one-off availability (special opening) for the owner.

    Behavior:
      - Parses owner-local inputs to UTC.
      - Blocks hard if overlapping existing openings (weekly or one-off).
      - If overlapping Time Off or active Appointments, raises a CONFIRM_REQUIRED payload unless
        confirm_if_conflicts=True (in which case it proceeds).

    Args:
        start_local, end_local: Owner-local times (ISO-ish).
        slot_minutes, buffer_minutes: Slot settings.
        note: Optional note.
        confirm_if_conflicts: Set True to proceed despite time off/appointment conflicts.
        config: Must include configurable.user_id (or owner_id).

    Returns:
        ToolAddAvailabilityOut (merged if neighboring openings touch/overlap).

    Raises:
        ToolException for owner missing, invalid times, conflicts, or other errors.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")
            owner_tz = owner.timezone or "America/Toronto"

            s_loc = _parse_owner_local_dt(start_local, owner_tz)
            e_loc = _parse_owner_local_dt(end_local, owner_tz)
            if e_loc <= s_loc:
                raise ToolException("end_local must be after start_local")
            s_utc, e_utc = _to_utc(s_loc), _to_utc(e_loc)

            conflicts = _list_conflicts(db, owner, s_utc, e_utc, owner_tz)

            # A) Duplicate/overlapping availability (block hard)
            if conflicts["openings"]:
                preview = "; ".join(
                    _fmt_local_range(o["start_utc"], o["end_utc"], owner_tz)
                    for o in conflicts["openings"][:5]
                )
                more = (
                    f" (+{len(conflicts['openings']) - 5} more)"
                    if len(conflicts["openings"]) > 5
                    else ""
                )
                raise ToolException(
                    "Already available during the requested time. Existing availability overlaps: "
                    f"{preview}{more}. Edit/delete the existing opening(s) instead of adding a duplicate."
                )

            # B) Time Off / Appointment conflicts → require explicit confirm
            toff = conflicts["timeoffs"]
            appts = conflicts["appointments"]
            if (toff or appts) and not confirm_if_conflicts:
                payload = {
                    "human": "Requested availability conflicts with existing items. Reply 'confirm' to proceed anyway, or adjust the time.",
                    "pending": {
                        "tool": "add_availability",
                        "args": {
                            "start_local": start_local,
                            "end_local": end_local,
                            "slot_minutes": int(slot_minutes),
                            "buffer_minutes": int(buffer_minutes),
                            "note": note,
                            "confirm_if_conflicts": True,
                        },
                    },
                    "conflicts": {
                        "time_off": [
                            _fmt_local_range(t.start_utc, t.end_utc, owner_tz)
                            for t in toff[:5]
                        ],
                        "appointments": [
                            _fmt_local_range(a.start_utc, a.end_utc, owner_tz)
                            for a in appts[:5]
                        ],
                    },
                }
                raise ToolException("CONFIRM_REQUIRED:" + json.dumps(payload))

            # C) Create the opening (re-use add_special_opening_tool)
            out = add_special_opening_tool(
                start_local=start_local,
                end_local=end_local,
                slot_minutes=slot_minutes,
                buffer_minutes=buffer_minutes,
                note=note,
                config=config,
            )

            # D) Optional merge reflection
            try:
                row = (
                    db.query(SpecialOpening)
                    .filter(
                        SpecialOpening.owner_id == owner.id, SpecialOpening.id == out.id
                    )
                    .first()
                )
                if row:
                    row = _merge_overlapping_or_touching_openings(db, owner, row)
                    return ToolAddAvailabilityOut(
                        id=str(row.id),
                        start_utc=row.start_utc,
                        end_utc=row.end_utc,
                        slot_minutes=row.slot_minutes,
                        buffer_minutes=row.buffer_minutes,
                        note=row.note,
                    )
            except Exception:
                pass

            return ToolAddAvailabilityOut(**out.model_dump())
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"add_availability failed: {e}")


class ToolListWeeklyRulesIn(BaseModel):
    day_of_week: Optional[int] = Field(
        None, ge=0, le=6, description="0=Mon..6=Sun; omit for all"
    )


class ToolWeeklyRuleOut(BaseModel):
    id: str
    day_of_week: int
    start_minute: int
    end_minute: int
    slot_minutes: int
    buffer_minutes: int
    note: Optional[str] = None
    start_hhmm: str
    end_hhmm: str


class ToolListWeeklyRulesOut(BaseModel):
    rules: List[Dict[str, Any]]


class ToolCreateWeeklyRuleIn(BaseModel):
    weekday: int = Field(..., ge=0, le=6)
    start_hhmm: str = Field(..., description="e.g. '09:00'")
    end_hhmm: str = Field(..., description="'12:00'")
    slot_minutes: int = Field(..., ge=5, le=360)
    buffer_minutes: int = Field(0, ge=0)
    note: Optional[str] = None


class ToolUpdateWeeklyRuleIn(BaseModel):
    rule_id: Optional[str] = Field(None, description="Explicit rule id to update")
    weekday: Optional[int] = Field(None, ge=0, le=6, description="0=Mon..6=Sun")
    start_local: Optional[str] = Field(None, description="'HH:MM' 24h")
    end_local: Optional[str] = Field(None, description="'HH:MM' 24h")
    slot_minutes: Optional[int] = Field(None, ge=5, le=360)
    buffer_minutes: Optional[int] = Field(None, ge=0)
    anchor_day: Optional[str] = Field(
        default=None,
        description="'today'/'tomorrow'/'YYYY-MM-DD'/'next friday' to validate conflicts for the nearest occurrence",
    )


class ToolDeleteWeeklyRuleIn(BaseModel):
    rule_id: str


class ToolDeleteWeeklyRuleOut(BaseModel):
    deleted: bool
    rule_id: str


def _parse_hhmm_to_minutes(hhmm: str) -> int:
    s = (hhmm or "").strip()
    if not re.match(r"^\d{1,2}:\d{2}$", s):
        raise RuntimeError(f"Invalid HH:MM: {hhmm!r}")
    h, m = s.split(":")
    h, m = int(h), int(m)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise RuntimeError(f"Invalid clock time: {hhmm!r}")
    return h * 60 + m


def _rule_to_out(r: AvailabilityRule) -> ToolWeeklyRuleOut:
    return ToolWeeklyRuleOut(
        id=str(r.id),
        day_of_week=int(getattr(r, "weekday")),
        start_minute=r.start_local.hour * 60 + r.start_local.minute,
        end_minute=r.end_local.hour * 60 + r.end_local.minute,
        slot_minutes=int(getattr(r, "slot_minutes", 30) or 30),
        buffer_minutes=int(getattr(r, "buffer_minutes", 0) or 0),
        note=getattr(r, "note", None),
        start_hhmm=r.start_local.strftime("%I:%M %p").lstrip("0"),
        end_hhmm=r.end_local.strftime("%I:%M %p").lstrip("0"),
    )


@tool("list_weekly_rules", return_direct=False)
def list_weekly_rules_tool(config: RunnableConfig) -> ToolListWeeklyRulesOut:
    """
    List the owner's weekly availability rules.

    This **read-only** tool returns every 'AvailabilityRule' that defines
    recurring availability for the authenticated owner (resolved from the tool
    run 'config'). Rules are ordered by weekday (Mon=0..Sun=6) and
    'start_local'.

    Parameters
    ----------
    config : RunnableConfig
        Must include 'configurable.user_id' (or 'owner_id'). The value is used
        to scope the query to a single owner.

    Returns
    -------
    ToolListWeeklyRulesOut
        An object with:
          - 'rules' (list[dict]): Each item has:
              * 'id' (str): Rule id
              * 'weekday' (int): 0=Mon .. 6=Sun
              * 'start_local' (str): "HH:MM" in the owner's local time
              * 'end_local' (str): "HH:MM" in the owner's local time
              * 'slot_minutes' (int)
              * 'buffer_minutes' (int)
              * 'active' (bool)

    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        rules = (
            db.query(AvailabilityRule)
            .filter(AvailabilityRule.owner_id == owner_id)
            .order_by(
                AvailabilityRule.weekday.asc(), AvailabilityRule.start_local.asc()
            )
            .all()
        )
        out = [
            {
                "id": str(r.id),
                "weekday": int(r.weekday),
                "start_local": r.start_local.strftime("%H:%M"),
                "end_local": r.end_local.strftime("%H:%M"),
                "slot_minutes": int(getattr(r, "slot_minutes", 0) or 0),
                "buffer_minutes": int(getattr(r, "buffer_minutes", 0) or 0),
                "active": bool(getattr(r, "active", True)),
            }
            for r in rules
        ]
        return ToolListWeeklyRulesOut(rules=out)


@tool("create_weekly_rule", args_schema=ToolCreateWeeklyRuleIn, return_direct=False)
def create_weekly_rule_tool(
    weekday: int,
    start_hhmm: str,
    end_hhmm: str,
    slot_minutes: int,
    buffer_minutes: int = 0,
    note: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolWeeklyRuleOut:
    """Create a weekly availability rule for a given weekday and time window.

    If an identical rule already exists, returns it instead of creating a
    duplicate. Slot and buffer minutes are applied where supported by the
    schema.

    Args:
        weekday: 0=Mon .. 6=Sun.
        start_hhmm: Start time in HH:MM.
        end_hhmm: End time in HH:MM.
        slot_minutes: Slot length minutes.
        buffer_minutes: Buffer minutes between slots.
        note: Optional note for deployments that include the field.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolWeeklyRuleOut' representing the created or existing rule.

    Raises:
        RuntimeError: If the time window is invalid (end before start).
    """
    owner_id = _owner_id_from_config(config)

    start_min = _parse_hhmm_to_minutes(start_hhmm)
    end_min = _parse_hhmm_to_minutes(end_hhmm)
    if end_min <= start_min:
        raise RuntimeError("end must be after start")

    start_t = _time(start_min // 60, start_min % 60)
    end_t = _time(end_min // 60, end_min % 60)

    with SessionLocal() as db:
        existing = (
            db.query(AvailabilityRule)
            .filter(
                AvailabilityRule.owner_id == owner_id,
                AvailabilityRule.weekday == int(weekday),
                AvailabilityRule.start_local == start_t,
                AvailabilityRule.end_local == end_t,
            )
            .first()
        )
        if existing:
            changed = False
            if hasattr(existing, "slot_minutes") and existing.slot_minutes != int(
                slot_minutes
            ):
                existing.slot_minutes = int(slot_minutes)
                changed = True
            if hasattr(existing, "buffer_minutes") and existing.buffer_minutes != int(
                buffer_minutes
            ):
                existing.buffer_minutes = int(buffer_minutes)
                changed = True
            if note is not None and hasattr(existing, "note") and existing.note != note:
                existing.note = note
                changed = True
            if changed:
                db.commit()
                db.refresh(existing)
            return _rule_to_out(existing)

        kwargs = dict(
            owner_id=str(owner_id),
            weekday=int(weekday),
            start_local=start_t,
            end_local=end_t,
        )
        if hasattr(AvailabilityRule, "slot_minutes"):
            kwargs["slot_minutes"] = int(slot_minutes)
        if hasattr(AvailabilityRule, "buffer_minutes"):
            kwargs["buffer_minutes"] = int(buffer_minutes)
        if note is not None and hasattr(AvailabilityRule, "note"):
            kwargs["note"] = note

        r = AvailabilityRule(**kwargs)
        if hasattr(r, "note") and note is not None:
            r.note = note

        db.add(r)
        db.commit()
        db.refresh(r)
        return _rule_to_out(r)


@tool("delete_weekly_rule", args_schema=ToolDeleteWeeklyRuleIn, return_direct=False)
def delete_weekly_rule_tool(
    rule_id: str, config: RunnableConfig
) -> ToolDeleteWeeklyRuleOut:
    """Delete a weekly availability rule by id.

    Args:
        rule_id: Rule identifier.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolDeleteWeeklyRuleOut' with deletion status.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        r: AvailabilityRule = (
            db.query(AvailabilityRule)
            .filter(
                AvailabilityRule.id == rule_id, AvailabilityRule.owner_id == owner_id
            )
            .first()
        )
        if not r:
            return ToolDeleteWeeklyRuleOut(deleted=True, rule_id=rule_id)
        db.delete(r)
        db.commit()
        return ToolDeleteWeeklyRuleOut(deleted=True, rule_id=str(rule_id))


# -----------------------------------------------------------------------------
# Recurring openings: create weekly pattern as individual special openings
# -----------------------------------------------------------------------------


class ToolCreateRecurringOpeningsIn(BaseModel):
    weekday: int = Field(..., ge=0, le=6, description="0=Mon..6=Sun")
    start_hhmm: str = Field(..., description="e.g. '09:00'")
    end_hhmm: str = Field(..., description="e.g. '17:00'")
    slot_minutes: int = Field(..., ge=5, le=360)
    buffer_minutes: int = Field(0, ge=0)
    weeks: int = Field(8, ge=1, le=52)
    start_date: Optional[str] = Field(
        None,
        description="Owner-local 'YYYY-MM-DD' to anchor the first occurrence (default=today)",
    )
    note: Optional[str] = None


class ToolCreateRecurringOpeningsOut(BaseModel):
    created: List[Dict]


@tool(
    "create_recurring_openings",
    args_schema=ToolCreateRecurringOpeningsIn,
    return_direct=False,
)
def create_recurring_openings_tool(
    weekday: int,
    start_hhmm: str,
    end_hhmm: str,
    slot_minutes: int,
    buffer_minutes: int = 0,
    weeks: int = 8,
    start_date: Optional[str] = None,
    note: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolCreateRecurringOpeningsOut:
    """Create a weekly pattern of one-off openings for N weeks.

    Each occurrence is a standalone 'SpecialOpening' row so it can be edited
    independently later.

    Args:
        weekday: 0=Mon .. 6=Sun for the weekly occurrence.
        start_hhmm: Start time in HH:MM.
        end_hhmm: End time in HH:MM.
        slot_minutes: Slot length in minutes.
        buffer_minutes: Buffer minutes between slots.
        weeks: Number of weeks to create (default 8).
        start_date: Optional 'YYYY-MM-DD' anchor for the first occurrence.
        note: Optional note to include on openings.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolCreateRecurringOpeningsOut' with created opening details.

    Raises:
        ToolException: For invalid inputs or missing owner.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal.begin() as db:
        owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")
        tz = ZoneInfo(owner.timezone)

        def _to_time(hhmm: str) -> _time:
            s = (hhmm or "").strip()
            try:
                h, m = [int(x) for x in s.split(":")[:2]]
            except Exception:
                raise ToolException("Invalid time; expected HH:MM")
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ToolException("Invalid time; expected HH:MM")
            return _time(h, m)

        st = _to_time(start_hhmm)
        et = _to_time(end_hhmm)
        if et <= st:
            raise ToolException("end_hhmm must be after start_hhmm")
        if weeks <= 0:
            raise ToolException("weeks must be >= 1")

        if start_date:
            try:
                y, m, d = [int(x) for x in start_date.split("-")]
                base = _date(y, m, d)
            except Exception:
                raise ToolException("Invalid start_date; expected YYYY-MM-DD")
        else:
            base = datetime.now(tz).date()

        delta = (int(weekday) - base.weekday()) % 7
        first = base if delta == 0 else (base + timedelta(days=delta))

        created: list[Dict] = []

        for i in range(int(weeks)):
            cur = first + timedelta(days=7 * i)
            s_loc = datetime(
                cur.year, cur.month, cur.day, st.hour, st.minute, tzinfo=tz
            )
            e_loc = datetime(
                cur.year, cur.month, cur.day, et.hour, et.minute, tzinfo=tz
            )

            out = add_special_opening_tool(
                start_local=s_loc.isoformat(timespec="minutes"),
                end_local=e_loc.isoformat(timespec="minutes"),
                slot_minutes=int(slot_minutes),
                buffer_minutes=int(buffer_minutes or 0),
                note=note or "Weekly opening",
                config=config,
            )
            created.append(
                {
                    "opening_id": out.id,
                    "start_local": _iso_local(out.start_utc, owner.timezone),
                    "end_local": _iso_local(out.end_utc, owner.timezone),
                    "slot_minutes": out.slot_minutes,
                    "buffer_minutes": out.buffer_minutes,
                    "note": out.note,
                }
            )

        return ToolCreateRecurringOpeningsOut(created=created)


@tool("truncate_after", args_schema=ToolTruncateAfterIn, return_direct=False)
def truncate_availability_after_tool(
    local_hhmm: str, day: Optional[str], scope: Optional[str], config: RunnableConfig
) -> ToolTruncateAfterOut:
    """Make the owner unavailable after a local time on a given day.

    If 'scope' is omitted and a weekly rule is affected, raises
    'ToolException' with a 'CHOICE_REQUIRED:{...}' payload describing two
    options:
    - scope='day' → add a TimeOff [cutoff, end-of-day] and trim specials
    - scope='weekly' → update the weekly rule's 'end_local' to the cutoff

    Args:
        local_hhmm: Local cutoff time like "15:40" or "3:30pm".
        day: Anchor day ("today" | "tomorrow" | "YYYY-MM-DD" | textual weekday).
        scope: "day" to add time off; "weekly" to modify the weekly rule.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolTruncateAfterOut' indicating changes or any required
        cancellations when appointments exist after the cutoff.

    Raises:
        ToolException: For owner issues, bad time format, ambiguity requiring
        user choice, or when writes fail.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")

        tz = ZoneInfo(owner.timezone or "America/Toronto")
        anchor = day or "today"
        the_day = _parse_owner_day(anchor, owner.timezone)
        try:
            parts = local_hhmm.strip().lower().replace(" ", "")
            if parts.endswith(("am", "pm")):
                ap = parts[-2:]
                parts = parts[:-2]
                hh, mm = (parts.split(":") + ["0"])[:2]
                hh, mm = int(hh), int(mm)
                if ap == "pm" and hh != 12:
                    hh += 12
                if ap == "am" and hh == 12:
                    hh = 0
            else:
                hh, mm = map(int, parts.split(":"))
        except Exception:
            raise ToolException("local_hhmm must be 'HH:MM' or like '3pm'/'3:30pm'")

        cutoff_local = datetime(
            the_day.year, the_day.month, the_day.day, hh, mm, tzinfo=tz
        )
        day_start_local = datetime(
            the_day.year, the_day.month, the_day.day, 0, 0, tzinfo=tz
        )
        day_end_local = datetime(
            the_day.year, the_day.month, the_day.day, 23, 59, tzinfo=tz
        )

        from agent.tools import _to_utc as _to_utc_inner

        u_start = _to_utc_inner(day_start_local)
        u_end = _to_utc_inner(day_end_local)
        u_cutoff = _to_utc_inner(cutoff_local)
        weekly_rules = (
            db.query(AvailabilityRule)
            .filter(
                AvailabilityRule.owner_id == owner.id,
                AvailabilityRule.weekday == the_day.weekday(),
            )
            .all()
        )
        affects_weekly = any(
            datetime(
                the_day.year,
                the_day.month,
                the_day.day,
                r.start_local.hour,
                r.start_local.minute,
                tzinfo=tz,
            )
            < cutoff_local
            < datetime(
                the_day.year,
                the_day.month,
                the_day.day,
                r.end_local.hour,
                r.end_local.minute,
                tzinfo=tz,
            )
            for r in weekly_rules
        )

        if affects_weekly and not scope:
            payload = {
                "human": (
                    f"Apply cutoff at {cutoff_local.strftime('%I:%M %p').lstrip('0')} only on {the_day.isoformat()} "
                    f"or every {the_day.strftime('%A')}?"
                ),
                "choice": {
                    "day": {
                        "tool": "truncate_after",
                        "args": {
                            "local_hhmm": local_hhmm,
                            "day": anchor,
                            "scope": "day",
                        },
                    },
                    "weekly": {
                        "tool": "update_weekly_rule",
                        "args": {
                            "weekday": int(the_day.weekday()),
                            "end_local": cutoff_local.strftime("%H:%M"),
                            "anchor_day": the_day.isoformat(),
                        },
                    },
                },
            }
            raise ToolException("CHOICE_REQUIRED:" + json.dumps(payload))
        scol, ecol = _appt_cols()
        appts_after = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.status.in_(ACTIVE_APPT_STATUSES),
                getattr(Appointment, scol) < u_end,
                getattr(Appointment, ecol) > u_cutoff,
            )
            .order_by(getattr(Appointment, scol).asc())
            .all()
        )
        if appts_after:
            blocked = [
                {
                    "id": str(a.id),
                    "start_local": _iso_local(a.start_utc, owner.timezone),
                    "end_local": _iso_local(a.end_utc, owner.timezone),
                    "status": a.status,
                }
                for a in appts_after
            ]
            return ToolTruncateAfterOut(
                ok=False,
                updated=[],
                deleted=[],
                cutoff_local=cutoff_local.isoformat(timespec="minutes"),
                requires_cancellation=True,
                blocked_appointments=blocked,
            )
        if scope == "day":
            row_to = TimeOff(
                owner_id=owner.id,
                start_utc=_to_utc_inner(cutoff_local),
                end_utc=_to_utc_inner(day_end_local),
                note="Day override after cutoff",
            )
            db.add(row_to)

            rows: List[SpecialOpening] = (
                db.query(SpecialOpening)
                .filter(
                    SpecialOpening.owner_id == owner.id,
                    SpecialOpening.start_utc < _to_utc_inner(day_end_local),
                    SpecialOpening.end_utc > _to_utc_inner(day_start_local),
                )
                .order_by(SpecialOpening.start_utc.asc())
                .all()
            )
            updated: List[str] = []
            deleted: List[str] = []
            for r in rows:
                s_loc = r.start_utc.astimezone(tz)
                e_loc = r.end_utc.astimezone(tz)
                if s_loc >= cutoff_local:
                    db.delete(r)
                    deleted.append(str(r.id))
                elif s_loc < cutoff_local < e_loc:
                    r.end_utc = _to_utc_inner(cutoff_local)
                    db.add(r)
                    updated.append(str(r.id))
            db.commit()
            return ToolTruncateAfterOut(
                ok=True,
                updated=updated,
                deleted=deleted,
                cutoff_local=cutoff_local.isoformat(timespec="minutes"),
            )
        if scope == "weekly":
            out = update_weekly_rule_tool(
                rule_id=None,
                weekday=int(the_day.weekday()),
                start_local=None,
                end_local=cutoff_local.strftime("%H:%M"),
                slot_minutes=None,
                buffer_minutes=None,
                anchor_day=the_day.isoformat(),
                config=config,
            )
            if out.requires_cancellation:
                return ToolTruncateAfterOut(
                    ok=False,
                    updated=[],
                    deleted=[],
                    cutoff_local=cutoff_local.isoformat(timespec="minutes"),
                    requires_cancellation=True,
                    blocked_appointments=out.blocked_appointments,
                )
            return ToolTruncateAfterOut(
                ok=True,
                updated=[],
                deleted=[],
                cutoff_local=cutoff_local.isoformat(timespec="minutes"),
            )
        rows: List[SpecialOpening] = (
            db.query(SpecialOpening)
            .filter(
                SpecialOpening.owner_id == owner.id,
                SpecialOpening.start_utc < u_end,
                SpecialOpening.end_utc > u_start,
            )
            .order_by(SpecialOpening.start_utc.asc())
            .all()
        )
        updated: List[str] = []
        deleted: List[str] = []
        for r in rows:
            s_loc = r.start_utc.astimezone(tz)
            e_loc = r.end_utc.astimezone(tz)
            if s_loc >= cutoff_local:
                db.delete(r)
                deleted.append(str(r.id))
            elif s_loc < cutoff_local < e_loc:
                r.end_utc = _to_utc_inner(cutoff_local)
                db.add(r)
                updated.append(str(r.id))
        db.commit()
        return ToolTruncateAfterOut(
            ok=True,
            updated=updated,
            deleted=deleted,
            cutoff_local=cutoff_local.isoformat(timespec="minutes"),
        )


class ToolUpdateWeeklyRuleOut(BaseModel):
    ok: bool
    rule: Optional[Dict[str, Any]] = None
    ambiguous: bool = False
    candidates: Optional[List[Dict[str, Any]]] = None
    requires_cancellation: bool = False
    blocked_appointments: List[Dict[str, Any]] = []


def _to_time(hhmm: str) -> _time:
    hhmm = hhmm.strip().lower()
    if hhmm.endswith("am") or hhmm.endswith("pm"):
        ap = hhmm[-2:]
        core = hhmm[:-2]
        hh, mm = (core.split(":") + ["0"])[:2]
        hh, mm = int(hh), int(mm)
        if ap == "pm" and hh != 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        return _time(hh, mm)
    h, m = (hhmm.split(":") + ["0"])[:2]
    return _time(int(h), int(m))


@tool("update_weekly_rule", args_schema=ToolUpdateWeeklyRuleIn, return_direct=False)
def update_weekly_rule_tool(
    rule_id: Optional[str],
    weekday: Optional[int],
    start_local: Optional[str],
    end_local: Optional[str],
    slot_minutes: Optional[int],
    buffer_minutes: Optional[int],
    anchor_day: Optional[str],
    config: RunnableConfig,
) -> ToolUpdateWeeklyRuleOut:
    """Update fields on a weekly availability rule.

    If multiple rules exist for the weekday and no 'rule_id' is provided,
    returns 'ambiguous=True' with candidates. If 'end_local' truncates the
    day for a concrete 'anchor_day' and booked appointments exist after the
    cutoff, returns 'requires_cancellation=True' with blocked appointments.

    Args:
        rule_id: Target rule id; optional if 'weekday' scopes uniquely.
        weekday: 0=Mon .. 6=Sun used to locate a rule when 'rule_id' is not given.
        start_local: New start HH:MM (optional).
        end_local: New end HH:MM (optional).
        slot_minutes: Slot length override (optional).
        buffer_minutes: Buffer minutes override (optional).
        anchor_day: Owner-local 'YYYY-MM-DD' used when evaluating conflicts for
            truncation.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolUpdateWeeklyRuleOut' describing the updated rule, or an
        ambiguous/blocked response as described above.

    Raises:
        ToolException: For missing owner, not found rule, or invalid inputs.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        q = db.query(AvailabilityRule).filter(AvailabilityRule.owner_id == owner_id)
        target: Optional[AvailabilityRule] = None

        if rule_id:
            target = q.filter(AvailabilityRule.id == rule_id).first()
            if not target:
                raise ToolException("Weekly rule not found")
        else:
            if weekday is None:
                raise ToolException("Provide rule_id or weekday")
            cands = (
                q.filter(AvailabilityRule.weekday == int(weekday))
                .order_by(AvailabilityRule.start_local.asc())
                .all()
            )
            if not cands:
                raise ToolException("Weekly rule not found")
            if len(cands) > 1:
                return ToolUpdateWeeklyRuleOut(
                    ok=False,
                    ambiguous=True,
                    candidates=[
                        {
                            "id": str(r.id),
                            "weekday": int(r.weekday),
                            "start_local": r.start_local.strftime("%H:%M"),
                            "end_local": r.end_local.strftime("%H:%M"),
                            "slot_minutes": int(getattr(r, "slot_minutes", 0) or 0),
                            "buffer_minutes": int(getattr(r, "buffer_minutes", 0) or 0),
                        }
                        for r in cands
                    ],
                )
            target = cands[0]

        owner = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")
        tz = ZoneInfo(owner.timezone or "America/Toronto")

        new_start = target.start_local
        new_end = target.end_local
        if start_local:
            new_start = _to_time(start_local)
        if end_local:
            new_end = _to_time(end_local)
        if new_end <= new_start:
            raise ToolException("end_local must be after start_local")

        if anchor_day and end_local:
            day = _parse_owner_day(anchor_day, owner.timezone)
            cutoff_local_dt = datetime(
                day.year, day.month, day.day, new_end.hour, new_end.minute, tzinfo=tz
            )
            from agent.tools import _to_utc as _to_utc_inner

            u_end = _to_utc_inner(
                datetime(day.year, day.month, day.day, 23, 59, tzinfo=tz)
            )
            u_cut = _to_utc_inner(cutoff_local_dt)

            scol, ecol = _appt_cols()
            appts_after = (
                db.query(Appointment)
                .filter(
                    Appointment.owner_id == owner.id,
                    Appointment.status.in_(ACTIVE_APPT_STATUSES),
                    getattr(Appointment, scol) < u_end,
                    getattr(Appointment, ecol) > u_cut,
                )
                .order_by(getattr(Appointment, scol).asc())
                .all()
            )
            if appts_after:
                blocked = [
                    {
                        "id": str(a.id),
                        "start_local": _iso_local(a.start_utc, owner.timezone),
                        "end_local": _iso_local(a.end_utc, owner.timezone),
                        "status": a.status,
                    }
                    for a in appts_after
                ]
                return ToolUpdateWeeklyRuleOut(
                    ok=False, requires_cancellation=True, blocked_appointments=blocked
                )

        target.start_local = new_start
        target.end_local = new_end
        if slot_minutes is not None:
            target.slot_minutes = int(slot_minutes)
        if buffer_minutes is not None:
            target.buffer_minutes = int(buffer_minutes)
        db.commit()
        db.refresh(target)
        return ToolUpdateWeeklyRuleOut(
            ok=True,
            rule={
                "id": str(target.id),
                "weekday": int(target.weekday),
                "start_local": target.start_local.strftime("%H:%M"),
                "end_local": target.end_local.strftime("%H:%M"),
                "slot_minutes": int(getattr(target, "slot_minutes", 0) or 0),
                "buffer_minutes": int(getattr(target, "buffer_minutes", 0) or 0),
            },
        )
