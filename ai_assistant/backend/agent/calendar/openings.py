from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta, time as _time, date as _date
from zoneinfo import ZoneInfo
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, ToolException
from app.db import SessionLocal
from app.models import User, SpecialOpening, TimeOff, Appointment
from sqlalchemy.exc import IntegrityError

from services.services_scheduling import (
    carve_opening_through_timeoff,
    merge_or_get_special_opening,
)
from agent.schemas import (
    ToolListOpeningsIn,
    ToolListOpeningsOut,
    ToolUpdateOpeningIn,
    ToolDeleteOpeningIn,
    ToolDeleteOpeningOut,
    ToolCreateRecurringOpeningsIn,
    ToolCreateRecurringOpeningsOut,
    ToolTruncateAfterIn,
    ToolTruncateAfterOut,
    ToolAddSpecialOpeningIn,
    ToolAddSpecialOpeningOut,
)
from agent.constants import ACTIVE_APPT_STATUSES
from agent.tools import _parse_owner_day, _parse_owner_local_dt, _to_utc
from agent.tool_ctx import owner_id_var
import os
import sqlalchemy as sa


def _owner_id_from_config(config: RunnableConfig) -> str:
    """Resolve the owner id from config, contextvar, or environment.

    Args:
        config: Runnable configuration with optional 'configurable' mapping.

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


def _iso_local(dt: datetime, tz_name: str) -> str:
    """Format a datetime as ISO-8601 in a timezone (minute precision)."""
    try:
        local_dt = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        local_dt = dt
    return local_dt.isoformat(timespec="minutes")


def _fmt_local_range(s_utc: datetime, e_utc: datetime, tz_name: str) -> str:
    """Format a UTC start/end pair as a single owner-local string."""
    return f"{_iso_local(s_utc, tz_name)} → {_iso_local(e_utc, tz_name)}"


def _appt_cols():
    """Return Appointment UTC start/end column names, supporting legacy schemas."""
    cols = {c.name for c in Appointment.__table__.columns}
    start_col = "start_utc" if "start_utc" in cols else "start_time_utc"
    end_col = "end_utc" if "end_utc" in cols else "end_time_utc"
    return start_col, end_col


def _resolve_opening_row(
    db, owner: User, opening_identifier: Union[int, str], day: Optional[str] = None
) -> Optional[SpecialOpening]:
    """Resolve a SpecialOpening by id or display index for a given day.

    Args:
        db: Database session.
        owner: Owner user row.
        opening_identifier: DB id/UUID string or display id 'open-<n>'.
        day: Owner-local anchor day used for display id resolution.

    Returns:
        Matching SpecialOpening row or None.
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
    # 2) If it's 'open-<n>', resolve against openings for the day
    import re

    m = re.match(r"^open-(\d+)$", s)
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
    # 3) Fallback numeric
    try:
        int(s)
    except Exception:
        return None
    return (
        db.query(SpecialOpening)
        .filter(SpecialOpening.owner_id == owner.id, SpecialOpening.id == s)
        .first()
    )


def _merge_overlapping_or_touching_openings(
    db, owner: User, new_row: SpecialOpening
) -> SpecialOpening:
    """Merge a new opening into existing overlapping/touching ones.

    Returns the surviving merged row.
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


@tool("add_special_opening", args_schema=ToolAddSpecialOpeningIn, return_direct=False)
def add_special_opening_tool(
    start_local: str,
    end_local: str,
    slot_minutes: int,
    buffer_minutes: int,
    note: Optional[str],
    config: RunnableConfig,
) -> ToolAddSpecialOpeningOut:
    """Create or merge a one‑off opening for the owner.

    Args:
        start_local: Owner‑local start time (YYYY‑MM‑DDTHH:MM).
        end_local: Owner‑local end time (YYYY‑MM‑DDTHH:MM).
        slot_minutes: Slot length in minutes.
        buffer_minutes: Buffer length in minutes.
        note: Optional note.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolAddSpecialOpeningOut' with stored UTC times and metadata.

    Raises:
        ToolException: For missing owner, invalid bounds, or internal errors.
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

            carve_opening_through_timeoff(db, owner.id, s_utc, e_utc)

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
    """List one‑off openings overlapping an owner‑local day.

    Args:
        day: Owner‑local day ("YYYY‑MM‑DD", "today", "tomorrow").
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolListOpeningsOut' with 'openings' items including local/UTC times.

    Raises:
        ToolException: For missing owner or unexpected errors.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            d = (
                _parse_owner_local_dt(day, owner.timezone).date()
                if day not in ("today", "tomorrow")
                else _parse_owner_local_dt(
                    (
                        _date.today().isoformat()
                        if day == "today"
                        else (
                            datetime.now(ZoneInfo(owner.timezone)).date()
                            + timedelta(days=1)
                        ).isoformat()
                    ),
                    owner.timezone,
                ).date()
            )
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
) -> Dict[str, Any]:
    """Update a one‑off opening and re‑merge if needed.

    Args:
        opening_id: DB id or display id 'open-<n>'; use 'day' to resolve display ids.
        start_local: Optional new owner‑local start (YYYY‑MM‑DDTHH:MM).
        end_local: Optional new owner‑local end (YYYY‑MM‑DDTHH:MM).
        slot_minutes: Optional slot length override.
        buffer_minutes: Optional buffer minutes override.
        note: Optional note update.
        day: Owner‑local day used to resolve display ids.
        config: Runnable configuration providing the owner id.

    Returns:
        Dict with updated fields: '{ opening_id, start_local, end_local, slot_minutes, buffer_minutes, note }'.

    Raises:
        ToolException: If owner cannot be resolved, opening is not found,
        bounds invalid, or on unexpected errors.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal.begin() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            row: Optional[SpecialOpening] = _resolve_opening_row(
                db, owner, opening_id, day=day
            )
            if not row:
                raise ToolException("Opening not found")

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
    """Delete a one-off opening by id or display index.

    Arguments
    - 'opening_id': Database id/UUID, or display id 'open-<n>' (0-based index
      for the given 'day' ordered by start time).
    - 'day': Required when using 'open-<n>' to specify which day to index.
    - 'config': LangChain Runnable config carrying the owner identity.

    Behavior
    - If the opening cannot be resolved, the operation is treated as idempotent
      and returns 'deleted=True'.

    Returns
    - 'ToolDeleteOpeningOut': '{ deleted, opening_id }'.

    Errors
    - 'ToolException("Owner not found")' if owner resolution fails.
    - 'ToolException("delete_opening failed: ...")' for unexpected errors.
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


@tool(
    "create_recurring_openings",
    args_schema=ToolCreateRecurringOpeningsIn,
    return_direct=False,
)
def create_recurring_openings_tool(
    weekday: int,
    start_hhmm: str,
    end_hhmm: str,
    weeks: int,
    start_date: Optional[str],
    slot_minutes: int,
    buffer_minutes: Optional[int],
    note: Optional[str],
    config: RunnableConfig,
) -> ToolCreateRecurringOpeningsOut:
    """Create a series of one-off openings on a weekly cadence.

    Arguments
    - 'weekday': 0=Monday … 6=Sunday (owner-local).
    - 'start_hhmm' / 'end_hhmm': HH:MM in owner-local time.
    - 'weeks': Number of weekly occurrences to create (>=1).
    - 'start_date': Optional 'YYYY-MM-DD' anchor for the first occurrence; when
      omitted, uses today in the owner’s timezone.
    - 'slot_minutes' / 'buffer_minutes': Opening parameters applied to each.
    - 'note': Optional note for each created opening.
    - 'config': LangChain Runnable config carrying the owner identity.

    Behavior
    - Creates each occurrence as a special opening via 'add_special_opening_tool'.
    - Adjacent/touching openings merge according to the usual rules.

    Returns
    - 'ToolCreateRecurringOpeningsOut': 'created' list with items containing
      '{ opening_id, start_local, end_local, slot_minutes, buffer_minutes, note }'.

    Errors
    - 'ToolException("Owner not found")' if owner resolution fails.
    - 'ToolException("Invalid time; expected HH:MM")', '"weeks must be >= 1"',
      or '"Invalid start_date; expected YYYY-MM-DD"' for invalid inputs.
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
    """Truncate availability after a local time on a day or weekly rule.

    Parses 'local_hhmm' in owner-local time (e.g., '15:30', '3pm'). If the
    cutoff intersects an existing opening, 'scope' of 'day' trims only that
    day via a time-off override; 'weekly' updates the weekly rule. When
    affected appointments exist after the cutoff, returns a non-OK result
    with 'requires_cancellation=True' and the blocking appointments listed.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
        if not owner:
            raise ToolException("Owner not found")

        tz = ZoneInfo(owner.timezone or "America/Toronto")
        anchor = day or "today"
        the_day = (
            _parse_owner_local_dt(anchor, owner.timezone).date()
            if anchor not in ("today", "tomorrow")
            else (
                _date.today()
                if anchor == "today"
                else (_date.today() + timedelta(days=1))
            )
        )

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
                cutoff_local = datetime(
                    the_day.year, the_day.month, the_day.day, hh, mm, tzinfo=tz
                )
            else:
                hh, mm = (parts.split(":") + ["0"])[:2]
                cutoff_local = datetime(
                    the_day.year,
                    the_day.month,
                    the_day.day,
                    int(hh),
                    int(mm),
                    tzinfo=tz,
                )
        except Exception:
            raise ToolException("Invalid time; expected HH:MM or 3pm/3:30pm")

        def _to_utc_inner(dt: datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            return dt.astimezone(ZoneInfo("UTC"))

        day_start_local = datetime(
            the_day.year, the_day.month, the_day.day, 0, 0, tzinfo=tz
        )
        day_end_local = datetime(
            the_day.year, the_day.month, the_day.day, 23, 59, tzinfo=tz
        )
        u_start = _to_utc_inner(day_start_local)
        u_end = _to_utc_inner(day_end_local)

        specials = (
            db.query(SpecialOpening)
            .filter(
                SpecialOpening.owner_id == owner.id,
                SpecialOpening.start_utc < u_end,
                SpecialOpening.end_utc > u_start,
            )
            .order_by(SpecialOpening.start_utc.asc())
            .all()
        )

        for sp in specials:
            s_loc = sp.start_utc.astimezone(tz)
            e_loc = sp.end_utc.astimezone(tz)
            if s_loc < cutoff_local < e_loc:
                scope = scope or "ask"
                break
        else:
            scope = scope or "day"

        if scope == "ask":

            choice_payload = {
                "human": "Cutoff intersects existing availability. Choose: update just this day or weekly rule.",
                "pending": [
                    {
                        "tool": "truncate_after",
                        "args": {
                            "local_hhmm": local_hhmm,
                            "day": anchor,
                            "scope": "day",
                        },
                    },
                    {
                        "tool": "truncate_after",
                        "args": {
                            "local_hhmm": local_hhmm,
                            "day": anchor,
                            "scope": "weekly",
                        },
                    },
                ],
            }
            raise ToolException(
                "CHOICE_REQUIRED:" + __import__("json").dumps(choice_payload)
            )

        scol, ecol = _appt_cols()
        appts_after = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.status.in_(ACTIVE_APPT_STATUSES),
                getattr(Appointment, scol) < _to_utc_inner(day_end_local),
                getattr(Appointment, ecol) > _to_utc_inner(cutoff_local),
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
            from agent.calendar.weekly_rules import update_weekly_rule_tool

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
