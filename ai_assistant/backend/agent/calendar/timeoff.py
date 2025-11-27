from __future__ import annotations
from typing import Optional, List, Literal
from datetime import datetime, timedelta, timezone as _tz
from zoneinfo import ZoneInfo
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, ToolException

from app.db import SessionLocal
from app.models import User, TimeOff, Appointment
from agent.schemas import (
    ToolAddTimeOffOut,
    ToolListTimeOffIn,
    ToolListTimeOffOut,
    ToolDeleteTimeOffIn,
    ToolDeleteTimeOffOut,
    ToolAddTimeOffIn,
    ToolUpdateTimeOffIn,
    ToolUpdateTimeOffOut,
    ToolNextTimeOffOut,
)
from agent.constants import ACTIVE_APPT_STATUSES
from agent.tools import _parse_owner_day, _parse_owner_local_dt, _to_utc
from agent.calendar.openings import _owner_id_from_config, _iso_local, _appt_cols


def _owner_local_day_window(owner_tz: str, day: str) -> tuple[datetime, datetime]:
    """Compute the owner‑local time window for a given day.

    Args:
        owner_tz: IANA timezone (e.g., "America/Toronto").
        day: Owner‑local anchor ("YYYY-MM-DD" or natural phrase like "today").

    Returns:
        A tuple '(start_local, end_local)' of timezone‑aware datetimes in
        'owner_tz' covering 00:00 through 23:59 of the day.
    """
    d = _parse_owner_day(day, owner_tz)
    tz = ZoneInfo(owner_tz)
    start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
    end_local = datetime(d.year, d.month, d.day, 23, 59, tzinfo=tz)
    return start_local, end_local


@tool("add_time_off", args_schema=ToolAddTimeOffIn, return_direct=False)
def add_time_off_tool(
    start_local: str, end_local: str, note: Optional[str], config: RunnableConfig
) -> ToolAddTimeOffOut:
    """Create a time‑off block for the owner.

    Args:
        start_local: Owner‑local start time ("YYYY‑MM‑DDTHH:MM").
        end_local: Owner‑local end time ("YYYY‑MM‑DDTHH:MM").
        note: Optional free‑form note.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolAddTimeOffOut' with the created row id and UTC times.

    Raises:
        ToolException: If the owner is missing, bounds are invalid, or the
        requested window overlaps a booked appointment (payload prefixed with
        'APPT_CONFLICT:'). Unexpected errors are wrapped with a message.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            s_loc = _parse_owner_local_dt(start_local, owner.timezone)
            e_loc = _parse_owner_local_dt(end_local, owner.timezone)
            if e_loc <= s_loc:
                raise ToolException("end_local must be after start_local")

            s_utc, e_utc = _to_utc(s_loc), _to_utc(e_loc)

            scol, ecol = _appt_cols()
            conflicts = (
                db.query(Appointment)
                .filter(
                    Appointment.owner_id == owner.id,
                    getattr(Appointment, scol) < e_utc,
                    getattr(Appointment, ecol) > s_utc,
                    Appointment.status == "booked",
                )
                .order_by(getattr(Appointment, scol).asc())
                .all()
            )

            if conflicts:
                items = [
                    {
                        "id": str(c.id),
                        "start_local": _iso_local(c.start_utc, owner.timezone),
                        "end_local": _iso_local(c.end_utc, owner.timezone),
                    }
                    for c in conflicts
                ]
                payload = {
                    "human": "There’s an appointment during that time. You must cancel or move it before adding time off.",
                    "blocked_appointments": items,
                }
                raise ToolException(
                    "APPT_CONFLICT:" + __import__("json").dumps(payload)
                )

            row = TimeOff(owner_id=owner.id, start_utc=s_utc, end_utc=e_utc, note=note)
            db.add(row)
            db.commit()
            db.refresh(row)

            return ToolAddTimeOffOut(
                id=str(row.id), start_utc=s_utc, end_utc=e_utc, note=note
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"add_time_off failed: {e}")


@tool("update_time_off", args_schema=ToolUpdateTimeOffIn, return_direct=False)
def update_time_off_tool(
    timeoff_id: Optional[str] = None,
    day: Optional[str] = None,
    start_local: Optional[str] = None,
    end_local: Optional[str] = None,
    note: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolUpdateTimeOffOut:
    """Modify an existing time‑off block.

    Args:
        timeoff_id: Target row id. If omitted, resolves uniquely by 'day'.
        day: Owner‑local anchor day used to find a time off when 'timeoff_id'
            is not provided.
        start_local: New start time (full datetime or "HH:MM" in owner local).
        end_local: New end time (full datetime or "HH:MM" in owner local).
        note: Optional note (use None to clear).
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolUpdateTimeOffOut' with the updated row details.

    Raises:
        ToolException: For missing owner, not found/ambiguous rows, invalid
        bounds, or conflicts with active appointments ('APPT_CONFLICT:...').
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal.begin() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")
            tz = ZoneInfo(owner.timezone)
            row: Optional[TimeOff] = None
            if timeoff_id:
                row = (
                    db.query(TimeOff)
                    .filter(TimeOff.owner_id == owner.id, TimeOff.id == str(timeoff_id))
                    .first()
                )
                if not row:
                    raise ToolException("Time off not found")
            else:
                if not day:
                    raise ToolException(
                        "Provide 'timeoff_id' or 'day' to resolve the time off to modify"
                    )
                d = _parse_owner_day(day, owner.timezone)
                d0 = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
                d1 = d0 + timedelta(days=1)
                u0, u1 = _to_utc(d0), _to_utc(d1)
                rows = (
                    db.query(TimeOff)
                    .filter(
                        TimeOff.owner_id == owner.id,
                        TimeOff.start_utc < u1,
                        TimeOff.end_utc > u0,
                    )
                    .order_by(TimeOff.start_utc.asc())
                    .all()
                )
                if not rows:
                    raise ToolException("No time off found on that day")
                if len(rows) > 1 and (not start_local or not end_local):
                    raise ToolException(
                        "AMBIGUOUS_TIME_OFF:Multiple time off blocks that day. Specify the timeoff_id or provide start_local and end_local."
                    )
                row = rows[0]

            # Determine new local times
            def _parse_local(s: Optional[str], default_dt: datetime) -> datetime:
                if not s:
                    return default_dt
                if len(s) <= 5 and ":" in s:
                    hh, mm = s.split(":")
                    return default_dt.replace(
                        hour=int(hh), minute=int(mm), second=0, microsecond=0
                    )
                return _parse_owner_local_dt(s, owner.timezone)

            cur_s_loc = row.start_utc.astimezone(tz)
            cur_e_loc = row.end_utc.astimezone(tz)
            s_loc = _parse_local(start_local, cur_s_loc)
            e_loc = _parse_local(end_local, cur_e_loc)
            if e_loc <= s_loc:
                raise ToolException("end_local must be after start_local")

            s_utc, e_utc = _to_utc(s_loc), _to_utc(e_loc)

            # Block if overlapping active appointment
            scol, ecol = _appt_cols()
            conflicts = (
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
            if conflicts:
                items = [
                    {
                        "id": str(c.id),
                        "start_local": _iso_local(c.start_utc, owner.timezone),
                        "end_local": _iso_local(c.end_utc, owner.timezone),
                        "status": c.status,
                    }
                    for c in conflicts
                ]
                payload = {
                    "human": "There’s an appointment during that time. You must cancel or move it before updating time off.",
                    "blocked_appointments": items,
                }
                raise ToolException(
                    "APPT_CONFLICT:" + __import__("json").dumps(payload)
                )

            row.start_utc = s_utc
            row.end_utc = e_utc
            if note is not None:
                row.note = note
            db.commit()
            db.refresh(row)

            return ToolUpdateTimeOffOut(
                id=str(row.id),
                start_utc=row.start_utc,
                end_utc=row.end_utc,
                note=row.note,
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"update_time_off failed: {e}")


@tool("list_time_off", args_schema=ToolListTimeOffIn, return_direct=False)
def list_time_off_tool(day: str, config: RunnableConfig) -> ToolListTimeOffOut:
    """List time‑off blocks that overlap an owner‑local day.

    Args:
        day: Owner‑local anchor ("YYYY‑MM‑DD", "today", "tomorrow", etc.).
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolListTimeOffOut' with 'timeoff' entries including local times.

    Raises:
        ToolException: For missing owner or unexpected errors.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")
            start_local, end_local = _owner_local_day_window(owner.timezone, day)
            start_utc, end_utc = _to_utc(start_local), _to_utc(end_local)
            rows = (
                db.query(TimeOff)
                .filter(
                    TimeOff.owner_id == owner.id,
                    TimeOff.start_utc < end_utc,
                    TimeOff.end_utc > start_utc,
                )
                .order_by(TimeOff.start_utc.asc())
                .all()
            )
            out = [
                {
                    "id": r.id,
                    "start_local": _iso_local(r.start_utc, owner.timezone),
                    "end_local": _iso_local(r.end_utc, owner.timezone),
                    "note": r.note,
                }
                for r in rows
            ]
            return ToolListTimeOffOut(timeoff=out)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"list_time_off failed: {e}")


@tool("next_time_off", return_direct=False)
def next_time_off_tool(config: RunnableConfig) -> ToolNextTimeOffOut:
    """Return the owner's next upcoming time-off block (if any).

    Uses current UTC time to find the earliest block whose end is in the future.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")
            now_utc = datetime.now(_tz.utc)
            row = (
                db.query(TimeOff)
                .filter(TimeOff.owner_id == owner.id, TimeOff.end_utc > now_utc)
                .order_by(TimeOff.start_utc.asc())
                .first()
            )
            if not row:
                return ToolNextTimeOffOut(found=False)
            return ToolNextTimeOffOut(
                found=True,
                start_local=_iso_local(row.start_utc, owner.timezone),
                end_local=_iso_local(row.end_utc, owner.timezone),
                note=row.note,
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"next_time_off failed: {e}")


@tool("delete_time_off", args_schema=ToolDeleteTimeOffIn, return_direct=False)
def delete_time_off_tool(
    mode: Literal["by_id", "by_day"] = "by_day",
    timeoff_id: Optional[str] = None,
    day: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolDeleteTimeOffOut:
    """Delete time‑off blocks by id or by owner‑local day.

    Args:
        mode: Either "by_id" to delete a single row, or "by_day" to delete all
            blocks overlapping the given day.
        timeoff_id: Required when 'mode='by_id''.
        day: Required when 'mode='by_day''; interpreted in owner timezone.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolDeleteTimeOffOut' with 'deleted_count' and 'deleted_ids'.

    Raises:
        ToolException: For missing owner, missing required arguments, or other
        unexpected errors.

    Notes:
        Deletions are permanent and do not cascade to appointments; they only
        remove the time‑off block.
    """
    try:
        owner_id = _owner_id_from_config(config)
        with SessionLocal() as db:
            owner: Optional[User] = db.query(User).filter(User.id == owner_id).first()
            if not owner:
                raise ToolException("Owner not found")

            deleted_ids: List[str] = []

            if mode == "by_id":
                if not timeoff_id:
                    raise ToolException("timeoff_id is required when mode='by_id'")
                row = (
                    db.query(TimeOff)
                    .filter(TimeOff.id == timeoff_id, TimeOff.owner_id == owner.id)
                    .first()
                )
                if row:
                    deleted_ids.append(str(row.id))
                    db.delete(row)
                    db.commit()
                return ToolDeleteTimeOffOut(
                    deleted_count=len(deleted_ids), deleted_ids=deleted_ids
                )

            if not day:
                raise ToolException("day is required when mode='by_day'")
            start_local, end_local = _owner_local_day_window(owner.timezone, day)
            start_utc, end_utc = _to_utc(start_local), _to_utc(end_local)
            rows = (
                db.query(TimeOff)
                .filter(
                    TimeOff.owner_id == owner.id,
                    TimeOff.start_utc < end_utc,
                    TimeOff.end_utc > start_utc,
                )
                .all()
            )
            for r in rows:
                deleted_ids.append(str(r.id))
                db.delete(r)
            if deleted_ids:
                db.commit()
            return ToolDeleteTimeOffOut(
                deleted_count=len(deleted_ids), deleted_ids=deleted_ids
            )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"delete_time_off failed: {e}")


__all__ = [
    "ToolAddTimeOffIn",
    "ToolAddTimeOffOut",
    "ToolUpdateTimeOffIn",
    "ToolUpdateTimeOffOut",
    "ToolListTimeOffIn",
    "ToolListTimeOffOut",
    "ToolDeleteTimeOffIn",
    "ToolDeleteTimeOffOut",
    "add_time_off_tool",
    "update_time_off_tool",
    "list_time_off_tool",
    "delete_time_off_tool",
]
