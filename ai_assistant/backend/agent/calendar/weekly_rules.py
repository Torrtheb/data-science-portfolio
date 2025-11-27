from __future__ import annotations

from typing import Optional
from datetime import datetime, time as _time
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, ToolException

from app.db import SessionLocal
from app.models import User, AvailabilityRule, SpecialOpening

from agent.tools import _parse_owner_day, _parse_owner_local_dt, _to_utc
from agent.calendar.openings import (
    add_special_opening_tool,
    _appt_cols,
    _iso_local,
    _owner_id_from_config,
    _fmt_local_range,
)
from agent.constants import ACTIVE_APPT_STATUSES
from app.models import Appointment
from agent.schemas import (
    ToolAddAvailabilityIn,
    ToolAddAvailabilityOut,
    ToolWeeklyRuleOut,
    ToolDeleteWeeklyRuleIn,
    ToolListWeeklyRulesOut,
    ToolCreateWeeklyRuleIn,
    ToolDeleteWeeklyRuleOut,
    ToolUpdateWeeklyRuleIn,
    ToolUpdateWeeklyRuleOut,
)


def _rule_to_out(r: AvailabilityRule) -> ToolWeeklyRuleOut:
    """Map an 'AvailabilityRule' ORM row to the tool's output schema.

    - Computes convenience fields like 'start_minute'/'end_minute' and
      user-friendly 'start_hhmm'/'end_hhmm'.
    - Handles optional columns ('slot_minutes', 'buffer_minutes', 'note').
    """
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


def _list_conflicts(db, owner: User, s_utc: datetime, e_utc: datetime, owner_tz: str):
    """Collect conflicting items with a proposed availability window.

    Returns a dictionary with:
    - 'timeoffs': List[TimeOff] overlapping [s_utc, e_utc)
    - 'appointments': List[Appointment] overlapping [s_utc, e_utc) and in
      'ACTIVE_APPT_STATUSES'
    - 'openings': List[dict] of existing opening windows overlapping the range
      on the same owner-local day (from 'owner_calendar_snapshot')
    """
    from app.models import TimeOff
    from services.services_scheduling import owner_calendar_snapshot

    toffs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner.id,
            TimeOff.start_utc < e_utc,
            TimeOff.end_utc > s_utc,
        )
        .order_by(TimeOff.start_utc.asc())
        .all()
    )
    scol, ecol = _appt_cols()
    appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            getattr(Appointment, scol) < e_utc,
            getattr(Appointment, ecol) > s_utc,
            Appointment.status.in_(ACTIVE_APPT_STATUSES),
        )
        .order_by(getattr(Appointment, scol).asc())
        .all()
    )
    tz = ZoneInfo(owner_tz)
    day = s_utc.astimezone(tz).date()
    snap = owner_calendar_snapshot(
        db, owner_id=owner.id, scope="today", anchor=day, tz_str=owner_tz
    )
    openings = []
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
            if st < e_utc and en > s_utc:
                openings.append({"start_utc": st, "end_utc": en, "id": ev.get("id")})
    return {"timeoffs": toffs, "appointments": appts, "openings": openings}


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
    """Create a one-off availability (special opening) for the owner.

    Behavior
    - Parses 'start_local'/'end_local' in the owner's timezone and stores as UTC.
    - Rejects if 'end_local <= start_local'.
    - If the requested window overlaps existing openings, raises a 'ToolException'
      explaining that availability already exists (with a preview list).
    - If time off or appointments overlap and 'confirm_if_conflicts' is False,
      raises 'ToolException("CONFIRM_REQUIRED:<json>")' with a structured payload
      containing the proposed tool call and conflict previews. Call again with
      'confirm_if_conflicts=True' to proceed.

    Returns
    - 'ToolAddAvailabilityOut': '{ id, start_utc, end_utc, slot_minutes, buffer_minutes, note }'.

    Errors
    - 'ToolException("Owner not found")' if owner resolution fails.
    - 'ToolException("end_local must be after start_local")' for invalid bounds.
    - 'ToolException("add_availability failed: ...")' for unexpected errors.
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

            toff = conflicts["timeoffs"]
            appts = conflicts["appointments"]
            if (toff or appts) and not confirm_if_conflicts:
                import json

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

            out = add_special_opening_tool(
                start_local=start_local,
                end_local=end_local,
                slot_minutes=slot_minutes,
                buffer_minutes=buffer_minutes,
                note=note,
                config=config,
            )

            try:
                row = (
                    db.query(SpecialOpening)
                    .filter(
                        SpecialOpening.owner_id == owner.id, SpecialOpening.id == out.id
                    )
                    .first()
                )
                if row:
                    from agent.tools_calendar import (
                        _merge_overlapping_or_touching_openings as _merge,
                    )

                    row = _merge(db, owner, row)
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


@tool("list_weekly_rules", return_direct=False)
def list_weekly_rules_tool(config: RunnableConfig) -> ToolListWeeklyRulesOut:
    """List the owner's weekly availability rules (read-only).

    Arguments
    - 'config': LangChain Runnable config carrying the owner identity.

    Returns
    - 'ToolListWeeklyRulesOut': 'rules' list with items containing
      '{ id, weekday, start_local, end_local, slot_minutes, buffer_minutes, active }'.
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
    """Create a weekly availability rule for a given weekday.

    Arguments
    - 'weekday': 0=Monday … 6=Sunday (owner-local).
    - 'start_hhmm' / 'end_hhmm': HH:MM strings in 24h format.
    - 'slot_minutes' / 'buffer_minutes': Rule parameters.
    - 'note': Optional rule note.
    - 'config': LangChain Runnable config carrying the owner identity.

    Behavior
    - If an identical rule exists for the owner (same weekday/start/end), it is
      reused and, if needed, updated for slot/buffer/note values; otherwise a
      new rule is created.

    Returns
    - 'ToolWeeklyRuleOut' describing the created/existing rule.

    Errors
    - 'RuntimeError("Invalid HH:MM")' or clock range errors for invalid time strings.
    - 'RuntimeError("end must be after start")' if the window is invalid.
    """
    owner_id = _owner_id_from_config(config)

    def _parse_hhmm_to_minutes(hhmm: str) -> int:
        import re

        s = (hhmm or "").strip()
        if not re.match(r"^\d{1,2}:\d{2}$", s):
            raise RuntimeError(f"Invalid HH:MM: {hhmm!r}")
        h, m = s.split(":")
        h, m = int(h), int(m)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise RuntimeError(f"Invalid clock time: {hhmm!r}")
        return h * 60 + m

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
    """Delete a weekly availability rule by id (idempotent).

    - If the rule does not exist for the owner, returns 'deleted=False'.
    - On success, returns 'deleted=True' and echoes the 'rule_id'.
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
            return ToolDeleteWeeklyRuleOut(deleted=False, rule_id=rule_id)
        db.delete(r)
        db.commit()
        return ToolDeleteWeeklyRuleOut(deleted=True, rule_id=rule_id)


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
    """Update a weekly availability rule.

    Selection
    - Provide 'rule_id' to target a specific rule.
    - If omitted, provide 'weekday'; if multiple rules exist that day, returns
      'ambiguous=True' with candidate rules to choose from.

    Updates
    - Supports changing 'start_local'/'end_local' (accepts 'HH:MM' or 'H:MMam/pm'),
      'slot_minutes', and 'buffer_minutes'.
    - Validates that end is after start.

    Conflicts
    - If 'anchor_day' is provided with a new 'end_local' (interpreted as a
      cutoff), checks for appointments after the cutoff on that day. If any are
      found, returns 'requires_cancellation=True' with 'blocked_appointments'.

    Returns
    - 'ToolUpdateWeeklyRuleOut' with 'ok=True' and the updated rule summary, or
      'ok=False' with 'ambiguous=True' or 'requires_cancellation=True'.

    Errors
    - 'ToolException("Weekly rule not found")' if the target cannot be resolved.
    - 'ToolException("Provide rule_id or weekday")' if neither selector is given.
    - 'ToolException("Owner not found")' if owner resolution fails.
    - 'ToolException("end_local must be after start_local")' for invalid bounds.
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
