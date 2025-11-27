from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import (
    User,
    AvailabilityRule,
    TimeOff,
    Appointment,
    SpecialOpening,
    ServiceOption,
)
from app.schemas import (
    SlotsQuery,
    Slot,
    ServiceOptionOut,
)
from ._helpers import (
    UTC,
    resolve_tz,
    _compute_final_slots_for_day,
    _get_owner,
    _split_slots,
)
from services.services_scheduling import (
    generate_daily_slots,
)
from services.services_scheduling import (
    _as_aware_utc,
)


router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])
"""Router for public/owner scheduling read APIs (slots, calendar, pricing)."""


@router.get("/public/slots", response_model=list[Slot])
def public_slots(
    d: date = Query(..., description="Owner-local calendar day (YYYY-MM-DD)"),
    tz: Optional[str] = Query(
        None,
        description=(
            "Viewer timezone (optional). Ignored for generation; output remains owner-local."
        ),
    ),
    db: Session = Depends(get_db),
) -> list[Slot]:
    """Return bookable slots for a single owner-local day (public).

    Mirrors the owner-authenticated '/slots' generator to keep results
    consistent for the public booking UI. The optional 'tz' is a viewer hint
    and does not affect generation which always uses the owner’s timezone.
    """
    owner = _get_owner(db)
    if not owner:
        raise HTTPException(404, "Owner not found")

    owner_tz = ZoneInfo(owner.timezone)

    # 1) Weekly-rule slots (service respects time-off internally)
    from types import SimpleNamespace

    weekly_pairs = generate_daily_slots(
        db,
        SimpleNamespace(id=owner.id, timezone=str(owner_tz.key)),
        d,
        tz_override=str(owner_tz.key),
    )

    # 2) Compute day boundaries in both local and UTC
    day_start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=owner_tz)
    day_end_local = day_start_local + timedelta(days=1)
    day_start_utc = day_start_local.astimezone(UTC)
    day_end_utc = day_end_local.astimezone(UTC)

    # 3) One-off openings overlapping the day (clamped to this day in local time)
    specials = (
        db.query(SpecialOpening)
        .filter(
            SpecialOpening.owner_id == owner.id,
            SpecialOpening.end_utc > day_start_utc,
            SpecialOpening.start_utc < day_end_utc,
        )
        .all()
    )

    appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.end_utc > day_start_utc,
            Appointment.start_utc < day_end_utc,
            Appointment.status != "canceled",
        )
        .all()
    )

    special_pairs: list[tuple[datetime, datetime]] = []

    from datetime import timedelta as _td

    # Owner-level flag to disable buffers entirely in public availability
    try:
        from services.features import get_owner_flag

        disable_buffers = get_owner_flag(
            str(owner.id), "no_edge_buffer", "FEATURE_NO_EDGE_BUFFER", default=True
        )
    except Exception:
        disable_buffers = True

    def _conflicts_with_appt(
        slot_s_local: datetime, slot_e_local: datetime, buf_min: int
    ) -> bool:
        s_utc = slot_s_local.astimezone(UTC)
        e_utc = slot_e_local.astimezone(UTC)
        pad = _td(minutes=0 if disable_buffers else int(buf_min or 0))
        for a in appts:
            a_s = _as_aware_utc(a.start_utc) - pad
            a_e = _as_aware_utc(a.end_utc) + pad
            if not (e_utc <= a_s or s_utc >= a_e):
                return True
        return False

    for o in specials:
        start_local = max(o.start_utc.astimezone(owner_tz), day_start_local)
        end_local = min(o.end_utc.astimezone(owner_tz), day_end_local)
        if end_local <= start_local:
            continue
        for s, e in _split_slots(
            start_local, end_local, o.slot_minutes, o.buffer_minutes
        ):
            if not _conflicts_with_appt(s, e, o.buffer_minutes):
                special_pairs.append((s, e))

    # 4) Raw union (dedupe via ISO strings)
    union = set((s.isoformat(), e.isoformat()) for s, e in weekly_pairs)
    for s, e in special_pairs:
        union.add((s.isoformat(), e.isoformat()))

    # 5) Subtract any time-off overlapping the day (defensive)
    offs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner.id,
            TimeOff.end_utc > day_start_utc,
            TimeOff.start_utc < day_end_utc,
        )
        .all()
    )

    out: list[Slot] = []
    for s_iso, e_iso in sorted(union):
        s_local = datetime.fromisoformat(s_iso)
        e_local = datetime.fromisoformat(e_iso)
        s_utc, e_utc = s_local.astimezone(UTC), e_local.astimezone(UTC)
        if any(not (e_utc <= off.start_utc or s_utc >= off.end_utc) for off in offs):
            continue
        out.append({"start": s_local, "end": e_local})

    return out


@router.get("/public/holidays", response_model=list[dict])
def public_holidays(
    start: date = Query(..., description="Start date (YYYY-MM-DD) in local tz"),
    end: date = Query(..., description="End date (YYYY-MM-DD) in local tz (inclusive)"),
    tz: Optional[str] = Query(None, description="IANA timezone; defaults to owner tz"),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Public holidays for a date range (for UI overlay). Uses default country/region.
    Returns list of { date, name, start_utc, end_utc }.
    """
    owner = _get_owner(db)
    if not owner:
        raise HTTPException(404, "Owner not found")

    tz_name = tz or getattr(owner, "timezone", None) or "America/Toronto"
    from zoneinfo import ZoneInfo

    tzinfo = ZoneInfo(tz_name)

    try:
        from agent.tools_holidays import (
            _fetch_holidays,
            DEFAULT_COUNTRY,
            DEFAULT_REGION,
        )
    except Exception:
        raise HTTPException(500, "Holiday module not available")

    cc = DEFAULT_COUNTRY
    rc = DEFAULT_REGION
    years: set[int] = set()
    cur = start
    while cur <= end:
        years.add(cur.year)
        cur = cur + timedelta(days=1)

    rows: list[dict] = []
    for y in sorted(years):
        rows.extend(_fetch_holidays(y, cc))

    out: list[dict] = []
    for h in rows:
        dstr = h.get("date")
        if not isinstance(dstr, str):
            continue
        try:
            d = datetime.strptime(dstr, "%Y-%m-%d").date()
        except Exception:
            continue
        if d < start or d > end:
            continue
        counties = h.get("counties")
        applies = (not counties) or (rc and rc in counties)
        if not applies:
            continue
        s_local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tzinfo)
        e_local = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=tzinfo)
        out.append(
            {
                "date": dstr,
                "name": str(h.get("localName") or h.get("name") or "Holiday"),
                "start_utc": s_local.astimezone(UTC),
                "end_utc": e_local.astimezone(UTC),
            }
        )

    out.sort(key=lambda x: x["date"])
    return out


@router.post("/slots", response_model=list[Slot])
def get_slots(
    q: SlotsQuery,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
    tz: Optional[str] = Query(None),
) -> list[Slot]:
    """Owner-authenticated slots for a single local day.

    Generates discrete, bookable slots from weekly rules and one-off openings
    for the given date in the owner’s timezone, subtracting overlapping time
    off and appointments with the opening’s configured buffer.
    """
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    owner_tz = resolve_tz(tz, user)
    from types import SimpleNamespace

    # Weekly-rule slots (already respects timeoffs inside generate_daily_slots)
    weekly_pairs = generate_daily_slots(
        db,
        SimpleNamespace(id=owner.id, timezone=str(owner_tz.key)),
        q.date,
        tz_override=str(owner_tz.key),
    )

    # Compute day boundaries in both local and UTC
    day_start_local = datetime(
        q.date.year, q.date.month, q.date.day, 0, 0, tzinfo=owner_tz
    )
    day_end_local = day_start_local + timedelta(days=1)
    day_start_utc = day_start_local.astimezone(UTC)
    day_end_utc = day_end_local.astimezone(UTC)

    # One-off openings that overlap the day
    specials = (
        db.query(SpecialOpening)
        .filter(
            SpecialOpening.owner_id == owner.id,
            SpecialOpening.end_utc > day_start_utc,
            SpecialOpening.start_utc < day_end_utc,
        )
        .all()
    )

    # Fetch overlapping appointments for this day
    appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.end_utc > day_start_utc,
            Appointment.start_utc < day_end_utc,
            Appointment.status != "canceled",
        )
        .all()
    )

    special_pairs: list[tuple[datetime, datetime]] = []
    from datetime import timedelta as _td

    def _conflicts_with_appt(
        slot_s_local: datetime, slot_e_local: datetime, buf_min: int
    ) -> bool:
        s_utc = slot_s_local.astimezone(UTC)
        e_utc = slot_e_local.astimezone(UTC)
        pad = _td(minutes=int(buf_min or 0))
        for a in appts:
            a_s = _as_aware_utc(a.start_utc) - pad
            a_e = _as_aware_utc(a.end_utc) + pad
            if not (e_utc <= a_s or s_utc >= a_e):
                return True
        return False

    for o in specials:
        start_local = max(o.start_utc.astimezone(owner_tz), day_start_local)
        end_local = min(o.end_utc.astimezone(owner_tz), day_end_local)
        if end_local <= start_local:
            continue
        for s, e in _split_slots(
            start_local, end_local, o.slot_minutes, o.buffer_minutes
        ):
            if not _conflicts_with_appt(s, e, o.buffer_minutes):
                special_pairs.append((s, e))
    all_pairs = set((s.isoformat(), e.isoformat()) for s, e in weekly_pairs)
    for s, e in special_pairs:
        all_pairs.add((s.isoformat(), e.isoformat()))

    # Subtract any time-off that overlaps (defensive; weekly generator already subtracts)
    offs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner.id,
            TimeOff.end_utc > day_start_utc,
            TimeOff.start_utc < day_end_utc,
        )
        .all()
    )

    out: list[Slot] = []
    for s_iso, e_iso in sorted(all_pairs):
        s_local = datetime.fromisoformat(s_iso)
        e_local = datetime.fromisoformat(e_iso)
        s_utc, e_utc = s_local.astimezone(UTC), e_local.astimezone(UTC)
        if any(not (e_utc <= off.start_utc or s_utc >= off.end_utc) for off in offs):
            continue
        out.append({"start": s_local, "end": e_local})

    return out


@router.get("/slots/range", response_model=Dict[str, List[Slot]])
def slots_range(
    start: date,
    days: int = 14,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
    tz: Optional[str] = Query(None),
) -> Dict[str, List[Slot]]:
    """Owner-authenticated slots over a date range.

    Returns a mapping of 'YYYY-MM-DD' → list of slot objects generated via the
    weekly rules for each day (time off and appointments subtracted by the
    generator).
    """
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")

    owner_tz = resolve_tz(tz, user)
    out: Dict[str, List[Slot]] = {}
    from types import SimpleNamespace

    for i in range(days):
        d = start + timedelta(days=i)
        pairs = generate_daily_slots(
            db,
            SimpleNamespace(id=owner.id, timezone=str(owner_tz.key)),
            d,
            tz_override=str(owner_tz.key),
        )
        out[str(d)] = [{"start": s, "end": e} for s, e in pairs]
    return out


@router.get("/calendar/range")
def calendar_range(
    start: date,
    days: int = 14,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
    tz: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Aggregated per-day calendar data for 'days' starting at 'start' (owner-local):
      - rules        : raw weekly availability windows for that weekday (local)
      - openings     : one-off opening windows overlapping that day (local)
      - timeoffs     : time off blocks overlapping that day (local)
      - appointments : appointments overlapping that day (local)
      - slots        : final bookable slots (local) from your generator (+ special openings)
    """
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    owner_tz = resolve_tz(tz, user)

    out: dict[str, dict[str, list[dict]]] = {}
    for i in range(days):
        d = start + timedelta(days=i)
        day_key = str(d)
        from types import SimpleNamespace

        pairs = generate_daily_slots(
            db,
            SimpleNamespace(id=owner.id, timezone=str(owner_tz.key)),
            d,
            tz_override=str(owner_tz.key),
        )
        slots = [{"start": s.isoformat(), "end": e.isoformat()} for s, e in pairs]

        # Day bounds
        day_start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=owner_tz)
        day_end_local = day_start_local + timedelta(days=1)
        day_start_utc, day_end_utc = (
            day_start_local.astimezone(UTC),
            day_end_local.astimezone(UTC),
        )

        # Timeoffs intersecting that day
        offs = (
            db.query(TimeOff)
            .filter(
                TimeOff.owner_id == owner.id,
                TimeOff.end_utc > day_start_utc,
                TimeOff.start_utc < day_end_utc,
            )
            .all()
        )
        timeoffs = [
            {
                "start": o.start_utc.astimezone(owner_tz).isoformat(),
                "end": o.end_utc.astimezone(owner_tz).isoformat(),
                "note": o.note,
            }
            for o in offs
        ]

        # Appointments that day
        appts = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.end_utc > day_start_utc,
                Appointment.start_utc < day_end_utc,
            )
            .all()
        )
        appointments = [
            {
                "start": a.start_utc.astimezone(owner_tz).isoformat(),
                "end": a.end_utc.astimezone(owner_tz).isoformat(),
                "status": a.status,
                "client_id": a.client_id,
                "person_id": a.person_id,
            }
            for a in appts
        ]

        # One-off openings overlapping the day
        opens = (
            db.query(SpecialOpening)
            .filter(
                SpecialOpening.owner_id == owner.id,
                SpecialOpening.end_utc > day_start_utc,
                SpecialOpening.start_utc < day_end_utc,
            )
            .all()
        )
        openings = [
            {
                "start": o.start_utc.astimezone(owner_tz).isoformat(),
                "end": o.end_utc.astimezone(owner_tz).isoformat(),
                "slot_minutes": o.slot_minutes,
                "buffer_minutes": o.buffer_minutes,
                "note": o.note,
            }
            for o in opens
        ]

        # Raw availability rules for that weekday (so UI can show “available bands”)
        rules_q = (
            db.query(AvailabilityRule)
            .filter_by(owner_id=owner.id, weekday=d.weekday())
            .all()
        )
        rules = [
            {
                "start": datetime(
                    d.year,
                    d.month,
                    d.day,
                    r.start_local.hour,
                    r.start_local.minute,
                    tzinfo=owner_tz,
                ).isoformat(),
                "end": datetime(
                    d.year,
                    d.month,
                    d.day,
                    r.end_local.hour,
                    r.end_local.minute,
                    tzinfo=owner_tz,
                ).isoformat(),
                "slot_minutes": r.slot_minutes,
                "buffer_minutes": r.buffer_minutes,
            }
            for r in rules_q
        ]

        out[day_key] = {
            "rules": rules,
            "openings": openings,
            "timeoffs": timeoffs,
            "appointments": appointments,
            "slots": slots,
        }

    return out


@router.get("/slots/debug")
def slots_debug(
    date: date, db: Session = Depends(get_db), user: TokenUser = Depends(require_owner)
) -> Dict[str, Any]:
    """
    Debug view for a single date showing how final slots are produced.
    Returns weekly_pairs, special_pairs, timeoffs, and final_slots.
    """
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    owner_tz = ZoneInfo(owner.timezone)
    day_start_local = datetime(date.year, date.month, date.day, 0, 0, tzinfo=owner_tz)
    day_end_local = day_start_local + timedelta(days=1)
    day_start_utc = day_start_local.astimezone(UTC)
    day_end_utc = day_end_local.astimezone(UTC)

    # 1) Weekly (from rules)
    weekly_pairs = generate_daily_slots(db, owner, date)
    weekly_pairs_iso = [
        {"start": s.isoformat(), "end": e.isoformat()} for s, e in weekly_pairs
    ]

    # 2) One-off openings for this date
    specials = (
        db.query(SpecialOpening)
        .filter(
            SpecialOpening.owner_id == owner.id,
            SpecialOpening.end_utc > day_start_utc,
            SpecialOpening.start_utc < day_end_utc,
        )
        .all()
    )

    special_pairs = []
    for o in specials:
        start_local = max(o.start_utc.astimezone(owner_tz), day_start_local)
        end_local = min(o.end_utc.astimezone(owner_tz), day_end_local)
        if end_local <= start_local:
            continue
        for s, e in _split_slots(
            start_local, end_local, o.slot_minutes, o.buffer_minutes
        ):
            special_pairs.append(
                {"start": s.isoformat(), "end": e.isoformat(), "opening_id": o.id}
            )

    # 3) Raw union (before subtracting timeoff)
    union = set((p["start"], p["end"]) for p in weekly_pairs_iso)
    for p in special_pairs:
        union.add((p["start"], p["end"]))

    # 4) Timeoffs that intersect this date
    offs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner.id,
            TimeOff.end_utc > day_start_utc,
            TimeOff.start_utc < day_end_utc,
        )
        .all()
    )
    offs_iso = [
        {
            "start_utc": o.start_utc.isoformat(),
            "end_utc": o.end_utc.isoformat(),
            "note": o.note,
        }
        for o in offs
    ]

    # 5) Final slots = union - timeoff overlaps (defensive filter)
    final_slots = []
    for s_iso, e_iso in sorted(union):
        s_local = datetime.fromisoformat(s_iso)
        e_local = datetime.fromisoformat(e_iso)
        s_utc, e_utc = s_local.astimezone(UTC), e_local.astimezone(UTC)
        if any(not (e_utc <= o.start_utc or s_utc >= o.end_utc) for o in offs):
            continue
        final_slots.append({"start": s_iso, "end": e_iso})

    return {
        "owner_id": owner.id,
        "date": str(date),
        "weekly_pairs": weekly_pairs_iso,
        "special_pairs": special_pairs,
        "timeoffs": offs_iso,
        "final_slots": final_slots,
    }


@router.get("/public/service-options", response_model=List[ServiceOptionOut])
def public_service_options(
    db: Session = Depends(get_db),
) -> List[ServiceOptionOut]:
    """
    Client-facing: list active durations and prices for the single owner.
    """
    owner = _get_owner(db)
    return (
        db.query(ServiceOption)
        .filter(ServiceOption.owner_id == owner.id, ServiceOption.is_active == 1)
        .order_by(ServiceOption.duration_minutes.asc())
        .all()
    )


@router.get("/public/slots-priced", response_model=List[dict])
def public_slots_priced(
    day: date = Query(..., description="Owner-local date (YYYY-MM-DD)"),
    duration_minutes: Optional[int] = Query(
        None, description="If provided, only return slots of this duration"
    ),
    db: Session = Depends(get_db),
) -> List[dict]:
    """
    Client-facing: return discrete, bookable slots for the given day,
    enriched with price info from active ServiceOptions.

    Response shape (per slot):
      {
        "start": ISO8601 (owner local),
        "end": ISO8601 (owner local),
        "duration_minutes": int,
        "price_cents": int,
        "currency": "USD" | ...
      }
    """
    owner = _get_owner(db)
    openings = _compute_final_slots_for_day(db, owner, day, owner.timezone)
    q = db.query(ServiceOption).filter(
        ServiceOption.owner_id == owner.id, ServiceOption.is_active == 1
    )
    if duration_minutes is not None:
        q = q.filter(ServiceOption.duration_minutes == duration_minutes)
    options = q.order_by(ServiceOption.duration_minutes.asc()).all()

    if not options or not openings:
        return []
    from datetime import timedelta as _td

    def _split_span(start_local: datetime, end_local: datetime, minutes: int):
        step = _td(minutes=minutes)
        cur = start_local
        while cur + step <= end_local:
            yield cur, cur + step
            cur += step

    out: List[dict] = []
    for opt in options:
        for s_local, e_local in openings:
            for slot_s, slot_e in _split_span(s_local, e_local, opt.duration_minutes):
                out.append(
                    {
                        "start": slot_s.isoformat(),
                        "end": slot_e.isoformat(),
                        "duration_minutes": opt.duration_minutes,
                        "price_cents": opt.price_cents,
                        "currency": opt.currency,
                    }
                )
    out.sort(key=lambda x: (x["start"], x["duration_minutes"]))
    return out


class PublicPricingOut(BaseModel):
    options: list[dict]
    admin_fee_cents: int | None = None


@router.get("/public/pricing", response_model=PublicPricingOut)
def public_pricing(db: Session = Depends(get_db)) -> PublicPricingOut:
    """
    Client-facing: list active durations and prices for the single owner.
    """
    owner = _get_owner(db)
    opts = (
        db.query(ServiceOption)
        .filter(ServiceOption.owner_id == owner.id, ServiceOption.is_active == 1)
        .order_by(ServiceOption.duration_minutes.asc())
        .all()
    )
    try:
        from services.admin_fee import get_admin_fee_setting

        fee = get_admin_fee_setting(db, owner.id)
        admin_fee_cents = int(fee.admin_fee_cents or 0)
    except Exception:
        admin_fee_cents = None

    return {
        "options": [
            {
                "duration_minutes": int(o.duration_minutes),
                "price_cents": int(o.price_cents),
            }
            for o in opts
        ],
        "admin_fee_cents": admin_fee_cents,
    }
