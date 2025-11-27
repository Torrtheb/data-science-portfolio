from __future__ import annotations
from datetime import datetime, timedelta, time, date
from typing import List, Optional, Tuple, Dict, Any, Iterable
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_, and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import uuid
from app.models import (
    AvailabilityRule,
    Appointment,
    TimeOff,
    User,
    ServiceOption,
    SpecialOpening,
    Person,
    ClientAccount,
    ClientEmail,
)
from app.models import PrepaidBundle
from services.wallets import auto_apply_wallet_funds
from agent.constants import ACTIVE_APPT_STATUSES

DEFAULT_APPT_EDGE_BUFFER_MIN = 0
UTC = ZoneInfo("UTC")


class ServiceBookingError(Exception):
    """
    Domain-level error for scheduling/booking operations.

    Attributes:
        code: Short machine-readable code (e.g., 'NO_AVAILABILITY', 'AMBIGUOUS_PERSON').
        message: Human-readable message suitable for surfacing to the user.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _as_aware_utc(dt: datetime) -> datetime:
    """
    Return an aware UTC datetime.

    If 'dt' is naive, attaches UTC. If 'dt' is aware (any TZ), converts to UTC.

    Args:
        dt: A naive or timezone-aware datetime.

    Returns:
        An aware UTC datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _to_utc(dt: datetime) -> datetime:
    """
    Convert a timezone-aware datetime to UTC.

    No-op if already UTC.

    Args:
        dt: An aware datetime.

    Returns:
        The same instant in UTC.
    """
    return dt.astimezone(UTC)


def _local_dt(owner_tz: str, d: date, t: time) -> datetime:
    """
    Construct an owner-local datetime for a given date and time-of-day.

    Args:
        owner_tz: IANA timezone string (e.g., 'America/Toronto').
        d: Date component.
        t: Time component.

    Returns:
        An aware datetime in 'owner_tz'.
    """
    return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=ZoneInfo(owner_tz))


def _overlaps(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> bool:
    """
    Half-open interval overlap for '[a_start, a_end)' and '[b_start, b_end)'.

    Returns:
        True if the intervals intersect, False otherwise.
    """
    return (a_start < b_end) and (b_start < a_end)


# ---------------------------------------------------------------------
# PERSON / IDENTITY RESOLUTION (HARDENED)
# ---------------------------------------------------------------------
def _account_primary_email(db: Session, account_id: int) -> Optional[str]:
    """
    Return the best email for a client account.

    Priority:
        1) Primary (or first) ClientEmail for the account
        2) The linked ClientAccount.client_user_id's User.email (if present)

    Args:
        db: Active SQLAlchemy session.
        account_id: ClientAccount id.

    Returns:
        Email string or None if no email can be determined.
    """
    email = (
        db.query(ClientEmail.email)
        .filter(ClientEmail.account_id == account_id)
        .order_by(ClientEmail.is_primary.desc(), ClientEmail.id.asc())
        .limit(1)
        .scalar()
    )
    if email:
        return email

    return (
        db.query(User.email)
        .join(ClientAccount, ClientAccount.client_user_id == User.id)
        .filter(ClientAccount.id == account_id)
        .limit(1)
        .scalar()
    )


def _resolve_person_for_owner(
    db: Session,
    owner_id: str,
    *,
    client_email: Optional[str],
    client_name: Optional[str],
    client_query: Optional[str],
) -> Tuple[Optional[Person], Optional[str], Optional[str]]:
    """
    Resolve a 'Person' within an owner's book using email/name/query.

    Resolution order:
        1) Email-based match:
           - Direct Person.email within this owner's accounts.
           - Any ClientEmail.email on accounts owned by the owner.
           - Linked auth.User.email of accounts owned by the owner.
           If exactly one account matches but has multiple People → AMBIGUOUS_PERSON.
        2) Name-based match (case-insensitive):
           - Person.full_name within owner’s accounts.
           - ClientAccount.name (if exactly one account and exactly one Person).
             If multiple accounts/people match → AMBIGUOUS_PERSON.
        3) Fallback:
           - Return (None, denorm_name, denorm_email) if no 'Person' found.

    Returns:
        (person | None, denorm_name | None, denorm_email | None)

    Raises:
        ServiceBookingError('AMBIGUOUS_PERSON', ...) when multiple candidates match.
    """

    owner_id = str(owner_id)
    email_input = (client_email or "").strip()
    email_lower = email_input.lower() or None
    name = (client_name or "").strip() or None
    cq = (client_query or "").strip() or None

    # 1) Email resolution (Person.email OR account email)
    if email_lower:
        person_by_email = (
            db.query(Person)
            .join(ClientAccount, ClientAccount.id == Person.account_id)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                func.lower(Person.email) == email_lower,
            )
            .order_by(Person.id.asc())
            .first()
        )
        if person_by_email:
            denorm_name = person_by_email.full_name or name
            acct_email = _account_primary_email(db, person_by_email.account_id)
            denorm_email = person_by_email.email or acct_email or email_input
            return person_by_email, denorm_name, denorm_email

        account_ids: set[int] = set(
            acc_id
            for (acc_id,) in (
                db.query(ClientAccount.id)
                .join(ClientEmail, ClientEmail.account_id == ClientAccount.id)
                .filter(
                    ClientAccount.owner_user_id == owner_id,
                    func.lower(ClientEmail.email) == email_lower,
                )
                .all()
            )
        )

        account_ids.update(
            acc_id
            for (acc_id,) in (
                db.query(ClientAccount.id)
                .join(User, User.id == ClientAccount.client_user_id)
                .filter(
                    ClientAccount.owner_user_id == owner_id,
                    func.lower(User.email) == email_lower,
                )
                .all()
            )
        )

        if len(account_ids) == 1:
            acct_id = next(iter(account_ids))
            acct = db.query(ClientAccount).filter(ClientAccount.id == acct_id).first()
            if acct:
                acct_people = (
                    db.query(Person)
                    .filter(Person.account_id == acct.id)
                    .order_by(Person.id.asc())
                    .all()
                )
                acct_email = _account_primary_email(db, acct.id) or email_input
                if len(acct_people) == 1:
                    p = acct_people[0]
                    denorm_name = p.full_name or name or acct.name
                    denorm_email = p.email or acct_email or email_input
                    return p, denorm_name, denorm_email
                if len(acct_people) > 1:
                    raise ServiceBookingError(
                        "AMBIGUOUS_PERSON",
                        "Multiple people match that email; please specify.",
                    )
                return None, (name or acct.name), acct_email
        elif len(account_ids) > 1:
            raise ServiceBookingError(
                "AMBIGUOUS_PERSON",
                "Multiple client accounts match that email; please clarify.",
            )

    # choose search token for name lookup
    search = (name or cq or "").strip()
    if search:
        like = f"%{search}%"

        # 2a) Try a direct Person name match (ilike for clarity on PG)
        people = (
            db.query(Person)
            .join(ClientAccount, ClientAccount.id == Person.account_id)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                Person.full_name.ilike(like),
            )
            .all()
        )
        if len(people) == 1:
            p = people[0]
            acct_email = _account_primary_email(db, p.account_id)
            return p, (p.full_name or name), (p.email or acct_email or email_input)

        if len(people) > 1:
            raise ServiceBookingError(
                "AMBIGUOUS_PERSON", "Multiple people match that name; please clarify."
            )

        # 2b) If no Person matched, try account name (common case: user says “Fluffy” = account)
        accts = (
            db.query(ClientAccount)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                ClientAccount.name.ilike(like),
            )
            .all()
        )
        if len(accts) == 1:
            acct = accts[0]
            acct_people = db.query(Person).filter(Person.account_id == acct.id).all()
            acct_email = _account_primary_email(db, acct.id) or email_input
            if len(acct_people) == 1:
                p = acct_people[0]
                return (
                    p,
                    (p.full_name or name or acct.name),
                    (p.email or acct_email or email_input),
                )
            if len(acct_people) > 1:
                raise ServiceBookingError(
                    "AMBIGUOUS_PERSON",
                    "This account has multiple people. Please specify which person.",
                )
            # account present but no person yet — fall back to denorm with account email/name
            return None, (name or acct.name), acct_email

        if len(accts) > 1:
            raise ServiceBookingError(
                "AMBIGUOUS_PERSON",
                "Multiple client accounts match that name; please clarify.",
            )

    # 3) Nothing resolved — return denormalized identity (if present)
    denorm_email = email_input or None
    denorm_name = name or None
    return None, denorm_name, denorm_email


# ---------------------------------------------------------------------
# SPECIAL OPENING HELPERS (unchanged logic, tidied)
# ---------------------------------------------------------------------
def _ranges_overlap(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    """
    Half-open interval overlap test for '[a0, a1)' and '[b0, b1)'.

    Returns:
        True if the intervals intersect, False otherwise.
    """
    return a0 < b1 and b0 < a1


def carve_opening_through_timeoff(
    db: Session,
    owner_id: str,
    start_utc: datetime,
    end_utc: datetime,
    *,
    note_suffix: str = " (carved by opening)",
) -> None:
    """
    Split/trim 'TimeOff' windows so a special opening can pass through.

    For each TimeOff intersecting '[start_utc, end_utc)':
        - Fully covered: delete
        - Head overlap: move 'end_utc' earlier
        - Tail overlap: move 'start_utc' later
        - Opening in the middle: split into two windows

    Args:
        db: Active SQLAlchemy session (locks rows via SELECT ... FOR UPDATE).
        owner_id: Owner id.
        start_utc: Opening start (UTC, aware).
        end_utc: Opening end (UTC, aware).
        note_suffix: Text appended to modified time-off notes for auditability.

    Returns:
        None (modifies DB state).
    """
    offs: Iterable[TimeOff] = (
        db.query(TimeOff)
        .filter(TimeOff.owner_id == owner_id)
        .filter(and_(TimeOff.start_utc < end_utc, TimeOff.end_utc > start_utc))
        .with_for_update()
        .all()
    )

    for off in offs:
        t0, t1 = _as_aware_utc(off.start_utc), _as_aware_utc(off.end_utc)
        if not _ranges_overlap(t0, t1, start_utc, end_utc):
            continue

        # Fully covered -> delete
        if start_utc <= t0 and t1 <= end_utc:
            db.delete(off)
            continue

        # Overlap head -> move end earlier
        if t0 < start_utc <= t1 <= end_utc:
            off.end_utc = start_utc
            off.note = (off.note or "") + note_suffix
            continue

        # Overlap tail -> move start later
        if start_utc <= t0 < end_utc < t1:
            off.start_utc = end_utc
            off.note = (off.note or "") + note_suffix
            continue

        # Opening in the middle -> split
        if t0 < start_utc and end_utc < t1:
            left_end = start_utc
            right_start = end_utc

            off.end_utc = left_end
            off.note = (off.note or "") + note_suffix

            # add right piece
            right = TimeOff(
                owner_id=owner_id,
                start_utc=right_start,
                end_utc=t1,
                note=(off.note or "Time off") + note_suffix,
            )
            db.add(right)


def merge_or_get_special_opening(
    db: Session,
    owner_id: str,
    start_utc: datetime,
    end_utc: datetime,
    *,
    slot_minutes: int,
    buffer_minutes: int,
    note: Optional[str] = None,
) -> SpecialOpening:
    """
    Return an existing covering special opening, or create/merge as needed.

    Behavior:
        - No overlaps → create a new SpecialOpening.
        - Fully covering overlap → update missing metadata on that row and return it.
        - Partial overlaps → merge all overlaps into a single union span and return it.

    Args:
        db: Active session.
        owner_id: Owner id.
        start_utc, end_utc: Opening window (UTC, aware).
        slot_minutes: Slot length to store (minutes).
        buffer_minutes: Edge buffer (minutes).
        note: Optional human note.

    Returns:
        The persisted SpecialOpening instance (possibly newly created/merged).
    """
    overlaps = (
        db.query(SpecialOpening)
        .filter(SpecialOpening.owner_id == owner_id)
        .filter(SpecialOpening.start_utc < end_utc)
        .filter(SpecialOpening.end_utc > start_utc)
        .with_for_update()
        .order_by(SpecialOpening.start_utc.asc())
        .all()
    )

    if not overlaps:
        sp = SpecialOpening(
            owner_id=owner_id,
            start_utc=start_utc,
            end_utc=end_utc,
            slot_minutes=int(slot_minutes),
            buffer_minutes=int(buffer_minutes),
            note=note or "Availability",
        )
        db.add(sp)
        db.flush()
        return sp

    # Reuse if fully covered
    for sp in overlaps:
        if sp.start_utc <= start_utc and end_utc <= sp.end_utc:
            if not getattr(sp, "slot_minutes", None):
                sp.slot_minutes = int(slot_minutes)
            if not getattr(sp, "buffer_minutes", None):
                sp.buffer_minutes = int(buffer_minutes)
            if not getattr(sp, "note", None):
                sp.note = note or "Availability"
            db.flush()
            return sp

    # Merge union [min_start, max_end]
    min_start = min([start_utc] + [sp.start_utc for sp in overlaps])
    max_end = max([end_utc] + [sp.end_utc for sp in overlaps])

    base = overlaps[0]
    base.start_utc = min_start
    base.end_utc = max_end
    base.slot_minutes = int(getattr(base, "slot_minutes", slot_minutes) or slot_minutes)
    base.buffer_minutes = int(
        getattr(base, "buffer_minutes", buffer_minutes) or buffer_minutes
    )
    base.note = getattr(base, "note", None) or note or "Availability"

    for sp in overlaps[1:]:
        db.delete(sp)

    db.flush()
    return base


# ---------------------------------------------------------------------
# AVAILABILITY & SLOTS
# ---------------------------------------------------------------------
def is_owner_time_bookable(
    db: Session, owner: User, s_utc: datetime, e_utc: datetime
) -> Tuple[bool, str]:
    """
    Check if a UTC interval is bookable for the owner.

    A time is bookable if:
        - It does NOT intersect any TimeOff.
        - It does NOT intersect any ACTIVE appointments.
        - It IS fully covered by either a SpecialOpening, or by a weekly AvailabilityRule.

    If weekly rules apply, the interval must lie within a single local day
    (no cross-boundary spanning within weekly rules).

    Args:
        db: Active session.
        owner: User row with timezone.
        s_utc, e_utc: Start/end (UTC, aware).

    Returns:
        (ok, reason) where 'ok' is True when bookable; 'reason' is a short string otherwise.
    """
    toff = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner.id,
            TimeOff.start_utc < e_utc,
            TimeOff.end_utc > s_utc,
        )
        .first()
    )
    if toff:
        return (False, "time off")

    appt = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.status.in_(ACTIVE_APPT_STATUSES),
            Appointment.start_utc < e_utc,
            Appointment.end_utc > s_utc,
        )
        .first()
    )
    if appt:
        return (False, "conflicts with another appointment")

    sp = (
        db.query(SpecialOpening)
        .filter(
            SpecialOpening.owner_id == owner.id,
            SpecialOpening.start_utc <= s_utc,
            SpecialOpening.end_utc >= e_utc,
        )
        .first()
    )
    if sp:
        return (True, "")

    tz = ZoneInfo(owner.timezone or "America/Toronto")
    s_loc = s_utc.astimezone(tz)
    e_loc = e_utc.astimezone(tz)
    if s_loc.date() != e_loc.date():
        return (False, "crosses day boundary")

    weekday = s_loc.weekday()
    rules = (
        db.query(AvailabilityRule)
        .filter(
            AvailabilityRule.owner_id == owner.id,
            AvailabilityRule.weekday == weekday,
        )
        .all()
    )

    for r in rules:
        # skip disabled rules and invalid lengths
        if getattr(r, "active", True) is False:
            continue
        if int(getattr(r, "slot_minutes", 0) or 0) <= 0:
            continue
        rs = s_loc.replace(
            hour=r.start_local.hour,
            minute=r.start_local.minute,
            second=0,
            microsecond=0,
        )
        re = s_loc.replace(
            hour=r.end_local.hour, minute=r.end_local.minute, second=0, microsecond=0
        )
        if rs <= s_loc and e_loc <= re:
            return (True, "")

    return (False, "no opening covers the requested time")


def generate_daily_slots(
    db: Session, owner: User, day: date, tz_override: str | None = None
) -> List[tuple[datetime, datetime]]:
    """
    Generate available local-time slots for a given owner and day.

    Composition:
        - Start with weekly rules and specials on that day (in owner-local time).
        - Convert TimeOff + ACTIVE appointments to busy intervals in UTC.
        - Subtract busy intervals (expanded by per-window buffer minutes).
        - Skip past slots (end <= now).

    Args:
        db: Active session.
        owner: Owner user row (provides timezone).
        day: Local day for which to compute slots.
        tz_override: Optional alternative timezone (for preview/testing).

    Returns:
        List of '(start_local, end_local)' slot tuples sorted by start time.
    """

    owner_tz = tz_override or owner.timezone or "America/Toronto"

    day_start_local = datetime(
        day.year, day.month, day.day, 0, 0, tzinfo=ZoneInfo(owner_tz)
    )
    day_end_local = day_start_local + timedelta(days=1)
    day_start_utc, day_end_utc = _to_utc(day_start_local), _to_utc(day_end_local)

    appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.status.in_(ACTIVE_APPT_STATUSES),
            Appointment.start_utc < day_end_utc,
            Appointment.end_utc > day_start_utc,
        )
        .all()
    )

    offs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner.id,
            TimeOff.start_utc < day_end_utc,
            TimeOff.end_utc > day_start_utc,
        )
        .all()
    )

    busy_utc = [
        (_as_aware_utc(a.start_utc), _as_aware_utc(a.end_utc)) for a in appts
    ] + [(_as_aware_utc(o.start_utc), _as_aware_utc(o.end_utc)) for o in offs]

    windows: list[tuple[datetime, datetime, int, int]] = []

    rules = (
        db.query(AvailabilityRule)
        .filter_by(owner_id=owner.id, weekday=day.weekday())
        .all()
    )
    disable_buffers = False
    try:
        from services.features import get_owner_flag

        disable_buffers = get_owner_flag(
            str(owner.id), "no_edge_buffer", "FEATURE_NO_EDGE_BUFFER", default=True
        )
    except Exception:
        disable_buffers = True
    for r in rules:
        slot_len_min = int(getattr(r, "slot_minutes", 0) or 0)
        if getattr(r, "active", True) is False or slot_len_min <= 0:
            continue
        start_local = _local_dt(owner_tz, day, r.start_local)
        end_local = _local_dt(owner_tz, day, r.end_local)
        if start_local < end_local:
            buf = 0 if disable_buffers else int(getattr(r, "buffer_minutes", 0) or 0)
            windows.append((start_local, end_local, slot_len_min, buf))

    specials = (
        db.query(SpecialOpening)
        .filter(
            SpecialOpening.owner_id == owner.id,
            SpecialOpening.start_utc < day_end_utc,
            SpecialOpening.end_utc > day_start_utc,
        )
        .all()
    )
    for s in specials:
        slot_len_min = int(getattr(s, "slot_minutes", 0) or 0)
        if slot_len_min <= 0:
            continue
        s_local = _as_aware_utc(s.start_utc).astimezone(ZoneInfo(owner_tz))
        e_local = _as_aware_utc(s.end_utc).astimezone(ZoneInfo(owner_tz))
        start_local = max(s_local, day_start_local)
        end_local = min(e_local, day_end_local)
        if start_local < end_local:
            buf = 0 if disable_buffers else int(getattr(s, "buffer_minutes", 0) or 0)
            windows.append((start_local, end_local, slot_len_min, buf))

    now_local = datetime.now(ZoneInfo(owner_tz))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[datetime, datetime]] = []

    for win_start_local, win_end_local, slot_minutes, buffer_minutes in windows:
        slot_len = timedelta(minutes=slot_minutes)
        buffer_len = timedelta(minutes=buffer_minutes)

        cur = win_start_local
        while cur + slot_len <= win_end_local:
            slot_start_local = cur
            slot_end_local = cur + slot_len

            if slot_end_local <= now_local:
                cur += slot_len
                continue

            slot_start_utc = _to_utc(slot_start_local)
            slot_end_utc = _to_utc(slot_end_local)

            conflict = False
            for b_start, b_end in busy_utc:
                b_start_buf = b_start - buffer_len
                b_end_buf = b_end + buffer_len
                if _overlaps(slot_start_utc, slot_end_utc, b_start_buf, b_end_buf):
                    conflict = True
                    break

            if not conflict:
                key = (
                    int(slot_start_local.timestamp()),
                    int(slot_end_local.timestamp()),
                )
                if key not in seen:
                    seen.add(key)
                    out.append((slot_start_local, slot_end_local))

            cur += slot_len

    out.sort(key=lambda s: s[0])
    return out


# ---------------------------------------------------------------------
# CALENDAR SNAPSHOT
# ---------------------------------------------------------------------
def _normalize_window(
    anchor: Optional[date], scope: str, tz_str: str
) -> tuple[datetime, datetime, ZoneInfo]:
    """
    Normalize a reporting window from 'scope' and optional 'anchor'.

    Scopes:
        - 'today' : [start of anchor day, +1 day)
        - 'week'  : [start of anchor week (Mon), +7 days)
        - 'month' : [first day of anchor month, first day of next month)

    Args:
        anchor: Optional anchor date (defaults to today in tz_str).
        scope: 'today' | 'week' | 'month' (anything else treated as month).
        tz_str: IANA timezone for the window.

    Returns:
        (window_start, window_end, tz) — both aware datetimes in 'tz_str'.
    """
    tz = ZoneInfo(tz_str)
    now = datetime.now(tz)
    base = datetime.combine(anchor or now.date(), time(0, 0), tz)
    if scope == "today":
        start = base
        end = start + timedelta(days=1)
    elif scope == "week":
        start = base - timedelta(days=base.weekday())
        end = start + timedelta(days=7)
    else:  # "month"
        first = base.replace(day=1)
        next_month = first.replace(
            year=first.year + (1 if first.month == 12 else 0),
            month=(1 if first.month == 12 else first.month + 1),
        )
        start = first
        end = next_month
    return start, end, tz


def _intersects(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> bool:
    """
    Half-open interval overlap for '[a_start, a_end)' and '[b_start, b_end)'.

    Returns:
        True if the intervals intersect, False otherwise.
    """
    return a_start < b_end and b_start < a_end


def _segmentize(boundaries: list[datetime]) -> list[tuple[datetime, datetime]]:
    """
    Convert sorted boundary instants into adjacent segments.

    Args:
        boundaries: Timestamps (possibly containing duplicates).

    Returns:
        List of '[b[i], b[i+1])' segment tuples.
    """
    boundaries = sorted(set(boundaries))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _collect_boundaries(*iterables: list[tuple[datetime, datetime]]) -> list[datetime]:
    """
    Collect unique boundaries from multiple (start, end) iterables.

    Returns:
        Flattened list of start/end instants (may contain duplicates).
    """
    b: list[datetime] = []
    for it in iterables:
        for s, e in it:
            b.append(s)
            b.append(e)
    return b


def _expand_weekly_rules(
    rules: list[AvailabilityRule],
    window_start: datetime,
    window_end: datetime,
    tz: ZoneInfo,
) -> list[tuple[datetime, datetime, AvailabilityRule]]:
    """
    Expand weekly rules into concrete UTC openings clipped to the window.

    Args:
        rules: Weekly 'AvailabilityRule' rows for the owner.
        window_start, window_end: Window boundaries (aware).
        tz: Owner timezone.

    Returns:
        List of '(start_utc, end_utc, rule)' tuples.
    """
    out: list[tuple[datetime, datetime, AvailabilityRule]] = []
    days = (window_end.date() - window_start.date()).days
    for d in range(days):
        day_start = (window_start + timedelta(days=d)).astimezone(tz)
        weekday = day_start.weekday()
        for r in rules:
            if getattr(r, "active", True) is False:
                continue
            slot_len = int(getattr(r, "slot_minutes", 0) or 0)
            if slot_len <= 0:
                continue
            if r.weekday != weekday:
                continue
            start_local = datetime.combine(day_start.date(), r.start_local, tz)
            end_local = datetime.combine(day_start.date(), r.end_local, tz)
            s = max(start_local, window_start.astimezone(tz)).astimezone(UTC)
            e = min(end_local, window_end.astimezone(tz)).astimezone(UTC)
            if s < e:
                out.append((s, e, r))
    return out


def _collect_time_off(
    offs: list[TimeOff], window_start: datetime, window_end: datetime
) -> list[tuple[datetime, datetime, TimeOff]]:
    """
    Collect time-off windows intersecting the provided window.

    Args:
        offs: TimeOff rows.
        window_start, window_end: Window boundaries (aware).

    Returns:
        List of '(start_utc, end_utc, timeoff)' tuples clipped to the window.
    """
    out = []
    for t in offs:
        s = max(_as_aware_utc(t.start_utc), _as_aware_utc(window_start))
        e = min(_as_aware_utc(t.end_utc), _as_aware_utc(window_end))
        if s < e:
            out.append((s, e, t))
    return out


def _apply_precedence(
    weekly: list[tuple[datetime, datetime, AvailabilityRule]],
    special: list[tuple[datetime, datetime, Any]],
    offs: list[tuple[datetime, datetime, TimeOff]],
    appointments: list[tuple[datetime, datetime, Appointment]],
    *,
    appt_edge_buffer_min: int = 5,
) -> tuple[
    list[tuple[datetime, datetime]],
    list[tuple[datetime, datetime]],
    list[tuple[datetime, datetime]],
]:
    """
    Resolve final openings/time-off after applying precedence and appointment buffers.

    Precedence:
        1) Time Off beats openings.
        2) Openings = union(weekly, specials).
        3) Subtract ACTIVE appointments (padded by 'appt_edge_buffer_min') from openings.

    Args:
        weekly: Weekly openings (UTC).
        special: Special opening windows (UTC).
        offs: Time off windows (UTC).
        appointments: Appointment windows (UTC).
        appt_edge_buffer_min: Minutes to expand appointment windows when subtracting.

    Returns:
        (openings, time_off, appt_spans), each a list of UTC '(start, end)' tuples.
    """
    cand: list[tuple[datetime, datetime, str]] = []
    for s, e, _r in weekly:
        cand.append((s, e, "opening"))
    for s, e, _o in special:
        cand.append((s, e, "opening"))
    for s, e, _t in offs:
        cand.append((s, e, "time_off"))

    appt_spans = [(s, e) for (s, e, _a) in appointments]

    boundaries = _collect_boundaries([(s, e) for (s, e, _typ) in cand], appt_spans)
    if not boundaries:
        return [], [(s, e) for (s, e, _t) in offs], appt_spans

    segments = _segmentize(boundaries)

    final_openings: list[tuple[datetime, datetime]] = []
    final_time_off: list[tuple[datetime, datetime]] = []

    for seg_start, seg_end in segments:
        overlapping_types = {
            typ for (s, e, typ) in cand if _intersects(s, e, seg_start, seg_end)
        }
        if not overlapping_types:
            continue
        if "time_off" in overlapping_types:
            final_time_off.append((seg_start, seg_end))
        elif "opening" in overlapping_types:
            final_openings.append((seg_start, seg_end))

    from datetime import timedelta as _td

    pad = _td(minutes=int(appt_edge_buffer_min or 0))

    def _subtract(
        spans: list[tuple[datetime, datetime]], cuts: list[tuple[datetime, datetime]]
    ) -> list[tuple[datetime, datetime]]:
        """
        Subtract 'cuts' from 'spans' using half-open interval arithmetic.

        Expands 'cuts' by 'pad' on both sides before subtraction.

        Returns:
            The list of remaining spans (non-empty intervals only).
        """
        if not spans or not cuts:
            return spans[:]
        result: list[tuple[datetime, datetime]] = []
        padded = [(cs - pad, ce + pad) for cs, ce in cuts]
        for s, e in spans:
            cur = [(s, e)]
            for cs, ce in padded:
                nxt: list[tuple[datetime, datetime]] = []
                for xs, xe in cur:
                    if not _intersects(xs, xe, cs, ce):
                        nxt.append((xs, xe))
                    else:
                        if xs < cs:
                            nxt.append((xs, cs))
                        if ce < xe:
                            nxt.append((ce, xe))
                cur = nxt
            result.extend([pair for pair in cur if pair[0] < pair[1]])
        return result

    final_openings = _subtract(final_openings, appt_spans)
    return final_openings, final_time_off, appt_spans


def owner_calendar_snapshot(
    db: Session, owner_id: str | int, scope: str, anchor: Optional[date], tz_str: str
) -> Dict[str, Any]:
    """
    Build a calendar snapshot for an owner with openings, time off, and events.

    Data sources:
        - Weekly 'AvailabilityRule'
        - 'SpecialOpening'
        - 'TimeOff'
        - 'Appointment' (booked/completed/canceled; only ACTIVE reduce openings)

    Args:
        db: Active session.
        owner_id: Owner id (string or int).
        scope: 'today' | 'week' | 'month'.
        anchor: Optional anchor date (defaults to "now" in tz_str).
        tz_str: Owner/display timezone for the snapshot.

    Returns:
        Dict with keys: {'tz', 'start', 'end', 'events'} where events include
        openings, time_off, and appointment entries.
    """

    window_start, window_end, tz = _normalize_window(anchor, scope, tz_str)
    rules = (
        db.execute(
            select(AvailabilityRule).where(AvailabilityRule.owner_id == owner_id)
        )
        .scalars()
        .all()
    )
    owner = db.query(User).filter(User.id == owner_id).first()

    offs = (
        db.execute(
            select(TimeOff).where(
                TimeOff.owner_id == owner_id,
                or_(
                    and_(
                        TimeOff.start_utc >= window_start.astimezone(UTC),
                        TimeOff.start_utc < window_end.astimezone(UTC),
                    ),
                    and_(
                        TimeOff.end_utc > window_start.astimezone(UTC),
                        TimeOff.end_utc <= window_end.astimezone(UTC),
                    ),
                    and_(
                        TimeOff.start_utc <= window_start.astimezone(UTC),
                        TimeOff.end_utc >= window_end.astimezone(UTC),
                    ),
                ),
            )
        )
        .scalars()
        .all()
    )

    appt_rows = db.execute(
        select(Appointment, Person)
        .outerjoin(Person, Appointment.person_id == Person.id)
        .where(
            Appointment.owner_id == owner_id,
            Appointment.status.in_(["booked", "completed", "canceled"]),
            or_(
                and_(
                    Appointment.start_utc >= window_start.astimezone(UTC),
                    Appointment.start_utc < window_end.astimezone(UTC),
                ),
                and_(
                    Appointment.end_utc > window_start.astimezone(UTC),
                    Appointment.end_utc <= window_end.astimezone(UTC),
                ),
                and_(
                    Appointment.start_utc <= window_start.astimezone(UTC),
                    Appointment.end_utc >= window_end.astimezone(UTC),
                ),
            ),
        )
        .order_by(Appointment.start_utc.asc())
    ).all()

    appt_blocks = [
        (a.start_utc, a.end_utc, a)
        for (a, _p) in appt_rows
        if getattr(a, "status", None) in ACTIVE_APPT_STATUSES
    ]

    specials = (
        db.execute(
            select(SpecialOpening).where(
                SpecialOpening.owner_id == owner_id,
                or_(
                    and_(
                        SpecialOpening.start_utc >= window_start.astimezone(UTC),
                        SpecialOpening.start_utc < window_end.astimezone(UTC),
                    ),
                    and_(
                        SpecialOpening.end_utc > window_start.astimezone(UTC),
                        SpecialOpening.end_utc <= window_end.astimezone(UTC),
                    ),
                    and_(
                        SpecialOpening.start_utc <= window_start.astimezone(UTC),
                        SpecialOpening.end_utc >= window_end.astimezone(UTC),
                    ),
                ),
            )
        )
        .scalars()
        .all()
    )

    special_blocks = [
        (_as_aware_utc(s.start_utc), _as_aware_utc(s.end_utc), s) for s in specials
    ]
    weekly_blocks = _expand_weekly_rules(rules, window_start, window_end, tz)
    off_blocks = _collect_time_off(offs, window_start, window_end)

    try:
        from services.features import get_owner_flag

        if get_owner_flag(
            str(owner_id), "no_edge_buffer", "FEATURE_NO_EDGE_BUFFER", default=True
        ):
            appt_edge_buffer_min = 0
        else:
            appt_edge_buffer_min = int(
                getattr(owner, "appt_edge_buffer_min", DEFAULT_APPT_EDGE_BUFFER_MIN)
                or DEFAULT_APPT_EDGE_BUFFER_MIN
            )
    except Exception:
        appt_edge_buffer_min = int(
            getattr(owner, "appt_edge_buffer_min", DEFAULT_APPT_EDGE_BUFFER_MIN)
            or DEFAULT_APPT_EDGE_BUFFER_MIN
        )

    final_openings, final_off, _appt_spans = _apply_precedence(
        weekly_blocks,
        special_blocks,
        off_blocks,
        appt_blocks,
        appt_edge_buffer_min=appt_edge_buffer_min,
    )

    def _mk_event(
        ev_id: str,
        typ: str,
        title: str,
        s: datetime,
        e: datetime,
        status: Optional[str] = None,
        meta: Optional[dict] = None,
    ):
        return dict(
            id=ev_id,
            type=typ,
            title=title,
            start=s,
            end=e,
            status=status,
            meta=meta or {},
        )

    events: list[dict[str, Any]] = []
    for i, (s, e) in enumerate(final_openings):
        events.append(_mk_event(f"open-{i}", "opening", "Opening", s, e))
    for i, (s, e) in enumerate(final_off):
        events.append(_mk_event(f"off-{i}", "time_off", "Time Off", s, e))

    for a, p in appt_rows:
        person_name = (
            (getattr(p, "full_name", None) if p else None)
            or (getattr(p, "name", None) if p else None)
            or (
                " ".join(
                    x
                    for x in [
                        getattr(p, "first_name", None),
                        getattr(p, "last_name", None),
                    ]
                    if x
                )
                if p
                else None
            )
            or getattr(a, "client_name", None)
        )
        person_id = getattr(p, "id", None) if p else None
        title = (
            person_name
            or getattr(a, "client_name", None)
            or f"Appt {getattr(a, 'id', '')}"
        )
        meta = {
            "person_id": str(person_id) if person_id else None,
            "person_name": person_name,
            "client_name": getattr(a, "client_name", None),
            "client_email": getattr(a, "client_email", None),
            "service_name": getattr(a, "service_name", None),
        }
        events.append(
            _mk_event(
                f"appt-{a.id}",
                "appointment",
                title,
                a.start_utc,
                a.end_utc,
                status=a.status,
                meta=meta,
            )
        )

    return {"tz": tz_str, "start": window_start, "end": window_end, "events": events}


# ---------------------------------------------------------------------
# SERVICE OPTIONS / LISTS
# ---------------------------------------------------------------------
def list_service_options(db: Session, owner_id: str) -> list[ServiceOption]:
    """
    Return all active 'ServiceOption' rows for an owner ordered by duration.

    Args:
        db: Active session.
        owner_id: Owner id.

    Returns:
        List of ServiceOption ORM rows.
    """
    return (
        db.query(ServiceOption)
        .filter(ServiceOption.owner_id == str(owner_id), ServiceOption.is_active == 1)
        .order_by(ServiceOption.duration_minutes.asc())
        .all()
    )


def list_owner_appointments(
    db: Session, owner_id: str | int, flt: str, tz_str: str
) -> list[Dict[str, Any]]:
    """
    List appointments for an owner filtered by a bucket.

    Supported filters:
        - 'today', 'this_week', 'this_month'
        - 'cancelled'
        - 'completed_last_week', 'completed_last_month', 'completed_all_time'

    Args:
        db: Active session.
        owner_id: Owner id.
        flt: Filter bucket key.
        tz_str: IANA timezone for interpreting "today/week/month".

    Returns:
        List of dictionaries for UI consumption.
    """
    tz = ZoneInfo(tz_str)
    now = datetime.now(tz)
    now_utc = datetime.now(UTC)

    # Normalize statuses based on time: past → completed, future/upcoming → booked.
    try:
        past_q = db.query(Appointment).filter(
            Appointment.owner_id == owner_id,
            Appointment.status.notin_(["completed", "canceled"]),
            Appointment.end_utc < now_utc,
        )
        future_q = db.query(Appointment).filter(
            Appointment.owner_id == owner_id,
            Appointment.status.notin_(["canceled", "completed"]),
            Appointment.start_utc >= now_utc,
        )
        past_updated = past_q.update(
            {Appointment.status: "completed"}, synchronize_session="fetch"
        )
        future_updated = future_q.update(
            {Appointment.status: "booked"}, synchronize_session="fetch"
        )
        if past_updated or future_updated:
            db.commit()
    except Exception:
        db.rollback()

    def start_of_week(dt: datetime) -> datetime:
        return (dt - timedelta(days=dt.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def start_of_month(dt: datetime) -> datetime:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    q = select(Appointment).where(Appointment.owner_id == owner_id)
    if flt == "today":
        s = now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
        e = s + timedelta(days=1)
        q = q.where(Appointment.start_utc >= s, Appointment.start_utc < e)
    elif flt == "this_week":
        s = start_of_week(now).astimezone(UTC)
        e = s + timedelta(days=7)
        q = q.where(Appointment.start_utc >= s, Appointment.start_utc < e)
    elif flt == "this_month":
        s = start_of_month(now).astimezone(UTC)
        e = (
            s.replace(month=1, year=s.year + 1)
            if s.month == 12
            else s.replace(month=s.month + 1)
        )
        q = q.where(Appointment.start_utc >= s, Appointment.start_utc < e)
    elif flt == "cancelled":
        q = q.where(Appointment.status == "canceled")
    elif flt == "completed_last_week":
        end_week = start_of_week(now)
        start_week = end_week - timedelta(days=7)
        s = start_week.astimezone(UTC)
        e = end_week.astimezone(UTC)
        q = q.where(
            Appointment.status == "completed",
            Appointment.start_utc >= s,
            Appointment.start_utc < e,
        )
    elif flt == "completed_last_month":
        start_this_month = start_of_month(now)
        start_last = (
            start_this_month.replace(year=start_this_month.year - 1, month=12)
            if start_this_month.month == 1
            else start_this_month.replace(month=start_this_month.month - 1)
        )
        s = start_last.astimezone(UTC)
        e = start_this_month.astimezone(UTC)
        q = q.where(
            Appointment.status == "completed",
            Appointment.start_utc >= s,
            Appointment.start_utc < e,
        )
    elif flt == "completed_all_time":
        q = q.where(Appointment.status == "completed")

    rows = db.execute(q.order_by(Appointment.start_utc.desc())).scalars().all()
    out: list[Dict[str, Any]] = []
    for a in rows:
        needs_edit = False
        if a.status == "completed":
            missing_attendance = getattr(a, "attendance_status", "unknown") == "unknown"
            missing_payment = getattr(
                a, "payment_status", "unpaid"
            ) == "unpaid" and getattr(a, "amount_paid_cents", None) in (None, 0)
            needs_edit = missing_attendance or missing_payment

        out.append(
            dict(
                id=str(a.id),
                title=getattr(a, "title", f"Appt {a.id}"),
                start=a.start_utc,
                end=a.end_utc,
                status=a.status,
                client_name=getattr(a, "client_name", None),
                client_email=getattr(a, "client_email", None),
                needs_edit=needs_edit,
            )
        )
    return out


# ---------------------------------------------------------------------
# BOOKING
# ---------------------------------------------------------------------
def _ensure_local(dt: datetime, owner_tz: str) -> datetime:
    """
    Ensure a datetime is localized to 'owner_tz'.

    If naive, attach 'owner_tz'. If aware, convert to 'owner_tz'.

    Args:
        dt: Datetime to normalize.
        owner_tz: IANA timezone string.

    Returns:
        Owner-local aware datetime.
    """
    tz = ZoneInfo(owner_tz or "America/Toronto")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _start_end_from_local(
    owner: User, start_local: datetime, duration_min: int
) -> Tuple[datetime, datetime]:
    """
    Convert an owner-local start time and duration to UTC start/end.

    Args:
        owner: Owner row providing timezone.
        start_local: Local start datetime (naive or aware).
        duration_min: Positive integer minutes.

    Returns:
        (start_utc, end_utc) both aware UTC.

    Raises:
        ServiceBookingError('INVALID_DURATION', ...) for non-positive duration.
    """
    if duration_min <= 0:
        raise ServiceBookingError("INVALID_DURATION", "Duration must be positive.")
    s_local = _ensure_local(start_local, owner.timezone)
    e_local = s_local + timedelta(minutes=duration_min)
    return _to_utc(s_local), _to_utc(e_local)


def _validate_duration_against_service_options(
    db: Session, owner_id: str, duration_min: int
) -> None:
    """
    Validate that the requested duration is allowed for the owner.

    If the owner has configured active 'ServiceOption' rows, the duration must be
    one of those values. If no options are configured, any positive duration
    is allowed.

    Args:
        db: Active session.
        owner_id: Owner id.
        duration_min: Requested duration minutes.

    Raises:
        ServiceBookingError('INVALID_DURATION', ...) when duration not allowed.
    """
    opts = (
        db.query(ServiceOption)
        .filter(ServiceOption.owner_id == str(owner_id), ServiceOption.is_active == 1)
        .all()
    )
    allowed = sorted({int(o.duration_minutes) for o in opts})
    if allowed and duration_min not in allowed:
        raise ServiceBookingError(
            "INVALID_DURATION",
            f"Duration {duration_min} not allowed. Allowed: {allowed}",
        )


def book_appointment(
    db: Session,
    owner_id: str,
    *,
    start_local: datetime,
    duration_min: int,
    client_name: Optional[str],
    client_email: Optional[str],
    client_query: Optional[str] = None,
    price_cents: Optional[int] = None,
    private_note: Optional[str] = None,
    create_person_if_missing: bool = False,
) -> Appointment:
    """
    Create a new appointment ensuring identity and bookability invariants.

    Steps:
        1) Validate owner and requested duration (against ServiceOptions).
        2) Convert local start/duration to UTC window.
        3) Check bookability (time off / conflicts / specials / weekly rules).
        4) Resolve Person; fallback to denormalized (name/email) if needed.
        5) Optionally create a Person under a default account when missing.
        6) Insert appointment; if legacy unique conflicts but existing row is
           'canceled' at same start, reuse that row by reviving it.
        7) Optionally set price override; commit and refresh.
        8) Auto-apply wallet funds (store credit) if available for this client.

    Args:
        db: Active session.
        owner_id: Owner id.
        start_local: Owner-local start datetime (naive/aware).
        duration_min: Duration in minutes (must be allowed).
        client_name: Optional human name for denormalized identity.
        client_email: Optional email for denormalized identity.
        client_query: Optional fuzzy name/email hint for resolving Person.
        price_cents: Optional one-off price override in cents.
        private_note: Optional private note saved on the appointment.
        create_person_if_missing: If True, create a Person when not resolvable.

    Returns:
        The created (or revived) Appointment ORM instance.

    Raises:
        ServiceBookingError with codes:
            - OWNER_NOT_FOUND
            - INVALID_DURATION
            - NO_AVAILABILITY / OVERLAP
            - MISSING_IDENTITY
            - AMBIGUOUS_PERSON
    """
    owner = db.query(User).filter(User.id == owner_id).first()
    if not owner:
        raise ServiceBookingError("OWNER_NOT_FOUND", "Owner not found.")

    _validate_duration_against_service_options(db, owner_id, duration_min)

    s_utc, e_utc = _start_end_from_local(owner, start_local, duration_min)

    ok, reason = is_owner_time_bookable(db, owner, s_utc, e_utc)
    if not ok:
        code = (
            "NO_AVAILABILITY"
            if reason
            in (
                "time off",
                "no opening covers the requested time",
                "crosses day boundary",
            )
            else "OVERLAP"
        )
        raise ServiceBookingError(code, f"Requested time is not bookable: {reason}")

    person, denorm_name, denorm_email = _resolve_person_for_owner(
        db,
        owner_id,
        client_email=client_email,
        client_name=client_name,
        client_query=client_query,
    )

    if not person and not (denorm_name and denorm_email):
        raise ServiceBookingError(
            "MISSING_IDENTITY",
            "Provide a person, or both client_name and client_email.",
        )

    if person is None and create_person_if_missing:
        acct = (
            db.query(ClientAccount)
            .filter(ClientAccount.owner_user_id == owner_id)
            .first()
        )
        if acct is None:
            acct = ClientAccount(owner_user_id=owner_id, name=None)
            db.add(acct)
            db.flush()
        person = Person(
            account_id=acct.id, full_name=denorm_name or "Client", email=denorm_email
        )
        db.add(person)
        db.flush()

    canonical_name = (person.full_name if person else None) or denorm_name or "Client"
    canonical_email = denorm_email or None
    fallback_note: Optional[str] = None

    if person:
        if not canonical_email:
            canonical_email = person.email or _account_primary_email(
                db, person.account_id
            )
        if not canonical_email and person.email:
            canonical_email = person.email
        if canonical_email and not person.email:
            fallback_note = f"Booked for {canonical_name} (using account email {canonical_email}; no direct email on file)."

    note_lines: list[str] = []
    if private_note:
        note_lines.append(private_note)
    if fallback_note and fallback_note not in note_lines:
        note_lines.append(fallback_note)

    # Explicit id for backends without gen_random_uuid (e.g., SQLite tests)
    appt_id = uuid.uuid4()
    appt = Appointment(
        id=appt_id,
        owner_id=owner_id,
        person_id=person.id if person else None,
        client_name=canonical_name,
        client_email=canonical_email,
        start_utc=s_utc,
        end_utc=e_utc,
        status="booked",
        owner_private_note="\n".join(note_lines) if note_lines else None,
    )
    # Denormalized client_id for analytics joins (if we can resolve it now)
    try:
        if person is not None:
            acct = (
                db.query(ClientAccount)
                .filter(ClientAccount.id == person.account_id)
                .first()
            )
            if acct and getattr(acct, "client_user_id", None):
                appt.client_id = acct.client_user_id
    except Exception:
        pass

    if price_cents is not None:
        appt.price_override_cents = int(price_cents)

    db.add(appt)
    try:
        db.commit()
    except IntegrityError:
        # Fallback: if a legacy unique index on (owner_id,start_utc) exists and the row at this
        # start is canceled, reuse it as the new booking rather than failing.
        db.rollback()
        exist = (
            db.query(Appointment)
            .filter(Appointment.owner_id == owner_id, Appointment.start_utc == s_utc)
            .first()
        )
        if exist and getattr(exist, "status", None) == "canceled":
            exist.end_utc = e_utc
            exist.status = "booked"
            exist.person_id = person.id if person else None
            exist.client_name = canonical_name
            exist.client_email = canonical_email
            exist.owner_private_note = "\n".join(note_lines) if note_lines else None
            exist.payment_status = "unpaid"
            exist.amount_paid_cents = 0
            exist.bundle_id = None
            db.add(exist)
            try:
                db.commit()
                try:
                    db.refresh(exist)
                except Exception:
                    pass
                return exist
            except IntegrityError:
                db.rollback()
                raise ServiceBookingError(
                    "NO_AVAILABILITY",
                    "Another appointment already exists at that exact start time.",
                )
        raise ServiceBookingError(
            "NO_AVAILABILITY",
            "Another appointment already exists at that exact start time.",
        )
    try:
        db.refresh(appt)
    except Exception:
        pass
    # Auto-apply wallet funds for this client if a wallet has balance
    try:
        client_user_id = None
        if person is not None:
            acct = (
                db.query(ClientAccount)
                .filter(ClientAccount.id == person.account_id)
                .first()
            )
            if acct and getattr(acct, "client_user_id", None):
                client_user_id = acct.client_user_id
        if not client_user_id and getattr(appt, "client_id", None):
            client_user_id = appt.client_id

        if client_user_id:
            wallet_ids = (
                db.query(PrepaidBundle.id)
                .filter(
                    PrepaidBundle.owner_id == owner_id,
                    PrepaidBundle.client_id == str(client_user_id),
                    PrepaidBundle.total_credits == 0,
                )
                .order_by(PrepaidBundle.created_at.desc())
                .all()
            )
            for (wallet_id,) in wallet_ids:
                auto_apply_wallet_funds(
                    db,
                    owner_id=str(owner_id),
                    bundle_id=int(wallet_id),
                    note_prefix="Auto-apply wallet funds after booking",
                )
                try:
                    db.refresh(appt)
                except Exception:
                    pass
    except Exception:
        db.rollback()
        try:
            db.refresh(appt)
        except Exception:
            pass
    return appt


def serialize_appointment(appt: Appointment) -> Dict[str, Any]:
    """
    Produce a compact, log/response-friendly dict for an appointment.

    Fields:
        appointment_id, start_utc, end_utc, duration_min, person_id,
        client_name, client_email, price_cents, status

    Args:
        appt: Appointment ORM instance.

    Returns:
        Dictionary with normalized fields; datetimes are aware UTC.
    """
    return dict(
        appointment_id=str(appt.id),
        start_utc=_as_aware_utc(appt.start_utc),
        end_utc=_as_aware_utc(appt.end_utc),
        duration_min=int((appt.end_utc - appt.start_utc).total_seconds() // 60),
        person_id=appt.person_id,
        client_name=getattr(appt, "client_name", None),
        client_email=getattr(appt, "client_email", None),
        price_cents=getattr(appt, "price_override_cents", None),
        status=appt.status,
    )
