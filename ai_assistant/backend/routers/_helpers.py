from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
from fastapi import HTTPException
import json
import os
from services.emailer import send_email as _send, render_basic_html
from sqlalchemy.orm import Session
from app.models import (
    User,
    TimeOff,
    Appointment,
    RoleEnum,
    SpecialOpening,
)
from datetime import timedelta, date
from services.services_scheduling import (
    generate_daily_slots,
)
from typing import NamedTuple, Literal


UTC = ZoneInfo("UTC")


def uuid_str() -> str:
    """Return a random UUID string suitable for identifiers."""
    return str(uuid.uuid4())


def _confirm_required(detail: dict | str, status_code: int = 409) -> HTTPException:
    """Build a standardized 409 response indicating confirmation is required.

    Args:
        detail: Human-readable string or a dict payload to serialize.
        status_code: HTTP status to use (defaults to 409).

    Returns:
        'HTTPException' populated with a namespaced confirmation detail string.
    """
    payload = json.dumps(detail) if isinstance(detail, dict) else str(detail)
    return HTTPException(status_code=status_code, detail="CONFIRM_REQUIRED:" + payload)


def _confirm_payload(
    human: str,
    endpoint: str,
    method: str,
    body: dict,
    conflicts: list[str] | list[dict] | None,
) -> dict:
    """Build a structured confirmation payload for clients to act upon.

    Args:
        human: Human-friendly summary of what is being confirmed.
        endpoint: HTTP endpoint to call if confirmed.
        method: HTTP method to use (e.g., POST).
        body: Request body to submit upon confirmation.
        conflicts: Optional list of conflicts (strings or dicts) for display.

    Returns:
        Dict suitable to place into 'detail' of an HTTP 409 to drive confirm UI.
    """
    return {
        "human": human,
        "pending_http": {
            "endpoint": endpoint,
            "method": method,
            "body": body,
        },
        "conflicts": conflicts or [],
    }


def _collect_conflicts(
    db: Session, owner_id: str, start_utc: datetime, end_utc: datetime
) -> tuple[list[Appointment], list[TimeOff]]:
    """Collect overlapping appointments and time off for a window.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner to scope the search.
        start_utc: Inclusive start (UTC).
        end_utc: Exclusive end (UTC).

    Returns:
        Tuple of (appointments, time-offs) overlapping the window.
    """
    from app.models import Appointment, TimeOff

    conflict_appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner_id,
            Appointment.end_utc > start_utc,
            Appointment.start_utc < end_utc,
            Appointment.status != "canceled",
        )
        .order_by(Appointment.start_utc.asc())
        .all()
    )
    conflict_offs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner_id,
            TimeOff.end_utc > start_utc,
            TimeOff.start_utc < end_utc,
        )
        .order_by(TimeOff.start_utc.asc())
        .all()
    )
    return conflict_appts, conflict_offs


def _format_conflicts(
    owner: User, conflict_appts: list[Appointment], conflict_offs: list[TimeOff]
) -> list[str]:
    """Format conflicts as short owner-local strings for display/logging.

    Args:
        owner: Owner providing the timezone context.
        conflict_appts: Appointments overlapping a proposed window.
        conflict_offs: Time-off windows overlapping a proposed window.

    Returns:
        List of compact strings such as "Time Off: Mon Jan 01, 9:00 AM → 10:00 AM".
    """
    from zoneinfo import ZoneInfo as _ZI

    def _fmt_local_range(s_utc: datetime, e_utc: datetime, tz_name: str) -> str:
        tz = _ZI(tz_name)
        return f"{s_utc.astimezone(tz).strftime('%a %b %d, %I:%M %p').replace(' 0', ' ')} → {e_utc.astimezone(tz).strftime('%I:%M %p').lstrip('0')}"

    conflicts_human: list[str] = []
    if conflict_offs:
        conflicts_human.append(
            "Time Off: "
            + "; ".join(
                _fmt_local_range(x.start_utc, x.end_utc, owner.timezone)
                for x in conflict_offs[:5]
            )
            + (f" (+{len(conflict_offs) - 5} more)" if len(conflict_offs) > 5 else "")
        )
    if conflict_appts:
        conflicts_human.append(
            "Appointments: "
            + "; ".join(
                _fmt_local_range(x.start_utc, x.end_utc, owner.timezone)
                for x in conflict_appts[:5]
            )
            + (f" (+{len(conflict_appts) - 5} more)" if len(conflict_appts) > 5 else "")
        )
    return conflicts_human


def _collect_unique_account_recipients_for_person_ids(
    db: Session, owner_id: str, person_ids: list[int]
) -> list[tuple[str, str | None]]:
    """Return unique (email, account_name) recipients for a set of people.

    Deduplicates by client account so that grouped attendees under the same
    account receive a single email.

    Args:
        db: SQLAlchemy session.
        owner_id: Owner scope.
        person_ids: List of person IDs to collect recipients for.

    Returns:
        List of '(email, account_display_name)' tuples.
    """
    if not person_ids:
        return []
    try:
        from services.services_scheduling import _account_primary_email
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
            User as AuthUser,
        )

        rows = (
            db.query(
                PersonModel.id,
                ClientAccountModel.id,
                ClientAccountModel.name,
                ClientAccountModel.client_user_id,
            )
            .join(ClientAccountModel, ClientAccountModel.id == PersonModel.account_id)
            .filter(
                ClientAccountModel.owner_user_id == owner_id,
                ClientAccountModel.deleted_at.is_(None),
                PersonModel.id.in_([int(p) for p in person_ids]),
            )
            .all()
        )
        seen_accts: set[int] = set()
        recips: list[tuple[str, str | None]] = []
        for _pid, acct_id, acct_name, client_user_id in rows:
            if acct_id in seen_accts:
                continue
            seen_accts.add(int(acct_id))
            email = _account_primary_email(db, int(acct_id))
            if not email and client_user_id:
                u = db.query(AuthUser).filter(AuthUser.id == client_user_id).first()
                email = getattr(u, "email", None)
                if not acct_name:
                    acct_name = getattr(u, "name", None)
            if email:
                recips.append((email, acct_name))
        return recips
    except Exception:
        return []


def ics(
    uid: str,
    start: datetime,
    end: datetime,
    summary: str,
    organizer_email: str,
    attendee_email: str,
) -> str:
    """Render a minimal iCalendar VEVENT with organizer and attendee.

    Converts timestamps to UTC (per iCalendar best practices) and uses the
    compact 'YYYYMMDDTHHMMSSZ' format.

    Args:
        uid: Unique event identifier.
        start: Event start (timezone-aware).
        end: Event end (timezone-aware).
        summary: One-line summary/subject.
        organizer_email: Organizer email address (mailto:).
        attendee_email: Attendee email address.

    Returns:
        ICS text for a single VEVENT inside a VCALENDAR.
    """
    s = start.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    e = end.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//AI-Scheduler//EN\n"
        "BEGIN:VEVENT\n"
        f"UID:{uid}\nDTSTAMP:{s}\nDTSTART:{s}\nDTEND:{e}\n"
        f"SUMMARY:{summary}\n"
        f"ORGANIZER:mailto:{organizer_email or ''}\n"
        f"ATTENDEE:mailto:{attendee_email or ''}\n"
        "END:VEVENT\nEND:VCALENDAR\n"
    )


def resolve_tz(tz_param: str | None, default_tz: str) -> ZoneInfo:
    """Resolve a ZoneInfo from a request parameter or fallback string.

    Args:
        tz_param: Optional timezone name from request.
        default_tz: Fallback timezone name when 'tz_param' is missing.

    Returns:
        ZoneInfo instance for the resolved timezone.

    Raises:
        HTTPException: 400 for invalid timezone names.
    """
    tz_name = tz_param or default_tz or "America/Toronto"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid timezone: {tz_name}")


def send_email(
    to: str,
    subject: str,
    text: str,
    html: str | None = None,
    ics_text: str | None = None,
) -> None:
    """Best-effort email send wrapper that logs but does not raise on failure.

    Args:
        to: Recipient email address.
        subject: Subject line.
        text: Plaintext body.
        html: Optional HTML body.
        ics_text: Optional ICS content attachment.
    """
    try:
        _send(to=to, subject=subject, text=text, html=html, ics_text=ics_text)
    except Exception as e:
        print(f"[EMAIL ERROR] to={to} subj={subject} err={e}")


EMAIL_BRAND_LOGO_URL = os.getenv("EMAIL_BRAND_LOGO_URL", "").strip()


def _html_from_text(owner_name: str, text: str) -> str:
    """Wrap plain text into a simple branded HTML (preserves line breaks).

    Args:
        owner_name: Brand/owner display name for the header/footer.
        text: Plaintext content to wrap (supports newlines).

    Returns:
        HTML string with minimal styling and branding.
    """
    safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n", "<br/>")
    brand = owner_name or "Our Studio"
    logo = (
        f'<img src="{EMAIL_BRAND_LOGO_URL}" alt="" style="height:20px;vertical-align:middle;margin-right:10px;border-radius:4px"/>'
        if EMAIL_BRAND_LOGO_URL
        else ""
    )
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{brand}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; background:#f6f7f9; padding:24px; }}
      .card {{ max-width:680px; margin:0 auto; background:#fff; border:1px solid #e5e7eb; border-radius:14px; overflow:hidden; }}
      .hdr {{ background:#065f46; color:#ecfdf5; padding:16px 20px; font-weight:600; }}
      .bd  {{ padding:20px; color:#111827; line-height:1.6; }}
      .ft  {{ padding:16px 20px; color:#6b7280; font-size:12px; border-top:1px solid #f3f4f6; }}
      a {{ color:#065f46; }}
    </style>
  </head>
  <body>
    <div class="card">
      <div class="hdr">{logo}{brand}</div>
      <div class="bd">{safe}</div>
      <div class="ft">You’re receiving this because you’re a client of {brand}.</div>
    </div>
  </body>
</html>"""


class AppointmentEmail(NamedTuple):
    """Structured result for an appointment email render."""

    subject: str
    text: str
    html: str
    ics_text: str | None


def build_appt_email(
    *,
    audience: Literal["client", "owner"],
    action: Literal["created", "updated", "canceled"],
    owner: User,
    start_local: datetime,
    end_local: datetime,
    appointment_id: str | None,
    initiator_label: str,
    status_label: str | None = None,
    recipient_name: str | None = None,
    message: str | None = None,
    old_start_local: datetime | None = None,
    old_end_local: datetime | None = None,
    include_ics: bool = False,
    organizer_email: str | None = None,
    attendee_email: str | None = None,
) -> AppointmentEmail:
    """Compose consistent subject, text, HTML and ICS for appointment emails.

    Args:
        audience: "client" or "owner" recipient.
        action: One of "created", "updated", "canceled".
        owner: Owner record providing timezone/name.
        start_local: Appointment start in owner-local time.
        end_local: Appointment end in owner-local time.
        appointment_id: Optional appointment id for ICS UID.
        initiator_label: Actor to surface in messages (e.g., "the owner").
        status_label: Optional status summary for update emails.
        recipient_name: Optional client name for greeting.
        message: Optional message to include.
        old_start_local: Previous start time (required for updates).
        old_end_local: Previous end time (required for updates).
        include_ics: Whether to attach an ICS.
        organizer_email: Organizer for ICS.
        attendee_email: Attendee for ICS.

    Returns:
        AppointmentEmail tuple containing subject, text, HTML, and ICS text.

    Raises:
        ValueError: For unsupported actions or missing old times on updates.
    """

    tz_label = owner.timezone or start_local.tzname() or "UTC"

    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M")

    def _fmt_time(dt: datetime) -> str:
        return dt.strftime("%H:%M")

    when_line = f"When: {_fmt(start_local)}–{_fmt_time(end_local)} ({tz_label})"

    lines: list[str] = []
    actor_label = initiator_label or "the owner"

    if audience == "client" and recipient_name:
        lines.append(f"Hi {recipient_name},")
        lines.append("")

    if action == "created":
        if audience == "client":
            lines.append("Your appointment is confirmed.")
        else:
            lines.append(f"New appointment booked by {actor_label}.")
        lines.append(when_line)
    elif action == "updated":
        if old_start_local is None or old_end_local is None:
            raise ValueError("old_start_local/old_end_local required for update emails")
        if audience == "client":
            lines.append(f"Your appointment has been updated by {actor_label}.")
        else:
            lines.append(f"Appointment updated by {actor_label}.")
        lines.append("")
        lines.append(
            f"Old: {_fmt(old_start_local)} → {_fmt_time(old_end_local)} ({tz_label})"
        )
        lines.append(f"New: {_fmt(start_local)} → {_fmt_time(end_local)} ({tz_label})")
        if status_label:
            lines.append(f"Status: {status_label}")
    elif action == "canceled":
        if audience == "client":
            lines.append(f"Your appointment was cancelled by {actor_label}.")
        else:
            lines.append(f"Appointment cancelled by {actor_label}.")
        lines.append(when_line)
    else:
        raise ValueError(f"Unsupported action '{action}'")

    if message:
        heading = "Message:"
        if audience == "client":
            heading = "Message from owner:"
        elif audience == "owner":
            heading = "Message from client:"
        lines += ["", heading, message]

    title_map = {
        ("client", "created"): "Appointment Confirmed",
        ("client", "updated"): "Appointment Updated",
        ("client", "canceled"): "Appointment Cancelled",
        ("owner", "created"): "Client Appointment Booked",
        ("owner", "updated"): "Client Appointment Updated",
        ("owner", "canceled"): "Client Appointment Cancelled",
    }

    subject_prefix_map = {
        ("client", "created"): "New appointment",
        ("client", "updated"): "Appointment updated",
        ("client", "canceled"): "Appointment cancelled",
        ("owner", "created"): "New appointment",
        ("owner", "updated"): "Appointment updated",
        ("owner", "canceled"): "Appointment cancelled",
    }

    subject_prefix = subject_prefix_map[(audience, action)]
    subject = (
        f"{subject_prefix} — {start_local.strftime('%b %d, %Y %H:%M')} ({tz_label})"
    )
    title = title_map[(audience, action)]

    text = "\n".join(lines)
    html = render_basic_html(title, lines)

    ics_text: str | None = None
    if include_ics and attendee_email:
        summary = f"Appointment with {owner.name or 'our studio'}"
        uid = appointment_id or uuid_str()
        ics_text = ics(
            uid=uid,
            start=start_local,
            end=end_local,
            summary=summary,
            organizer_email=organizer_email or owner.email or "",
            attendee_email=attendee_email,
        )

    return AppointmentEmail(subject=subject, text=text, html=html, ics_text=ics_text)


def _get_owner(db: Session) -> User:
    """Return the first owner, creating a default dev owner if missing.

    The default owner is created only in no-auth/dev contexts to unblock flows.

    Args:
        db: SQLAlchemy session.

    Returns:
        'User' instance with role OWNER.
    """
    owner = (
        db.query(User)
        .filter(User.role == RoleEnum.OWNER)
        .order_by(User.createdAt.asc())
        .first()
    )
    if not owner:
        owner = User(
            id=uuid_str(),
            name="Default Owner",
            email="owner@example.com",
            role=RoleEnum.OWNER,
            timezone="America/Toronto",
            createdAt=datetime.utcnow(),
            updatedAt=datetime.utcnow(),
        )
        db.add(owner)
        db.commit()
        db.refresh(owner)
    return owner


def _compute_final_slots_for_day(
    db: Session,
    owner: User,
    day: date,
    tz_name: str,
) -> list[tuple[datetime, datetime]]:
    """Compute final bookable slots for a local day.

    Combines weekly rule slots with special openings, then subtracts time-off
    and existing appointments. Returns merged continuous windows as pairs in
    owner-local time.

    Args:
        db: SQLAlchemy session.
        owner: Owner record.
        day: The local date to evaluate.
        tz_name: Timezone name to use for local computations.

    Returns:
        List of '(start_local, end_local)' pairs representing availability.
    """
    owner_tz = ZoneInfo(tz_name)
    from types import SimpleNamespace

    weekly_pairs = generate_daily_slots(
        db=db,
        owner=SimpleNamespace(id=owner.id, timezone=tz_name),
        day=day,
        tz_override=tz_name,
    )

    day_start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=owner_tz)
    day_end_local = day_start_local + timedelta(days=1)
    day_start_utc, day_end_utc = (
        day_start_local.astimezone(UTC),
        day_end_local.astimezone(UTC),
    )

    specials = (
        db.query(SpecialOpening)
        .filter(
            SpecialOpening.owner_id == owner.id,
            SpecialOpening.end_utc > day_start_utc,
            SpecialOpening.start_utc < day_end_utc,
        )
        .all()
    )

    special_pairs: list[tuple[datetime, datetime]] = []
    try:
        from services.features import get_owner_flag

        disable_buffers = get_owner_flag(
            str(owner.id), "no_edge_buffer", "FEATURE_NO_EDGE_BUFFER", default=True
        )
    except Exception:
        disable_buffers = True

    def _split(
        start_local: datetime, end_local: datetime, slot_mins: int, buffer_mins: int
    ):
        cur = start_local
        step = timedelta(minutes=slot_mins)
        buf = timedelta(minutes=(0 if disable_buffers else buffer_mins))
        out: list[tuple[datetime, datetime]] = []
        while cur + step <= end_local:
            s = cur
            e = cur + step
            out.append((s, e))
            cur = e + buf
        return out

    for o in specials:
        s_loc = max(o.start_utc.astimezone(owner_tz), day_start_local)
        e_loc = min(o.end_utc.astimezone(owner_tz), day_end_local)
        if e_loc <= s_loc:
            continue
        special_pairs.extend(_split(s_loc, e_loc, o.slot_minutes, o.buffer_minutes))

    union = set((s.isoformat(), e.isoformat()) for s, e in weekly_pairs)
    for s, e in special_pairs:
        union.add((s.isoformat(), e.isoformat()))
    offs = (
        db.query(TimeOff)
        .filter(
            TimeOff.owner_id == owner.id,
            TimeOff.end_utc > day_start_utc,
            TimeOff.start_utc < day_end_utc,
        )
        .all()
    )
    appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.status != "canceled",
            Appointment.end_utc > day_start_utc,
            Appointment.start_utc < day_end_utc,
        )
        .all()
    )

    busy_utc: list[tuple[datetime, datetime]] = [
        (o.start_utc, o.end_utc) for o in offs
    ] + [(a.start_utc, a.end_utc) for a in appts]

    def _overlaps_utc(s_loc: datetime, e_loc: datetime) -> bool:
        s_utc, e_utc = s_loc.astimezone(UTC), e_loc.astimezone(UTC)
        for bs, be in busy_utc:
            if s_utc < be and bs < e_utc:
                return True
        return False

    final_pairs: list[tuple[datetime, datetime]] = []
    for s_iso, e_iso in sorted(union):
        s_loc = datetime.fromisoformat(s_iso)
        e_loc = datetime.fromisoformat(e_iso)
        if not _overlaps_utc(s_loc, e_loc):
            final_pairs.append((s_loc, e_loc))
    if not final_pairs:
        return []
    final_pairs.sort(key=lambda x: x[0])
    merged: list[tuple[datetime, datetime]] = []
    cur_s, cur_e = final_pairs[0]
    for s, e in final_pairs[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    return merged


def _split_slots(
    start_local: datetime, end_local: datetime, slot_minutes: int, buffer_minutes: int
):
    """Split a local window into slot-sized pairs with an optional buffer.

    Args:
        start_local: Window start in local time.
        end_local: Window end in local time.
        slot_minutes: Slot length in minutes.
        buffer_minutes: Buffer between slots in minutes.

    Returns:
        List of '(start, end)' local-time pairs.
    """
    pairs: list[tuple[datetime, datetime]] = []
    cur = start_local
    while True:
        nxt = cur + timedelta(minutes=slot_minutes)
        if nxt > end_local:
            break
        pairs.append((cur, nxt))
        cur = nxt + timedelta(minutes=buffer_minutes)
    return pairs
