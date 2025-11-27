from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from typing import Literal
from sqlalchemy import or_
import uuid

from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import User, Appointment, PrepaidBundle

from ._helpers import (
    UTC,
    uuid_str,
    send_email,
    build_appt_email,
    _confirm_required,
    _confirm_payload,
    _collect_conflicts,
    _format_conflicts,
    _collect_unique_account_recipients_for_person_ids,
)
from services.payments import get_default_price_cents
from sqlalchemy.exc import IntegrityError

router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])


@router.get("/appointments/group/{group_id}", response_model=dict)
def get_group_details(
    group_id: str,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Return group appointment details and attendee payment summaries."""
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    rows: list[Appointment] = (
        db.query(Appointment)
        .filter(Appointment.owner_id == user.sub, Appointment.group_id == gid)
        .order_by(Appointment.start_utc.asc(), Appointment.id.asc())
        .all()
    )
    if not rows:
        raise HTTPException(404, "Group not found")

    from services.payments import _service_price_map, compute_financials

    price_map = _service_price_map(db, owner_user_id=user.sub)

    def _person_name(a: Appointment) -> str:
        if a.person_id:
            from app.models import Person

            p = db.query(Person).filter(Person.id == a.person_id).first()
            if p and getattr(p, "full_name", None):
                return p.full_name
        return a.client_name or "Client"

    attendees = []
    for a in rows:
        fin = compute_financials(db, a, price_map)
        attendees.append(
            {
                "appointment_id": str(a.id),
                "person_id": a.person_id,
                "name": _person_name(a),
                "status": a.status,
                "payment_status": fin.get("payment_status"),
                "price_cents": fin.get("price_cents"),
                "paid_cash_cents": fin.get("paid_cash_cents"),
                "bundle_applied_cents": fin.get("bundle_applied_cents"),
                "owed_cents": fin.get("owed_cents"),
            }
        )

    return {
        "group_id": group_id,
        "start_utc": rows[0].start_utc,
        "end_utc": rows[0].end_utc,
        "attendees": attendees,
    }


class GroupCreatePayload(BaseModel):
    """Payload to create a grouped appointment for multiple people."""

    start_local: datetime
    duration_minutes: int
    person_ids: list[int]
    status: Literal["booked", "completed", "canceled"] = "booked"
    allow_override: bool = False
    confirm_if_conflicts: bool = False
    message: str | None = None


@router.post("/appointments/admin-create-group", response_model=dict)
def admin_create_group(
    payload: GroupCreatePayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    """Create a grouped appointment across the provided 'person_ids'.
    Applies conflict checks unless 'allow_override' is true; when conflicts are
    found and 'confirm_if_conflicts' is false, returns a '409 CONFIRM_REQUIRED'
    payload for explicit confirmation. Reuses existing single seats at the same
    time when present, normalizes prices (including an optional group price for
    60-minute sessions), and sends notifications to affected accounts.
    """
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    if not payload.person_ids:
        raise HTTPException(400, "person_ids is required")
    if payload.duration_minutes <= 0:
        raise HTTPException(400, "duration_minutes must be positive")

    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    end_local = start_local + timedelta(minutes=int(payload.duration_minutes))
    start_utc, end_utc = start_local.astimezone(UTC), end_local.astimezone(UTC)

    if not payload.allow_override:
        conflict_appts, conflict_offs = _collect_conflicts(
            db, owner.id, start_utc, end_utc
        )
        if (conflict_appts or conflict_offs) and not payload.confirm_if_conflicts:
            detail = _confirm_payload(
                human="Booking conflicts detected. Reply 'confirm' to proceed anyway, or adjust the time.",
                endpoint="/api/scheduling/appointments/admin-create-group",
                method="POST",
                body={
                    **payload.model_dump(mode="json"),
                    "start_local": start_local.isoformat(timespec="minutes"),
                    "confirm_if_conflicts": True,
                },
                conflicts=_format_conflicts(owner, conflict_appts, conflict_offs),
            )
            raise _confirm_required(detail)

    requested_ids: set[int] = {int(pid) for pid in payload.person_ids}
    try:
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
        )

        valid_person_rows = (
            db.query(PersonModel.id)
            .join(ClientAccountModel, ClientAccountModel.id == PersonModel.account_id)
            .filter(
                PersonModel.id.in_(requested_ids),
                ClientAccountModel.owner_user_id == owner.id,
                ClientAccountModel.deleted_at.is_(None),
            )
            .all()
        )
        valid_ids = {int(pid) for (pid,) in valid_person_rows}
        invalid_ids = sorted(list(requested_ids - valid_ids))
        if invalid_ids:
            raise HTTPException(
                400,
                detail={
                    "error": "INVALID_PERSON_IDS",
                    "message": "Some attendees are not found or not associated with your account.",
                    "invalid_person_ids": invalid_ids,
                },
            )
    except HTTPException:
        raise
    except Exception:
        pass

    existing = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == owner.id,
            Appointment.start_utc == start_utc,
            Appointment.end_utc == end_utc,
            Appointment.status != "canceled",
            Appointment.person_id.in_(requested_ids),
        )
        .all()
    )

    group_uuid = None
    for r in existing:
        if getattr(r, "group_id", None):
            group_uuid = r.group_id
            break
    if group_uuid is None:
        group_uuid = uuid.UUID(uuid_str())

    attached_count = 0
    appt_ids: list[str] = []
    success_pids: set[int] = set()

    for r in existing:
        if r.group_id != group_uuid:
            r.group_id = group_uuid
        gp = getattr(owner, "group_price_60_cents", None)
        if int(payload.duration_minutes) == 60 and gp is not None and int(gp) > 0:
            r.price_override_cents = int(gp)
        else:
            r.price_override_cents = get_default_price_cents(
                db,
                owner_user_id=owner.id,
                duration_minutes=int(payload.duration_minutes),
            )
        r.status = payload.status
        r.payment_status = "unpaid"
        r.amount_paid_cents = r.amount_paid_cents or 0
        db.add(r)
        appt_ids.append(str(r.id))
        attached_count += 1
        if getattr(r, "person_id", None) is not None:
            success_pids.add(int(r.person_id))

    created = 0
    for pid in requested_ids - {
        int(getattr(r, "person_id"))
        for r in existing
        if getattr(r, "person_id", None) is not None
    }:
        a = Appointment(
            id=uuid.uuid4(),
            owner_id=owner.id,
            person_id=int(pid),
            start_utc=start_utc,
            end_utc=end_utc,
            status=payload.status,
            group_id=group_uuid,
        )
        try:
            from app.models import (
                Person as PersonModel,
                ClientAccount as ClientAccountModel,
            )

            person_row = (
                db.query(PersonModel).filter(PersonModel.id == int(pid)).first()
            )
            acct_row = None
            user_row = None
            if person_row:
                from app.models import ClientAccount as ClientAccountModel

                acct_row = (
                    db.query(ClientAccountModel)
                    .filter(
                        ClientAccountModel.id == int(person_row.account_id),
                        ClientAccountModel.owner_user_id == owner.id,
                        ClientAccountModel.deleted_at.is_(None),
                    )
                    .first()
                )
            if acct_row and getattr(acct_row, "client_user_id", None):
                user_row = (
                    db.query(User).filter(User.id == acct_row.client_user_id).first()
                )
            if user_row:
                a.client_id = user_row.id
                a.client_name = getattr(user_row, "name", None) or getattr(
                    acct_row, "name", None
                )
                a.client_email = getattr(user_row, "email", None) or getattr(
                    person_row, "email", None
                )
            else:
                a.client_name = getattr(person_row, "full_name", None) or getattr(
                    acct_row, "name", None
                )
                a.client_email = getattr(person_row, "email", None)
        except Exception:
            pass
        gp = getattr(owner, "group_price_60_cents", None)
        if int(payload.duration_minutes) == 60 and gp is not None and int(gp) > 0:
            a.price_override_cents = int(gp)
        else:
            a.price_override_cents = get_default_price_cents(
                db,
                owner_user_id=owner.id,
                duration_minutes=int(payload.duration_minutes),
            )
        a.payment_status = "unpaid"
        a.amount_paid_cents = a.amount_paid_cents or 0
        db.add(a)
        try:
            db.flush()
            created += 1
            appt_ids.append(str(a.id))
            success_pids.add(int(pid))
        except IntegrityError:
            db.rollback()
            exist_row = (
                db.query(Appointment)
                .filter(
                    Appointment.owner_id == owner.id,
                    Appointment.start_utc == start_utc,
                    Appointment.status != "canceled",
                    Appointment.person_id == int(pid),
                )
                .first()
            )
            if exist_row:
                if exist_row.group_id != group_uuid:
                    exist_row.group_id = group_uuid
                exist_row.end_utc = end_utc
                gp2 = getattr(owner, "group_price_60_cents", None)
                if (
                    int(payload.duration_minutes) == 60
                    and gp2 is not None
                    and int(gp2) > 0
                ):
                    exist_row.price_override_cents = int(gp2)
                else:
                    exist_row.price_override_cents = get_default_price_cents(
                        db,
                        owner_user_id=owner.id,
                        duration_minutes=int(payload.duration_minutes),
                    )
                exist_row.status = payload.status
                exist_row.payment_status = "unpaid"
                exist_row.amount_paid_cents = exist_row.amount_paid_cents or 0
                db.add(exist_row)
                try:
                    db.flush()
                    attached_count += 1
                    appt_ids.append(str(exist_row.id))
                    if getattr(exist_row, "person_id", None) is not None:
                        success_pids.add(int(exist_row.person_id))
                except Exception:
                    db.rollback()
                    continue
            else:
                continue
        except Exception:
            db.rollback()
            continue
    db.commit()

    try:
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
        )

        persons = (
            db.query(PersonModel).filter(PersonModel.id.in_(payload.person_ids)).all()
        )
        acct_ids = {
            int(p.account_id) for p in persons if getattr(p, "account_id", None)
        }
        if acct_ids:
            accounts = (
                db.query(ClientAccountModel)
                .filter(
                    ClientAccountModel.id.in_(acct_ids),
                    ClientAccountModel.owner_user_id == owner.id,
                    ClientAccountModel.deleted_at.is_(None),
                )
                .all()
            )
            client_user_ids = {
                str(a.client_user_id)
                for a in accounts
                if getattr(a, "client_user_id", None)
            }
            if client_user_ids:
                wallet_rows = (
                    db.query(PrepaidBundle.id)
                    .filter(
                        PrepaidBundle.owner_id == owner.id,
                        PrepaidBundle.client_id.in_(client_user_ids),
                        PrepaidBundle.total_credits == 0,
                    )
                    .order_by(PrepaidBundle.created_at.desc())
                    .all()
                )
                for (wid,) in wallet_rows:
                    from services.wallets import auto_apply_wallet_funds

                    auto_apply_wallet_funds(
                        db,
                        owner_id=str(owner.id),
                        bundle_id=int(wid),
                        note_prefix="Auto-apply wallet funds after group booking",
                    )
    except Exception:
        pass

    total_added = attached_count + created
    requested_total = len(requested_ids)
    if total_added < requested_total:
        missing_ids = sorted(list(requested_ids - success_pids))
        raise HTTPException(
            409,
            detail={
                "error": "PARTIAL_GROUP_CREATE",
                "message": (
                    "Could not add all attendees. Possible causes: (1) database still has a legacy "
                    "unique index on (owner_id, start_utc), or (2) one or more person_ids are invalid/"
                    "not associated with your account."
                ),
                "hint": "Ensure only ix_owner_start_person_active_unique exists and that all person_ids belong to your clients.",
                "added_person_ids": sorted(list(success_pids)),
                "missing_person_ids": missing_ids,
            },
        )

    try:
        recips = _collect_unique_account_recipients_for_person_ids(
            db, owner.id, list(success_pids)
        )
        if recips:
            for to_email, to_name in recips:
                email_pkg = build_appt_email(
                    audience="client",
                    action="created",
                    owner=owner,
                    start_local=start_local,
                    end_local=end_local,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label=payload.status,
                    recipient_name=to_name or to_email,
                    message=payload.message,
                    include_ics=True,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    email_pkg.subject,
                    email_pkg.text,
                    email_pkg.html,
                    email_pkg.ics_text,
                )
    except Exception:
        pass

    return {
        "ok": True,
        "group_id": str(group_uuid),
        "count": total_added,
        "appointment_ids": appt_ids,
    }


class GroupRecurringCreatePayload(BaseModel):
    start_local: datetime
    duration_minutes: int
    repeat_every_weeks: int = 1
    occurrences: int | None = None
    until_date: date | None = None
    person_ids: list[int]
    allow_override: bool = False
    confirm_if_conflicts: bool = False
    message: str | None = None


@router.post("/appointments/admin-create-group/recurring", response_model=dict)
def admin_create_group_recurring(
    payload: GroupRecurringCreatePayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    if not payload.person_ids:
        raise HTTPException(400, "person_ids is required")
    if payload.duration_minutes <= 0:
        raise HTTPException(400, "duration_minutes must be positive")

    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    step = timedelta(weeks=int(payload.repeat_every_weeks or 1))
    dur = timedelta(minutes=int(payload.duration_minutes))
    limit = payload.until_date

    occs: list[tuple[datetime, datetime, datetime, datetime]] = []
    cur = start_local
    count = 0
    while True:
        if limit is not None and cur.date() > limit:
            break
        end_loc = cur + dur
        occs.append((cur, end_loc, cur.astimezone(UTC), end_loc.astimezone(UTC)))
        count += 1
        if payload.occurrences is not None and count >= payload.occurrences:
            break
        if payload.occurrences is None and count >= 104:
            break
        cur = cur + step

    if not payload.allow_override and not payload.confirm_if_conflicts:
        conflicts: list[dict] = []
        for _s_loc, _e_loc, s_utc, e_utc in occs:
            a, t = _collect_conflicts(db, owner.id, s_utc, e_utc)
            if a or t:
                conflicts.append(
                    {
                        "start_local": _s_loc.isoformat(timespec="minutes"),
                        "conflicts": _format_conflicts(owner, a, t),
                    }
                )
        if conflicts:
            detail = _confirm_payload(
                human="Booking conflicts detected for one or more occurrences. Reply 'confirm' to proceed anyway.",
                endpoint="/api/scheduling/appointments/admin-create-group/recurring",
                method="POST",
                body={
                    **payload.model_dump(mode="json"),
                    "start_local": start_local.isoformat(timespec="minutes"),
                    "confirm_if_conflicts": True,
                },
                conflicts=conflicts,
            )
            raise _confirm_required(detail)

    created_groups: list[dict] = []
    try:
        from app.models import (
            Person as PersonModel,
            ClientAccount as ClientAccountModel,
        )

        requested_ids: set[int] = {int(pid) for pid in payload.person_ids}
        valid_person_rows = (
            db.query(PersonModel.id)
            .join(ClientAccountModel, ClientAccountModel.id == PersonModel.account_id)
            .filter(
                PersonModel.id.in_(requested_ids),
                ClientAccountModel.owner_user_id == owner.id,
                ClientAccountModel.deleted_at.is_(None),
            )
            .all()
        )
        valid_ids = {int(pid) for (pid,) in valid_person_rows}
        invalid_ids = sorted(list(requested_ids - valid_ids))
        if invalid_ids:
            raise HTTPException(
                400,
                detail={
                    "error": "INVALID_PERSON_IDS",
                    "message": "Some attendees are not found or not associated with your account.",
                    "invalid_person_ids": invalid_ids,
                },
            )
    except HTTPException:
        raise
    except Exception:
        pass
    for s_loc, e_loc, s_utc, e_utc in occs:
        gid = uuid.uuid4()
        per_group = 0
        for pid in payload.person_ids:
            a = Appointment(
                id=uuid.uuid4(),
                owner_id=owner.id,
                person_id=int(pid),
                start_utc=s_utc,
                end_utc=e_utc,
                status="booked",
                group_id=gid,
            )
            try:
                from app.models import (
                    Person as PersonModel,
                    ClientAccount as ClientAccountModel,
                )

                person_row = (
                    db.query(PersonModel).filter(PersonModel.id == int(pid)).first()
                )
                acct_row = None
                user_row = None
                if person_row:
                    acct_row = (
                        db.query(ClientAccountModel)
                        .filter(
                            ClientAccountModel.id == int(person_row.account_id),
                            ClientAccountModel.owner_user_id == owner.id,
                            ClientAccountModel.deleted_at.is_(None),
                        )
                        .first()
                    )
                if acct_row and getattr(acct_row, "client_user_id", None):
                    user_row = (
                        db.query(User)
                        .filter(User.id == acct_row.client_user_id)
                        .first()
                    )
                if user_row:
                    a.client_id = user_row.id
                    a.client_name = getattr(user_row, "name", None) or getattr(
                        acct_row, "name", None
                    )
                    a.client_email = getattr(user_row, "email", None) or getattr(
                        person_row, "email", None
                    )
                else:
                    a.client_name = getattr(person_row, "full_name", None) or getattr(
                        acct_row, "name", None
                    )
                    a.client_email = getattr(person_row, "email", None)
            except Exception:
                pass
            gp = getattr(owner, "group_price_60_cents", None)
            if int(payload.duration_minutes) == 60 and gp is not None and int(gp) > 0:
                a.price_override_cents = int(gp)
            else:
                a.price_override_cents = get_default_price_cents(
                    db,
                    owner_user_id=owner.id,
                    duration_minutes=int(payload.duration_minutes),
                )
            a.payment_status = "unpaid"
            a.amount_paid_cents = a.amount_paid_cents or 0
            db.add(a)
            try:
                db.flush()
                per_group += 1
            except IntegrityError:
                db.rollback()
                exist_row = (
                    db.query(Appointment)
                    .filter(
                        Appointment.owner_id == owner.id,
                        Appointment.start_utc == s_utc,
                        Appointment.status != "canceled",
                        Appointment.person_id == int(pid),
                    )
                    .first()
                )
                if exist_row:
                    if exist_row.group_id != gid:
                        exist_row.group_id = gid
                    exist_row.end_utc = e_utc
                    gp2 = getattr(owner, "group_price_60_cents", None)
                    if (
                        int(payload.duration_minutes) == 60
                        and gp2 is not None
                        and int(gp2) > 0
                    ):
                        exist_row.price_override_cents = int(gp2)
                    else:
                        exist_row.price_override_cents = get_default_price_cents(
                            db,
                            owner_user_id=owner.id,
                            duration_minutes=int(payload.duration_minutes),
                        )
                    exist_row.status = "booked"
                    exist_row.payment_status = "unpaid"
                    exist_row.amount_paid_cents = exist_row.amount_paid_cents or 0
                    db.add(exist_row)
                    try:
                        db.flush()
                        per_group += 1
                    except Exception:
                        db.rollback()
                        continue
                else:
                    continue
        if per_group > 0:
            created_groups.append({"group_id": str(gid), "count": per_group})
    db.commit()

    try:
        if created_groups:
            recips = _collect_unique_account_recipients_for_person_ids(
                db, owner.id, [int(p) for p in payload.person_ids]
            )
            owner_tz = ZoneInfo(owner.timezone)
            first_s_loc = occs[0][0].astimezone(owner_tz)
            first_e_loc = occs[0][1].astimezone(owner_tz)
            for to_email, to_name in recips:
                email_pkg = build_appt_email(
                    audience="client",
                    action="created",
                    owner=owner,
                    start_local=first_s_loc,
                    end_local=first_e_loc,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label="booked",
                    recipient_name=to_name or to_email,
                    message=payload.message,
                    include_ics=True,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    email_pkg.subject,
                    email_pkg.text,
                    email_pkg.html,
                    email_pkg.ics_text,
                )
    except Exception:
        pass

    total_requested = len(payload.person_ids) * len(created_groups)
    total_created = sum(g["count"] for g in created_groups)
    if total_created < total_requested:
        raise HTTPException(
            409,
            detail={
                "error": "PARTIAL_GROUP_CREATE",
                "message": (
                    "Could not add all attendees across recurring occurrences. Possible causes: (1) database still has a legacy "
                    "unique index on (owner_id, start_utc), or (2) one or more person_ids are invalid/not associated with your account."
                ),
                "hint": "Ensure only ix_owner_start_person_active_unique exists and that all person_ids belong to your clients.",
            },
        )
    return {"ok": True, "count": total_created, "groups": created_groups}


class GroupModifyTimePayload(BaseModel):
    start_local: datetime
    duration_minutes: int
    confirm_if_conflicts: bool = False


@router.put("/appointments/group/{group_id}/time", response_model=dict)
def admin_group_update_time(
    group_id: str,
    payload: GroupModifyTimePayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    rows = (
        db.query(Appointment)
        .filter(Appointment.owner_id == user.sub, Appointment.group_id == gid)
        .all()
    )
    if not rows:
        raise HTTPException(404, "Group not found")

    owner_tz = ZoneInfo(owner.timezone)
    start_local = (
        payload.start_local.replace(tzinfo=owner_tz)
        if payload.start_local.tzinfo is None
        else payload.start_local.astimezone(owner_tz)
    )
    end_local = start_local + timedelta(minutes=int(payload.duration_minutes))
    start_utc, end_utc = start_local.astimezone(UTC), end_local.astimezone(UTC)

    if not payload.confirm_if_conflicts:
        conflict_appts, conflict_offs = _collect_conflicts(
            db, owner.id, start_utc, end_utc
        )
        conflict_appts = [
            a for a in conflict_appts if getattr(a, "group_id", None) != gid
        ]
        if conflict_appts or conflict_offs:
            raise HTTPException(
                409,
                {
                    "human": "Conflicts detected.",
                    "conflicts": _format_conflicts(
                        owner, conflict_appts, conflict_offs
                    ),
                },
            )

    old_start_local = rows[0].start_utc.astimezone(owner_tz)
    old_end_local = rows[0].end_utc.astimezone(owner_tz)

    for a in rows:
        a.start_utc = start_utc
        a.end_utc = end_utc
        db.add(a)
    db.commit()

    try:
        person_ids = [
            int(a.person_id) for a in rows if getattr(a, "person_id", None) is not None
        ]
        recips = _collect_unique_account_recipients_for_person_ids(
            db, owner.id, person_ids
        )
        for to_email, to_name in recips:
            email_pkg = build_appt_email(
                audience="client",
                action="updated",
                owner=owner,
                start_local=start_local,
                end_local=end_local,
                appointment_id=None,
                initiator_label=owner.name or "the owner",
                status_label="booked",
                recipient_name=to_name or to_email,
                message=None,
                old_start_local=old_start_local,
                old_end_local=old_end_local,
                include_ics=True,
                organizer_email=owner.email,
                attendee_email=to_email,
            )
            background_tasks.add_task(
                send_email,
                to_email,
                email_pkg.subject,
                email_pkg.text,
                email_pkg.html,
                email_pkg.ics_text,
            )
    except Exception:
        pass

    return {"ok": True, "group_id": group_id, "updated": len(rows)}


class GroupAttendeesPayload(BaseModel):
    person_ids: list[int]
    appointment_ids: list[str] | None = None


@router.post("/appointments/group/{group_id}/attendees", response_model=dict)
def admin_group_add_attendees(
    group_id: str,
    payload: GroupAttendeesPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    dur = 60
    owner = db.query(User).filter(User.id == user.sub).first()
    if not owner:
        raise HTTPException(404, "Owner not found")
    rows = (
        db.query(Appointment)
        .filter(Appointment.owner_id == user.sub, Appointment.group_id == gid)
        .all()
    )
    if not rows:
        raise HTTPException(404, "Group not found")
    start_utc, end_utc = rows[0].start_utc, rows[0].end_utc
    existing_pids = {
        int(a.person_id) for a in rows if getattr(a, "person_id", None) is not None
    }
    added = 0
    added_pids: set[int] = set()
    for pid in payload.person_ids:
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid in existing_pids:
            continue
        existing_same_time = (
            db.query(Appointment)
            .filter(
                Appointment.owner_id == owner.id,
                Appointment.start_utc == start_utc,
                Appointment.status != "canceled",
                Appointment.person_id == int(pid),
            )
            .first()
        )
        if existing_same_time:
            if existing_same_time.group_id != gid:
                existing_same_time.group_id = gid
            existing_same_time.end_utc = end_utc
            gp = getattr(owner, "group_price_60_cents", None)
            if int(dur) == 60 and gp is not None and int(gp) > 0:
                existing_same_time.price_override_cents = int(gp)
            else:
                existing_same_time.price_override_cents = get_default_price_cents(
                    db, owner_user_id=owner.id, duration_minutes=dur
                )
            existing_same_time.payment_status = "unpaid"
            existing_same_time.amount_paid_cents = (
                existing_same_time.amount_paid_cents or 0
            )
            db.add(existing_same_time)
            try:
                db.flush()
                added += 1
                added_pids.add(int(pid))
            except Exception:
                db.rollback()
            continue

        a = Appointment(
            id=uuid.uuid4(),
            owner_id=owner.id,
            person_id=int(pid),
            start_utc=start_utc,
            end_utc=end_utc,
            status="booked",
            group_id=gid,
        )
        gp = getattr(owner, "group_price_60_cents", None)
        if int(dur) == 60 and gp is not None and int(gp) > 0:
            a.price_override_cents = int(gp)
        else:
            a.price_override_cents = get_default_price_cents(
                db, owner_user_id=owner.id, duration_minutes=dur
            )
        a.payment_status = "unpaid"
        a.amount_paid_cents = a.amount_paid_cents or 0
        db.add(a)
        try:
            db.flush()
            added += 1
            added_pids.add(int(pid))
        except IntegrityError:
            db.rollback()
            continue
    requested = {int(p) for p in payload.person_ids}
    missing = [p for p in requested if p not in existing_pids]
    if added == 0 and missing:
        raise HTTPException(
            409,
            detail=(
                "Could not add attendees. Your database likely still enforces a legacy "
                "unique index on (owner_id, start_utc) that prevents multiple attendees at the "
                "same time. Apply the Alembic migration 'sched_0007_group_lessons' to relax the "
                "constraint to (owner_id, start_utc, person_id)."
            ),
        )

    db.commit()

    try:
        if added_pids:
            recips = _collect_unique_account_recipients_for_person_ids(
                db, owner.id, list(added_pids)
            )
            owner_tz = ZoneInfo(owner.timezone)
            s_loc = start_utc.astimezone(owner_tz)
            e_loc = end_utc.astimezone(owner_tz)
            for to_email, to_name in recips:
                email_pkg = build_appt_email(
                    audience="client",
                    action="created",
                    owner=owner,
                    start_local=s_loc,
                    end_local=e_loc,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label="booked",
                    recipient_name=to_name or to_email,
                    message=None,
                    include_ics=True,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    email_pkg.subject,
                    email_pkg.text,
                    email_pkg.html,
                    email_pkg.ics_text,
                )
    except Exception:
        pass

    return {"ok": True, "added": added}


@router.delete("/appointments/group/{group_id}/attendees", response_model=dict)
def admin_group_remove_attendees(
    group_id: str,
    payload: GroupAttendeesPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    q = db.query(Appointment).filter(
        Appointment.owner_id == user.sub,
        Appointment.group_id == gid,
        Appointment.status != "canceled",
    )
    have_pids = [int(p) for p in (payload.person_ids or []) if int(p) > 0]
    have_aids = [str(a) for a in (payload.appointment_ids or []) if a]
    if have_pids and have_aids:
        q = q.filter(
            or_(Appointment.person_id.in_(have_pids), Appointment.id.in_(have_aids))
        )
    elif have_pids:
        q = q.filter(Appointment.person_id.in_(have_pids))
    elif have_aids:
        q = q.filter(Appointment.id.in_(have_aids))
    else:
        return {"ok": True, "removed": 0}
    rows = q.all()
    person_ids = [
        int(a.person_id) for a in rows if getattr(a, "person_id", None) is not None
    ]
    owner = db.query(User).filter(User.id == user.sub).first()
    owner_tz = ZoneInfo(owner.timezone) if owner else ZoneInfo("UTC")
    if rows:
        s_loc = rows[0].start_utc.astimezone(owner_tz)
        e_loc = rows[0].end_utc.astimezone(owner_tz)
    for a in rows:
        a.status = "canceled"
        db.add(a)
    db.commit()

    try:
        if person_ids and owner:
            recips = _collect_unique_account_recipients_for_person_ids(
                db, owner.id, person_ids
            )
            for to_email, to_name in recips:
                pkg = build_appt_email(
                    audience="client",
                    action="canceled",
                    owner=owner,
                    start_local=s_loc,
                    end_local=e_loc,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label="canceled",
                    recipient_name=to_name or to_email,
                    message=None,
                    include_ics=False,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    pkg.subject,
                    pkg.text,
                    pkg.html,
                    pkg.ics_text,
                )
    except Exception:
        pass

    return {"ok": True, "removed": len(rows)}


@router.put("/appointments/group/{group_id}/cancel", response_model=dict)
def admin_group_cancel(
    group_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
):
    try:
        gid = uuid.UUID(group_id)
    except Exception:
        raise HTTPException(400, "Invalid group_id")
    rows = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == user.sub,
            Appointment.group_id == gid,
            Appointment.status != "canceled",
        )
        .all()
    )
    owner = db.query(User).filter(User.id == user.sub).first()
    owner_tz = ZoneInfo(owner.timezone) if owner else ZoneInfo("UTC")
    person_ids = [
        int(a.person_id) for a in rows if getattr(a, "person_id", None) is not None
    ]
    if rows:
        s_loc = rows[0].start_utc.astimezone(owner_tz)
        e_loc = rows[0].end_utc.astimezone(owner_tz)
    for a in rows:
        a.status = "canceled"
        db.add(a)
    db.commit()

    try:
        if person_ids and owner:
            recips = _collect_unique_account_recipients_for_person_ids(
                db, owner.id, person_ids
            )
            for to_email, to_name in recips:
                pkg = build_appt_email(
                    audience="client",
                    action="canceled",
                    owner=owner,
                    start_local=s_loc,
                    end_local=e_loc,
                    appointment_id=None,
                    initiator_label=owner.name or "the owner",
                    status_label="canceled",
                    recipient_name=to_name or to_email,
                    message=None,
                    include_ics=False,
                    organizer_email=owner.email,
                    attendee_email=to_email,
                )
                background_tasks.add_task(
                    send_email,
                    to_email,
                    pkg.subject,
                    pkg.text,
                    pkg.html,
                    pkg.ics_text,
                )
    except Exception:
        pass

    return {"ok": True, "canceled": len(rows)}
