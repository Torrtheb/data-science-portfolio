from __future__ import annotations
from datetime import date
from fastapi import APIRouter, Depends, Query, HTTPException, Body
from sqlalchemy.orm import Session

from app.db import get_db
from app.core.auth import require_owner
from app.schemas import OwnerPaymentsSummary
from services.payments import gather_owner_payments
from datetime import datetime, timedelta, timezone as pytimezone
from sqlalchemy import and_, or_
from app.models import Appointment, ClientAccount, Person, ServiceOption, User

router = APIRouter(prefix="/api/owner/analytics", tags=["owner-analytics"])
"""Router for owner-facing analytics and financial endpoints."""


@router.get("/payments/summary", response_model=OwnerPaymentsSummary)
def payments_summary(
    start: date = Query(..., description="Inclusive start date (YYYY-MM-DD)"),
    end: date = Query(..., description="Inclusive end date (YYYY-MM-DD)"),
    client_account_ids: list[int] | None = Query(
        None, description="Repeatable ?client_account_ids=1&client_account_ids=2"
    ),
    user=Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Aggregate payments across client accounts within a date window.

    Validates that 'end >= start' and scopes results to the authenticated owner.
    Returns totals and per-client aggregates.
    """
    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")
    payload = gather_owner_payments(
        db,
        start,
        end,
        client_account_ids=client_account_ids,
        owner_user_id=user.sub,
    )
    return payload


def _utc_bounds(start: date, end: date):
    """Convert inclusive local dates to a UTC half-open interval.

    Returns '(start_dt, end_dt)' representing '[start, end+1day)'.
    """
    s = datetime.combine(start, datetime.min.time()).replace(tzinfo=pytimezone.utc)
    e = (datetime.combine(end, datetime.min.time()) + timedelta(days=1)).replace(
        tzinfo=pytimezone.utc
    )
    return s, e


@router.get("/payments/debug")
def payments_debug(
    start: date = Query(...),
    end: date = Query(...),
    client_account_id: int | None = Query(None),
    user=Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Diagnostics for payment rollup inputs and computed totals.

    Returns each appointment row included in the window with computed financial
    fields and the same totals produced by the summary endpoint.
    """
    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")

    start_dt, end_dt = _utc_bounds(start, end)
    opts = (
        db.query(ServiceOption)
        .filter(ServiceOption.owner_id == user.sub, ServiceOption.is_active == 1)
        .all()
    )
    price_by_duration = {int(o.duration_minutes): int(o.price_cents) for o in opts}
    q = (
        db.query(Appointment, ClientAccount)
        .outerjoin(
            ClientAccount,
            and_(
                ClientAccount.client_user_id == Appointment.client_id,
                ClientAccount.owner_user_id == Appointment.owner_id,
            ),
        )
        .outerjoin(Person, Person.id == Appointment.person_id)
        .filter(
            Appointment.owner_id == user.sub,
            Appointment.status != "canceled",
            Appointment.start_utc >= start_dt,
            Appointment.start_utc < end_dt,
        )
    )

    if client_account_id is not None:
        q = q.filter(
            or_(
                ClientAccount.id == client_account_id,
                Person.account_id == client_account_id,
            )
        )

    rows = q.all()
    user_ids = {a.client_id for (a, _) in rows if a.client_id}
    users = (
        {u.id: u for u in db.query(User).filter(User.id.in_(user_ids)).all()}
        if user_ids
        else {}
    )

    debug_rows = []
    totals = {
        "appointments": 0,
        "late": 0,
        "paid_appts": 0,
        "unpaid_appts": 0,
        "total_expected_cents": 0,
        "total_paid_cents": 0,
        "total_owed_cents": 0,
        "count_by_status": {
            "paid": 0,
            "partial": 0,
            "unpaid": 0,
            "bundle": 0,
            "unknown": 0,
        },
    }

    def status_from(a, expected):
        if getattr(a, "bundle_id", None):
            return "bundle"
        ps = (getattr(a, "payment_status", None) or "").lower()
        if ps == "paid":
            return "paid"
        paid = int(a.amount_paid_cents or 0)
        if expected is None:
            return "unpaid" if paid <= 0 else "partial"
        if paid >= expected:
            return "paid"
        if paid > 0:
            return "partial"
        return "unpaid"

    for a, acct in rows:
        dur = (
            int((a.end_utc - a.start_utc).total_seconds() // 60)
            if a.start_utc and a.end_utc
            else 0
        )
        expected = price_by_duration.get(dur)
        paid = int(a.amount_paid_cents or 0)
        st = status_from(a, expected)
        try:
            p = (
                db.query(Person).filter(Person.id == a.person_id).first()
                if getattr(a, "person_id", None)
                else None
            )
            person_name = getattr(p, "full_name", None) if p else None
        except Exception:
            person_name = None
        client_label = (
            person_name
            or getattr(a, "client_name", None)
            or getattr(acct, "name", None)
            or (
                users.get(a.client_id).name
                if a.client_id and users.get(a.client_id) and users[a.client_id].name
                else None
            )
            or (
                users.get(a.client_id).email
                if a.client_id and users.get(a.client_id) and users[a.client_id].email
                else None
            )
            or "Client"
        )

        debug_rows.append(
            {
                "appointment_id": a.id,
                "client_account_id": getattr(acct, "id", 0),
                "client_label": client_label,
                "start_utc": a.start_utc,
                "end_utc": a.end_utc,
                "status_model": a.status,
                "attendance_status": getattr(a, "attendance_status", None),
                "duration_minutes": dur,
                "expected_price_cents": expected,
                "amount_paid_cents": paid,
                "payment_status_model": getattr(a, "payment_status", None),
                "computed_payment_status": st,
            }
        )
        totals["appointments"] += 1
        if getattr(a, "attendance_status", None) == "late":
            totals["late"] += 1
        if st in ("paid", "bundle"):
            totals["paid_appts"] += 1
        else:
            totals["unpaid_appts"] += 1
        totals["count_by_status"][st] = totals["count_by_status"].get(st, 0) + 1

        if expected is not None:
            owed = max(expected - paid, 0)
            totals["total_expected_cents"] += expected
            totals["total_paid_cents"] += paid
            totals["total_owed_cents"] += owed
        else:
            totals["total_paid_cents"] += paid

    return {
        "window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "start_dt_utc": start_dt.isoformat(),
            "end_dt_utc": end_dt.isoformat(),
        },
        "service_options": opts,
        "totals": totals,
        "rows": debug_rows,
    }


from app.schemas import (
    AppointmentFinancialRow,
    FinancialSummary,
    AppointmentUpdateOwner,
)
from services.payments import (
    list_owner_financial_rows,
    summarize_financial_rows,
    _sum_wallet_balance_current,
)
from services.features import get_owner_flag
from app.models import PrepaidBundle, PrepaidLedger
from sqlalchemy import func
from typing import Optional, Sequence
from fastapi import Query


@router.get("/appointments", response_model=list[AppointmentFinancialRow])
def list_financial_appointments(
    start: date = Query(..., description="YYYY-MM-DD"),
    end: date = Query(..., description="YYYY-MM-DD"),
    status: Optional[Sequence[str]] = Query(
        None, description="repeat ?status=booked&status=completed"
    ),
    payment_status: Optional[Sequence[str]] = Query(
        None, description="repeat ?payment_status=paid&payment_status=unpaid"
    ),
    client_account_id: Optional[int] = Query(None),
    user=Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Return per-appointment financial rows with optional filters."""
    if end < start:
        raise HTTPException(400, "end must be >= start")
    rows = list_owner_financial_rows(
        db=db,
        start=start,
        end=end,
        owner_user_id=user.sub,
        status=status,
        payment_status=payment_status,
        client_account_id=client_account_id,
    )
    return rows


@router.get("/summary", response_model=FinancialSummary)
def financial_summary(
    start: date = Query(..., description="YYYY-MM-DD"),
    end: date = Query(..., description="YYYY-MM-DD"),
    status: Optional[Sequence[str]] = Query(None),
    payment_status: Optional[Sequence[str]] = Query(None),
    client_account_id: Optional[int] = Query(None),
    user=Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Return totals across filtered financial rows with wallet options.

    Applies owner scope and optional status/payment filters. Includes an option
    to treat wallet deposits as paid revenue and always includes current wallet
    balance for context.
    """
    if end < start:
        raise HTTPException(400, "end must be >= start")
    rows = list_owner_financial_rows(
        db=db,
        start=start,
        end=end,
        owner_user_id=user.sub,
        status=status,
        payment_status=payment_status,
        client_account_id=client_account_id,
    )
    totals = summarize_financial_rows(rows)
    if get_owner_flag(
        user.sub,
        "wallet_deposits_as_paid",
        "FEATURE_WALLET_DEPOSITS_AS_PAID",
        default=False,
    ):
        start_dt, end_dt = _utc_bounds(start, end)
        dep_q = (
            db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
            .join(PrepaidBundle, PrepaidBundle.id == PrepaidLedger.bundle_id)
            .filter(
                PrepaidBundle.owner_id == user.sub,
                PrepaidBundle.total_credits == 0,
                PrepaidLedger.amount_cents > 0,
                PrepaidLedger.created_at >= start_dt,
                PrepaidLedger.created_at < end_dt,
            )
        )
        if client_account_id is not None:
            from app.models import ClientAccount as _CA

            dep_q = dep_q.join(
                _CA,
                and_(
                    _CA.client_user_id == PrepaidBundle.client_id,
                    _CA.owner_user_id == PrepaidBundle.owner_id,
                ),
            )
            dep_q = dep_q.filter(_CA.id == int(client_account_id))
        try:
            deposits = int(dep_q.scalar() or 0)
        except Exception:
            deposits = 0
        totals["total_bundle_cents"] = deposits
        totals["total_paid_cents"] = int(totals.get("total_cash_cents", 0)) + deposits
    try:
        acct_ids = [int(client_account_id)] if client_account_id is not None else None
        totals["total_wallet_balance_cents"] = _sum_wallet_balance_current(
            db, owner_user_id=user.sub, client_account_ids=acct_ids
        )
    except Exception:
        totals["total_wallet_balance_cents"] = 0
    return totals


@router.post("/cleanup-canceled", response_model=dict)
def cleanup_canceled_payments(
    dry_run: bool = Query(True, description="If true, preview changes without writing"),
    db: Session = Depends(get_db),
    user=Depends(require_owner),
):
    """Cleanup payment fields for canceled appointments and restore wallet funds.

    For each canceled appointment owned by the current owner:
      - If any PrepaidLedger rows exist with amount_cents < 0 for this appointment, add a matching
        restore entry (amount_cents > 0) per bundle to net to zero.
      - Set amount_paid_cents = 0, payment_status = 'unpaid', paid_at = NULL.

    Returns a summary and (in dry_run) the list of affected appointment ids.
    """
    appts = (
        db.query(Appointment)
        .filter(
            Appointment.owner_id == user.sub,
            Appointment.status == "canceled",
        )
        .all()
    )

    affected = []
    restored_total = 0
    for appt in appts:
        rows = (
            db.query(
                PrepaidLedger.bundle_id,
                func.coalesce(func.sum(PrepaidLedger.amount_cents), 0).label("sum"),
            )
            .filter(PrepaidLedger.appointment_id == appt.id)
            .group_by(PrepaidLedger.bundle_id)
            .all()
        )
        will_change = False
        for bid, s in rows:
            s = int(s or 0)
            if s < 0 and bid:
                will_change = True
                if not dry_run:
                    db.add(
                        PrepaidLedger(
                            bundle_id=int(bid),
                            event="restore",
                            delta_credits=0,
                            amount_cents=+(-s),
                            appointment_id=appt.id,
                            note="Cleanup canceled appointment funds",
                        )
                    )
                restored_total += -s

        if (getattr(appt, "amount_paid_cents", 0) or 0) != 0 or will_change:
            affected.append(str(appt.id))
            if not dry_run:
                appt.amount_paid_cents = 0
                appt.payment_status = "unpaid"
                if hasattr(appt, "paid_at"):
                    appt.paid_at = None
                db.add(appt)

    if not dry_run and affected:
        db.commit()

    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "affected_appointments": affected,
        "restored_wallet_cents": int(restored_total),
        "count": len(affected),
    }


@router.put("/appointments/{appt_id}", response_model=dict)
def update_financials(
    appt_id: str,
    payload: AppointmentUpdateOwner = Body(...),
    user=Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Update limited financial fields for a single appointment.

    Editable fields: 'payment_status', 'amount_paid_cents', 'price_override_cents',
    'paid_at', and 'owner_private_note'. On 'refunded', normalizes amounts to $0.

    Raises:
        HTTPException: 404 if the appointment is not found for this owner.
    """
    appt = (
        db.query(Appointment)
        .filter(Appointment.id == appt_id, Appointment.owner_id == user.sub)
        .first()
    )
    if not appt:
        raise HTTPException(404, "Appointment not found")
    if payload.payment_status is not None:
        appt.payment_status = payload.payment_status
        if appt.payment_status == "refunded":
            try:
                appt.amount_paid_cents = 0
            except Exception:
                pass
            try:
                appt.price_override_cents = 0
            except Exception:
                pass
    if payload.amount_paid_cents is not None:
        appt.amount_paid_cents = int(payload.amount_paid_cents)
    if payload.price_override_cents is not None:
        appt.price_override_cents = int(payload.price_override_cents)
    if payload.owner_private_note is not None:
        appt.owner_private_note = payload.owner_private_note
    if payload.paid_at is not None:
        appt.paid_at = payload.paid_at

    db.add(appt)
    db.commit()
    db.refresh(appt)
    return {"ok": True}
