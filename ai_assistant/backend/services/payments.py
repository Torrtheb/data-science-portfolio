from __future__ import annotations
from datetime import datetime, date, timedelta, timezone as pytimezone
from typing import Dict, Optional, Tuple, Literal, List, Any, Sequence
from sqlalchemy.orm import Session
from services.features import get_owner_flag
from sqlalchemy import and_, or_, func

from app.models import (
    Appointment,
    ClientAccount,
    PrepaidBundle,
    PrepaidLedger,
    Person,
    ServiceOption,
    User,
)

PaymentStatus = Literal["paid", "partial", "unpaid", "bundle", "unknown"]


def _daterange_to_utc(start: date, end: date) -> Tuple[datetime, datetime]:
    """Convert an inclusive date range to UTC datetime bounds.

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        Tuple of '(start_dt, end_dt)' representing a half-open interval in UTC
        where 'start_dt' is midnight at 'start' and 'end_dt' is midnight of the
        day after 'end'.
    """
    start_dt = datetime.combine(start, datetime.min.time()).replace(
        tzinfo=pytimezone.utc
    )
    end_dt = (datetime.combine(end, datetime.min.time()) + timedelta(days=1)).replace(
        tzinfo=pytimezone.utc
    )
    return start_dt, end_dt


def _service_price_map(
    db: Session, owner_user_id: Optional[str] = None
) -> Dict[int, int]:
    """Build a duration→price map from active service options.

    Args:
        db: SQLAlchemy session.
        owner_user_id: Optional owner scope; when provided restricts options.

    Returns:
        Mapping '{duration_minutes: price_cents}'.
    """
    q = db.query(ServiceOption).filter(ServiceOption.is_active == 1)
    if owner_user_id:
        q = q.filter(ServiceOption.owner_id == owner_user_id)
    rows = q.all()
    return {int(r.duration_minutes): int(r.price_cents) for r in rows}


def get_default_price_cents(
    db: Session, owner_user_id: str, duration_minutes: int
) -> int:
    """Fetch the configured price for a duration under an owner.

    Args:
        db: SQLAlchemy session.
        owner_user_id: Owner whose options to search.
        duration_minutes: Duration in minutes.

    Returns:
        Price in cents if configured and active, else 0.
    """
    row = (
        db.query(ServiceOption)
        .filter(
            ServiceOption.owner_id == owner_user_id,
            ServiceOption.duration_minutes == int(duration_minutes),
            ServiceOption.is_active == 1,
        )
        .first()
    )
    return int(row.price_cents) if row else 0


def expected_price_cents(appt: Appointment, price_map: Dict[int, int]) -> Optional[int]:
    """Derive expected price for an appointment.

    Prefers per-appointment override; otherwise computes duration (minutes)
    and looks up the value in 'price_map'.

    Args:
        appt: Appointment instance.
        price_map: Mapping duration→price.

    Returns:
        Price in cents or None when duration is missing/invalid.
    """
    if getattr(appt, "price_override_cents", None) is not None:
        return int(appt.price_override_cents)

    if appt.start_utc and appt.end_utc:
        dur = int((appt.end_utc - appt.start_utc).total_seconds() // 60)
        if dur <= 0:
            return None
        return price_map.get(dur)
    return None


def infer_payment_status(appt: Appointment, expected: Optional[int]) -> PaymentStatus:
    """Infer normalized payment status for an appointment.

    Priority rules:
    - If 'bundle_id' is set, return "bundle".
    - Respect model flags: "paid", "waived", "refunded".
    - Otherwise compare 'amount_paid_cents' to 'expected' to return
      "paid", "partial", or "unpaid". When 'expected' is None, any payment
      yields "partial".

    Args:
        appt: Appointment record.
        expected: Expected price in cents, or None.

    Returns:
        One of: "paid" | "partial" | "unpaid" | "bundle" | "unknown".
    """
    if getattr(appt, "bundle_id", None):
        return "bundle"
    ps = (getattr(appt, "payment_status", None) or "").lower()
    if ps == "paid":
        return "paid"
    if ps in ("waived",):
        return "paid"
    if ps in ("refunded",):
        return "unpaid"
    if ps == "paid":
        return "paid"

    paid = int(appt.amount_paid_cents or 0)
    if expected is None:
        return "unpaid" if paid <= 0 else "partial"
    if paid >= expected:
        return "paid"
    if paid > 0:
        return "partial"
    return "unpaid"


def gather_owner_payments(
    db: Session,
    start: date,
    end: date,
    client_account_ids: Optional[Sequence[int]] = None,
    owner_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate owner payments across accounts within a date window.

    Excludes canceled appointments from revenue. Counts any appointment that
    overlaps the window: '(end_utc > start_dt) and (start_utc < end_dt)'.

    Args:
        db: SQLAlchemy session.
        start: Start date (inclusive).
        end: End date (inclusive).
        client_account_ids: Optional list of client account IDs to include.
        owner_user_id: Optional owner scope; filters to a single owner when set.

    Returns:
        Dict with 'start', 'end', 'totals', and 'results' (per-client aggregates).
    """
    start_dt, end_dt = _daterange_to_utc(start, end)
    price_map = _service_price_map(db, owner_user_id)

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
            Appointment.end_utc > start_dt,
            Appointment.start_utc < end_dt,
            Appointment.status != "canceled",
        )
    )
    if owner_user_id:
        q = q.filter(Appointment.owner_id == owner_user_id)

    if client_account_ids:
        ids = list(client_account_ids)
        q = q.filter(or_(ClientAccount.id.in_(ids), Person.account_id.in_(ids)))

    rows: List[tuple[Appointment, Optional[ClientAccount]]] = q.all()

    per_client: Dict[int, Dict[str, Any]] = {}
    totals = {
        "appointments": 0,
        "late": 0,
        "paid_appts": 0,
        "unpaid_appts": 0,
        "total_expected_cents": 0,
        "total_paid_cents": 0,
        "total_owed_cents": 0,
    }

    user_ids = {a.client_id for (a, _) in rows if a.client_id}
    users = (
        {u.id: u for u in db.query(User).filter(User.id.in_(user_ids)).all()}
        if user_ids
        else {}
    )

    for a, acct in rows:
        cid = int(getattr(acct, "id", 0))

        if cid not in per_client:
            fallback = users.get(a.client_id)
            display_name = (
                getattr(acct, "name", None)
                or (fallback.name if fallback and fallback.name else None)
                or (fallback.email if fallback and fallback.email else None)
                or "Client"
            )
            per_client[cid] = {
                "client_account_id": cid,
                "client_name": display_name,
                "appointments": 0,
                "late": 0,
                "paid_appts": 0,
                "unpaid_appts": 0,
                "total_expected_cents": 0,
                "total_paid_cents": 0,
                "total_owed_cents": 0,
            }

        expected = expected_price_cents(a, price_map)
        paid = int(a.amount_paid_cents or 0)
        status = infer_payment_status(a, expected)

        per_client[cid]["appointments"] += 1
        totals["appointments"] += 1

        if getattr(a, "attendance_status", None) == "late":
            per_client[cid]["late"] += 1
            totals["late"] += 1

        if status in ("paid", "bundle"):
            per_client[cid]["paid_appts"] += 1
            totals["paid_appts"] += 1
        else:
            per_client[cid]["unpaid_appts"] += 1
            totals["unpaid_appts"] += 1

        if expected is not None:
            owed = max(expected - paid, 0)
            per_client[cid]["total_expected_cents"] += expected
            per_client[cid]["total_paid_cents"] += paid
            per_client[cid]["total_owed_cents"] += owed

            totals["total_expected_cents"] += expected
            totals["total_paid_cents"] += paid
            totals["total_owed_cents"] += owed
        else:
            per_client[cid]["total_paid_cents"] += paid
            totals["total_paid_cents"] += paid

    def _sum_wallet_deposits(
        owner_user_id: str,
        start_dt: datetime,
        end_dt: datetime,
        account_ids: Optional[Sequence[int]] = None,
    ) -> int:
        dep_q = (
            db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
            .join(PrepaidBundle, PrepaidBundle.id == PrepaidLedger.bundle_id)
            .outerjoin(
                ClientAccount,
                and_(
                    ClientAccount.client_user_id == PrepaidBundle.client_id,
                    ClientAccount.owner_user_id == PrepaidBundle.owner_id,
                ),
            )
            .filter(
                PrepaidBundle.owner_id == owner_user_id,
                PrepaidBundle.total_credits == 0,
                PrepaidLedger.amount_cents > 0,
                PrepaidLedger.created_at >= start_dt,
                PrepaidLedger.created_at < end_dt,
            )
        )
        if account_ids:
            dep_q = dep_q.filter(ClientAccount.id.in_(list(account_ids)))
        try:
            val = dep_q.scalar()
            return int(val or 0)
        except Exception:
            return 0

    if owner_user_id and get_owner_flag(
        str(owner_user_id),
        "wallet_deposits_as_paid",
        "FEATURE_WALLET_DEPOSITS_AS_PAID",
        default=False,
    ):
        deposit_total = _sum_wallet_deposits(
            owner_user_id, start_dt, end_dt, client_account_ids
        )
        totals["total_bundle_cents"] = int(deposit_total)
        totals["total_paid_cents"] = int(totals.get("total_cash_cents", 0)) + int(
            deposit_total
        )

    result_list = list(per_client.values())
    result_list.sort(
        key=lambda r: (r["total_paid_cents"], r["appointments"]), reverse=True
    )

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "totals": totals,
        "results": result_list,
    }


def build_client_rows(db: Session, user_client_account_id: int) -> Dict[str, Any]:
    """Build client-facing payments rollup for a single account.

    Resolves the account to its 'auth.User.id' and retrieves appointments by
    'Appointment.client_id'.

    Args:
        db: SQLAlchemy session.
        user_client_account_id: The client account ID owned by the user.

    Returns:
        Dict containing a 'summary' section and detailed 'rows' per appointment.
    """
    price_map = _service_price_map(db)
    price_map: Dict[int, int]
    acct = (
        db.query(ClientAccount)
        .filter(ClientAccount.id == user_client_account_id)
        .first()
    )
    if not acct:
        return {
            "summary": {
                "total_appointments": 0,
                "late_appointments": 0,
                "paid_appointments": 0,
                "unpaid_appointments": 0,
                "total_expected_cents": 0,
                "total_paid_cents": 0,
                "total_owed_cents": 0,
            },
            "rows": [],
        }
    price_map = _service_price_map(db, acct.owner_user_id)

    client_user_id = acct.client_user_id

    rows = (
        db.query(Appointment)
        .filter(Appointment.client_id == client_user_id)
        .order_by(Appointment.start_utc.desc())
        .all()
    )

    out_rows: List[Dict[str, Any]] = []
    roll = {
        "total_appointments": 0,
        "late_appointments": 0,
        "paid_appointments": 0,
        "unpaid_appointments": 0,
        "total_expected_cents": 0,
        "total_paid_cents": 0,
        "total_owed_cents": 0,
    }

    for a in rows:
        expected = expected_price_cents(a, price_map)
        paid = int(a.amount_paid_cents or 0)
        status = infer_payment_status(a, expected)
        duration_minutes = (
            int((a.end_utc - a.start_utc).total_seconds() // 60)
            if a.start_utc and a.end_utc
            else 0
        )

        out_rows.append(
            {
                "id": str(a.id),
                "start_utc": a.start_utc.isoformat() if a.start_utc else "",
                "duration_minutes": duration_minutes,
                "status": a.status,
                "attendance": getattr(a, "attendance_status", None),
                "price_cents": expected,
                "amount_paid_cents": paid,
                "payment_status": status,
            }
        )

        roll["total_appointments"] += 1
        if getattr(a, "attendance_status", None) == "late":
            roll["late_appointments"] += 1
        if status in ("paid", "bundle"):
            roll["paid_appointments"] += 1
        else:
            roll["unpaid_appointments"] += 1

        if expected is not None:
            roll["total_expected_cents"] += expected
            roll["total_paid_cents"] += paid
            roll["total_owed_cents"] += max(expected - paid, 0)
        else:
            roll["total_paid_cents"] += paid

    return {"summary": roll, "rows": out_rows}


def compute_price_cents(
    db: Session, appt: Appointment, price_map: Dict[int, int]
) -> Optional[int]:
    """Use per-appointment override if present; otherwise derive from duration via active ServiceOptions.

    Args:
        db: SQLAlchemy session (not used directly).
        appt: Appointment to price.
        price_map: Mapping of duration minutes to price cents.

    Returns:
        Price in cents if resolvable; otherwise None.
    """
    if getattr(appt, "price_override_cents", None) is not None:
        return int(appt.price_override_cents)
    if appt.start_utc and appt.end_utc:
        dur = int((appt.end_utc - appt.start_utc).total_seconds() // 60)
        return price_map.get(dur)
    return None


def compute_bundle_applied_cents(db: Session, appt: Appointment) -> int:
    """Legacy helper to sum bundle funds applied to an appointment.

    Only sums when 'PrepaidLedger.amount_cents' is available; otherwise returns 0.
    Prefer '_compute_bundle_applied_cents_safe' for robust behavior.

    Args:
        db: SQLAlchemy session.
        appt: Appointment.

    Returns:
        Non-negative integer cents of applied bundle funds.
    """
    if PrepaidLedger is None:
        return 0
    if not hasattr(PrepaidLedger, "appointment_id"):
        return 0
    if not hasattr(PrepaidLedger, "amount_cents"):
        return 0
    q = db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0)).filter(
        PrepaidLedger.appointment_id == appt.id
    )
    return int(q.scalar() or 0)


def _compute_bundle_applied_cents_safe(
    db: Session, appt: Appointment, expected: Optional[int]
) -> int:
    """Compute wallet contribution for an appointment with safe semantics.

    - If a monetary ledger exists, net all appointment-linked rows where
      consumption is negative and restoration is positive. The returned value
      is clamped to a non-negative number of cents actually applied.
    - If the ledger schema is unavailable, return 0.

    Args:
        db: SQLAlchemy session.
        appt: Appointment (may or may not have a bundle_id).
        expected: Expected price (unused; reserved for future heuristics).

    Returns:
        Applied bundle amount in cents (>= 0).
    """
    if not getattr(appt, "bundle_id", None):
        return 0

    if PrepaidLedger is not None and hasattr(PrepaidLedger, "amount_cents"):
        total = (
            db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
            .filter(PrepaidLedger.appointment_id == appt.id)
            .scalar()
        )
        try:
            val = int(total or 0)
        except Exception:
            val = 0
        return -val if val < 0 else 0
    return 0


def compute_financials(
    db: Session, appt: Appointment, price_map: Dict[int, int]
) -> Dict[str, Any]:
    """Compute expected price, paid amounts, owed, and normalized status.

    Args:
        db: SQLAlchemy session.
        appt: Appointment instance.
        price_map: Mapping from duration minutes to price cents.

    Returns:
        Dict with keys:
          - 'price_cents': Expected price or None
          - 'paid_cash_cents': Cash paid (int)
          - 'bundle_applied_cents': Wallet/bundle contribution (int)
          - 'owed_cents': Max(price - paid_total, 0)
          - 'payment_status': paid | partial | unpaid | waived | refunded | bundle
    """
    expected = compute_price_cents(db, appt, price_map)
    cash = int(appt.amount_paid_cents or 0)
    bundle = _compute_bundle_applied_cents_safe(db, appt, expected)

    paid_total = cash + bundle
    owed = 0
    if expected is not None:
        owed = max(expected - paid_total, 0)
    ps_model = (getattr(appt, "payment_status", None) or "").lower()
    has_bundle = bool(getattr(appt, "bundle_id", None))
    if ps_model in ("paid", "waived", "refunded"):
        payment_status = "paid" if ps_model == "paid" else ps_model
    else:
        if expected is None:
            payment_status = "partial" if paid_total > 0 else "unpaid"
        else:
            if paid_total >= expected:
                payment_status = "bundle" if has_bundle else "paid"
            elif paid_total > 0:
                payment_status = "partial"
            else:
                payment_status = "unpaid"

    if payment_status in {"paid", "waived", "refunded", "bundle"}:
        owed = 0

    return {
        "price_cents": expected,
        "paid_cash_cents": cash,
        "bundle_applied_cents": bundle,
        "owed_cents": owed,
        "payment_status": payment_status,
    }


def _sum_wallet_balance_current(
    db: Session,
    *,
    owner_user_id: str,
    client_account_ids: Optional[Sequence[int]] = None,
) -> int:
    """Current wallet balance across all wallets for this owner.

    Sums 'PrepaidLedger.amount_cents' for wallet bundles ('total_credits == 0').
    Optionally filters to a set of client accounts via joined account IDs.

    Args:
        db: SQLAlchemy session.
        owner_user_id: Owner to scope bundles/ledgers.
        client_account_ids: Optional list of client account IDs to include.

    Returns:
        Integer cents representing the current wallet balance.
    """
    q = (
        db.query(func.coalesce(func.sum(PrepaidLedger.amount_cents), 0))
        .join(PrepaidBundle, PrepaidBundle.id == PrepaidLedger.bundle_id)
        .outerjoin(
            ClientAccount,
            and_(
                ClientAccount.client_user_id == PrepaidBundle.client_id,
                ClientAccount.owner_user_id == PrepaidBundle.owner_id,
            ),
        )
        .filter(
            PrepaidBundle.owner_id == owner_user_id,
            PrepaidBundle.total_credits == 0,
            ClientAccount.deleted_at.is_(None),
        )
    )
    if client_account_ids:
        q = q.filter(ClientAccount.id.in_(list(client_account_ids)))
    try:
        v = q.scalar()
        return int(v or 0)
    except Exception:
        return 0


def list_owner_financial_rows(
    db: Session,
    start: date,
    end: date,
    owner_user_id: str,
    status: Optional[Sequence[str]] = None,
    payment_status: Optional[Sequence[str]] = None,
    client_account_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return per-appointment financial rows with optional filters.

    Args:
        db: SQLAlchemy session.
        start: Start date (inclusive).
        end: End date (inclusive).
        owner_user_id: Owner to scope appointments.
        status: Optional filter of appointment statuses.
        payment_status: Optional filter on computed payment statuses.
        client_account_id: Optional single account filter.

    Returns:
        List of dict rows including appointment fields and computed financials.
    """
    start_dt, end_dt = _daterange_to_utc(start, end)
    price_map = _service_price_map(db, owner_user_id)

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
            Appointment.owner_id == owner_user_id,
            Appointment.end_utc > start_dt,
            Appointment.start_utc < end_dt,
        )
    )
    if status:
        q = q.filter(Appointment.status.in_(status))
    if client_account_id is not None:
        cid = int(client_account_id)
        q = q.filter(or_(ClientAccount.id == cid, Person.account_id == cid))

    rows = q.order_by(Appointment.start_utc.desc()).all()
    user_ids = {a.client_id for (a, _) in rows if a.client_id}
    users = (
        {u.id: u for u in db.query(User).filter(User.id.in_(user_ids)).all()}
        if user_ids
        else {}
    )

    out: List[Dict[str, Any]] = []
    for a, acct in rows:
        fin = compute_financials(db, a, price_map)
        if payment_status and fin["payment_status"] not in set(payment_status):
            continue

        duration_minutes = (
            int((a.end_utc - a.start_utc).total_seconds() // 60)
            if a.start_utc and a.end_utc
            else 0
        )
        person_name = None
        person_email = None
        try:
            p = (
                db.query(Person).filter(Person.id == a.person_id).first()
                if getattr(a, "person_id", None)
                else None
            )
            if p:
                person_name = getattr(p, "full_name", None)
                person_email = getattr(p, "email", None)
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

        out.append(
            {
                "id": str(a.id),
                "start_utc": a.start_utc,
                "end_utc": a.end_utc,
                "client_account_id": int(getattr(acct, "id", 0)) if acct else None,
                "client_label": client_label,
                "status": a.status,
                "duration_minutes": duration_minutes,
                "attendance_status": getattr(a, "attendance_status", None),
                "lesson_person_name": person_name,
                "lesson_person_email": person_email,
                "is_group": bool(getattr(a, "group_id", None)),
                "cancel_reason": getattr(a, "cancel_reason", None),
                **fin,
            }
        )
    return out


def summarize_financial_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize totals from per-appointment financial rows.

    Args:
        rows: Rows produced by 'list_owner_financial_rows'.

    Returns:
        Dict of aggregate totals including expected, paid (cash+bundle), cash,
        bundle, owed, and number of no-shows.
    """
    total_expected = 0
    total_cash = 0
    total_bundle = 0
    total_paid = 0
    total_owed = 0
    total_no_show = 0
    for r in rows:
        if r.get("price_cents") is not None:
            total_expected += int(r["price_cents"] or 0)
        total_cash += int(r["paid_cash_cents"] or 0)
        total_bundle += int(r["bundle_applied_cents"] or 0)
        total_paid += int(r["paid_cash_cents"] or 0) + int(
            r["bundle_applied_cents"] or 0
        )
        total_owed += int(r["owed_cents"] or 0)
        if (r.get("attendance_status") or "").lower() == "no_show":
            total_no_show += 1
    return {
        "total_appointments": len(rows),
        "total_expected_cents": total_expected,
        "total_paid_cents": total_paid,
        "total_cash_cents": total_cash,
        "total_bundle_cents": total_bundle,
        "total_owed_cents": total_owed,
        "total_no_show": total_no_show,
    }
