from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, ToolException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import (
    User,
    ClientAccount,
    ClientEmail,
    Person,
    Appointment,
    OutboxEmail,
    OutboxEmailRecipient,
    OutboxEmailStatus,
)
from agent.schemas import (
    ToolSendEmailIn,
    ToolSendEmailOut,
    ToolListServiceOptionsOut,
    ToolFinancialSummaryIn,
    ToolFinancialSummaryOut,
    ToolDateRangeIn,
    CustomerPaymentsOut,
    CustomerBalancesOut,
    TotalOwedOut,
    ToolOwnerDashboardIn,
    ToolOwnerDashboardOut,
    EmailDraftIn,
    EmailDraftOut,
    EmailApprovalIn,
)
from services.emailer import send_email, render_basic_html
from services.payments import list_owner_financial_rows, summarize_financial_rows
from services.services_scheduling import list_service_options

log = logging.getLogger(__name__)

# ----------------------------- Helpers -----------------------------

ALLOW_AGENT_DIRECT_EMAIL = os.getenv("ALLOW_AGENT_DIRECT_EMAIL", "0") == "1"


def _norm(s: Optional[str]) -> Optional[str]:
    """Trim string values; preserve None/other types."""
    return s.strip() if isinstance(s, str) else s


def _lower(s: Optional[str]) -> Optional[str]:
    """Lowercase and trim string values; preserve None/other types."""
    return s.lower().strip() if isinstance(s, str) else s


def _owner_id_from_config(config: RunnableConfig) -> str:
    """Extract the owner id from a runnable config.

    Args:
        config: Runnable configuration with a 'configurable' sub-dict.

    Returns:
        Owner id as a string.

    Raises:
        RuntimeError: If the owner id is missing from the configuration.
    """
    cfg = config or {}
    cfg = (
        cfg.get("configurable", {})
        if isinstance(cfg, dict)
        else getattr(cfg, "configurable", {}) or {}
    )
    owner_id = cfg.get("user_id") or cfg.get("owner_id")
    if not owner_id:
        raise RuntimeError("Missing owner id in tool config")
    return str(owner_id)


def _primary_account_email(db: Session, account_id: int) -> Optional[str]:
    """Return primary email for an account.

    Checks 'ClientEmail' rows first (preferring 'is_primary'), then falls
    back to the linked user email when available.

    Args:
        db: Database session.
        account_id: Client account id.

    Returns:
        Primary email string if found; otherwise None.
    """
    row = (
        db.query(ClientEmail)
        .filter(ClientEmail.account_id == account_id)
        .order_by(ClientEmail.is_primary.desc(), ClientEmail.id.asc())
        .first()
    )
    if row and getattr(row, "email", None):
        return row.email

    acct = db.query(ClientAccount).filter(ClientAccount.id == account_id).first()
    if acct and getattr(acct, "client_user_id", None):
        owner_email = (
            db.query(User.email).filter(User.id == acct.client_user_id).scalar()
        )
        if owner_email:
            return owner_email

    return None


def _fmt_cents(cents: int, currency: str = "$") -> str:
    """Format an amount in cents as a currency string.

    Args:
        cents: Integer cents value.
        currency: Currency symbol to prefix (default "$").

    Returns:
        Formatted amount like "$12.34".
    """
    try:
        v = int(cents)
    except Exception:
        v = 0
    return f"{currency}{v / 100:,.2f}"


# -------------------------- Identity Resolver --------------------------


def resolve_person(
    db: Session,
    *,
    owner_id: str,
    client_name: Optional[str],
    client_email: Optional[str],
) -> Tuple[Person, str, str]:
    """
    Identity precedence (NO new ClientAccount creation here):
      1) client_email → Person.email exact (case-insensitive) under this owner.
      2) client_name → Person.full_name exact (case-insensitive).
         - 1 match → use it
         - >1     → raise ToolException("AMBIGUOUS_PERSON:[...]")
      3) client_name → Account.name exact (case-insensitive).
         - 1 person → use it
         - >1       → raise ToolException("AMBIGUOUS_PERSON:[...]")
         - 0        → create a *shadow Person* with that name under the account
      4) client_email → Account via ClientEmail
         - 1 person → use it
         - >1       → raise ToolException("AMBIGUOUS_PERSON:[...]")
         - 0        → create a *shadow Person* with account name or email user-part
      5) else → ToolException("NO_MATCH: Provide person_id or a known name/email")

    Args:
        db: Database session.
        owner_id: Owner user id to scope lookups.
        client_name: Candidate client full name.
        client_email: Candidate client email.

    Returns:
        Tuple of (person, canonical_name, canonical_email).

    Raises:
        ToolException: If identity is ambiguous or cannot be resolved.
    """
    name = _norm(client_name)
    email = _norm(client_email)

    # 1) Email → Person.email
    if email:
        p = (
            db.query(Person)
            .join(ClientAccount, ClientAccount.id == Person.account_id)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                func.lower(Person.email) == func.lower(email),
            )
            .first()
        )
        if p:
            canon_email = p.email or email or _primary_account_email(db, p.account_id)
            return p, (p.full_name or name or ""), (canon_email or "")

    # 2) Name → exact Person match
    if name:
        persons = (
            db.query(Person)
            .join(ClientAccount, ClientAccount.id == Person.account_id)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                func.lower(Person.full_name) == func.lower(name),
            )
            .order_by(Person.id.asc())
            .all()
        )
        if len(persons) == 1:
            p = persons[0]
            canon_email = p.email or email or _primary_account_email(db, p.account_id)
            return p, (p.full_name or name), (canon_email or "")
        if len(persons) > 1:
            choices = [
                {"person_id": str(p.id), "full_name": p.full_name, "email": p.email}
                for p in persons
            ]
            raise ToolException("AMBIGUOUS_PERSON:" + json.dumps(choices))

        # 3) Name → Account match
        acct = (
            db.query(ClientAccount)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                func.lower(ClientAccount.name) == func.lower(name),
            )
            .first()
        )
        if acct:
            people = (
                db.query(Person)
                .filter(Person.account_id == acct.id)
                .order_by(Person.id.asc())
                .all()
            )
            if len(people) == 1:
                p = people[0]
                canon_email = p.email or email or _primary_account_email(db, acct.id)
                return p, (p.full_name or name), (canon_email or "")
            if len(people) > 1:
                choices = [
                    {"person_id": str(p.id), "full_name": p.full_name, "email": p.email}
                    for p in people
                ]
                raise ToolException("AMBIGUOUS_PERSON:" + json.dumps(choices))
            shadow = Person(
                account_id=acct.id, full_name=(acct.name or name), email=None
            )
            db.add(shadow)
            db.flush()
            canon_email = email or _primary_account_email(db, acct.id)
            return shadow, shadow.full_name, (canon_email or "")

    # 4) Email → Account via ClientEmail
    if email:
        acct = (
            db.query(ClientAccount)
            .join(ClientEmail, ClientEmail.account_id == ClientAccount.id)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                func.lower(ClientEmail.email) == func.lower(email),
            )
            .first()
        )
        if acct:
            people = (
                db.query(Person)
                .filter(Person.account_id == acct.id)
                .order_by(Person.id.asc())
                .all()
            )
            if len(people) == 1:
                p = people[0]
                canon_email = p.email or email or _primary_account_email(db, acct.id)
                return p, (p.full_name or name or acct.name or ""), (canon_email or "")
            if len(people) > 1:
                choices = [
                    {"person_id": str(p.id), "full_name": p.full_name, "email": p.email}
                    for p in people
                ]
                raise ToolException("AMBIGUOUS_PERSON:" + json.dumps(choices))
            shadow = Person(
                account_id=acct.id,
                full_name=(acct.name or name or email.split("@")[0]),
                email=None,
            )
            db.add(shadow)
            db.flush()
            canon_email = email or _primary_account_email(db, acct.id)
            return shadow, shadow.full_name, (canon_email or "")

    raise ToolException("NO_MATCH: Provide person_id or a known name/email")


def _account_email_fallback(db: Session, person: Person) -> Optional[str]:
    """Return account primary email when a person email is missing.

    Args:
        db: Database session.
        person: Person instance with an 'account_id'.

    Returns:
        Primary account email or None if unavailable.
    """
    if getattr(person, "account_id", None) is None:
        return None
    return _primary_account_email(db, person.account_id)


def attach_identity_to_appointment(
    appt: Appointment,
    *,
    person: Person,
    canonical_name: str,
    canonical_email: str,
    db: Session,
) -> None:
    """Set both FK and denormalized identity fields on an appointment.

    Args:
        appt: Appointment ORM instance to mutate.
        person: Resolved person to attach.
        canonical_name: Preferred name label to store.
        canonical_email: Preferred email to store.
        db: Active database session used for fallbacks.
    """
    appt.person_id = person.id

    resolved_name = (
        canonical_name
        or person.full_name
        or (
            getattr(person.account, "name", None)
            if getattr(person, "account", None)
            else None
        )
    )
    if not resolved_name:
        resolved_name = "Client"
    appt.client_name = resolved_name

    resolved_email = (
        canonical_email or person.email or _account_email_fallback(db, person)
    )
    appt.client_email = resolved_email

    # If we had to fall back to an account email because the person has none, append a helpful note.
    if resolved_email and not (canonical_email or person.email):
        note_line = f"Booked for {resolved_name} (using account email {resolved_email}; no direct email on file)."
        existing_note = getattr(appt, "owner_private_note", None) or ""
        if note_line not in existing_note:
            appt.owner_private_note = (
                existing_note + ("\n" if existing_note else "") + note_line
            )


# --------------------------- Email Draft / Send ---------------------------


def _persist_draft(
    db: Session, owner_user_id: str, payload: EmailDraftIn
) -> OutboxEmail:
    """Persist an email draft and any recipients.

    If 'payload.to' is missing but recipients are present, anchors the draft
    to the first recipient for preview. Writes recipient rows into
    'OutboxEmailRecipient'.

    Args:
        db: Database session.
        owner_user_id: Owner scope for the draft.
        payload: Draft payload including subject and lines.

    Returns:
        The persisted 'OutboxEmail' row (refreshed).
    """
    anchor_to = payload.to
    anchor_name = payload.to_name
    if not anchor_to and payload.recipients:
        first = payload.recipients[0]
        email = (
            getattr(first, "email", None)
            if not isinstance(first, dict)
            else first.get("email")
        )
        name = (
            getattr(first, "name", None)
            if not isinstance(first, dict)
            else first.get("name")
        )
        if email:
            anchor_to = str(email)
            if not anchor_name:
                anchor_name = name

    text = "\n".join(payload.lines or [])
    html = render_basic_html(payload.subject, payload.lines or [])

    ob = OutboxEmail(
        owner_user_id=owner_user_id,
        to_email=anchor_to,
        to_name=anchor_name,
        subject=payload.subject,
        text_body=text,
        preview_html=html,
        status=OutboxEmailStatus.PENDING.value,
    )
    db.add(ob)
    db.commit()
    db.refresh(ob)
    if payload.recipients:
        for r in payload.recipients:
            email = (
                getattr(r, "email", None) if not isinstance(r, dict) else r.get("email")
            )
            name = (
                getattr(r, "name", None) if not isinstance(r, dict) else r.get("name")
            )
            if email:
                db.add(
                    OutboxEmailRecipient(outbox_id=ob.id, email=str(email), name=name)
                )
        db.commit()
        db.refresh(ob)

    return ob


def _send_if_approved(db: Session, draft_id: str | int) -> OutboxEmail:
    """Send a previously approved draft in place.

    No-op if the draft is not approved yet.

    Args:
        db: Database session.
        draft_id: OutboxEmail id.

    Returns:
        The updated 'OutboxEmail' row.
    """
    ob = db.query(OutboxEmail).filter(OutboxEmail.id == draft_id).one()
    if (ob.status or "").lower() != OutboxEmailStatus.APPROVED.value:
        return ob
    send_email(
        to=ob.to_email, subject=ob.subject, text=ob.text_body, html=ob.preview_html
    )
    ob.status = OutboxEmailStatus.SENT.value
    ob.sent_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(ob)
    return ob


# ------------------------------ Tools ------------------------------


@tool("send_email", args_schema=ToolSendEmailIn, return_direct=False)
def send_email_tool(
    to: str,
    subject: str,
    text: Optional[str],
    html: Optional[str],
    config: RunnableConfig,
) -> ToolSendEmailOut:
    """Queue a simple SMTP email immediately.

    If only text is provided, minimal branded HTML is rendered. This direct-send
    path is guarded by the 'ALLOW_AGENT_DIRECT_EMAIL' environment flag.

    Args:
        to: Recipient email address.
        subject: Subject line.
        text: Plaintext body (optional).
        html: HTML body (optional; rendered if missing and text provided).
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolSendEmailOut' with a queued id.

    Raises:
        ToolException: If direct email is disabled by configuration.
    """
    _ = _owner_id_from_config(config)
    if not ALLOW_AGENT_DIRECT_EMAIL:
        raise ToolException(
            "DIRECT_EMAIL_DISABLED: Agent must create a draft for owner approval. Use create_email_draft instead."
        )
    if not html and text:
        html = render_basic_html(subject, text.splitlines())
    send_email(to=to, subject=subject, text=(text or ""), html=html)
    return ToolSendEmailOut(queued_id="ok")


@tool("list_service_options", return_direct=False)
def list_service_options_tool(config: RunnableConfig) -> ToolListServiceOptionsOut:
    """Return active service options (durations/prices) for the owner.

    Args:
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolListServiceOptionsOut' with normalized option rows.
    """
    owner_id = _owner_id_from_config(config)
    with SessionLocal() as db:
        opts = list_service_options(db, owner_id)
        out = [
            {
                "duration_minutes": int(o.duration_minutes),
                "price_cents": int(o.price_cents),
                "currency": o.currency,
                "price_pretty": _fmt_cents(int(o.price_cents), "$"),
            }
            for o in opts
        ]
        return ToolListServiceOptionsOut(options=out)


@tool("financial_summary", args_schema=ToolFinancialSummaryIn, return_direct=False)
def financial_summary_tool(
    start: str,
    end: str,
    client_account_id: Optional[int],
    status: Optional[List[str]],
    payment_status: Optional[List[str]],
    config: RunnableConfig,
) -> ToolFinancialSummaryOut:
    """Return owner financial rows and totals for a date range.

    Args:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).
        client_account_id: Optional account filter.
        status: Optional list of appointment statuses to include.
        payment_status: Optional list of payment statuses to include.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolFinancialSummaryOut' containing totals and detail rows.

    Raises:
        ToolException: If the date range is invalid.
    """
    owner_id = _owner_id_from_config(config)
    from datetime import date as _date

    try:
        start_d, end_d = _date.fromisoformat(start), _date.fromisoformat(end)
    except Exception:
        raise ToolException("BAD_DATE_RANGE: use YYYY-MM-DD for start and end")
    with SessionLocal() as db:
        rows = list_owner_financial_rows(
            db,
            start=start_d,
            end=end_d,
            owner_user_id=owner_id,
            status=status,
            payment_status=payment_status,
            client_account_id=client_account_id,
        )
        totals = summarize_financial_rows(rows)
        return ToolFinancialSummaryOut(totals=totals, results=rows)


@tool("customer_payments", args_schema=ToolDateRangeIn, return_direct=False)
def customer_payments_tool(
    start: Optional[str] = None,
    end: Optional[str] = None,
    config: RunnableConfig = None,
) -> CustomerPaymentsOut:
    """Return per-customer payments within an optional date window.

    Args:
        start: Optional inclusive start date (YYYY-MM-DD). Defaults to Jan 1.
        end: Optional inclusive end date (YYYY-MM-DD). Defaults to today.
        config: Runnable configuration providing the owner id.

    Returns:
        'CustomerPaymentsOut' grouped by client, including line items.

    Raises:
        ToolException: If the date window is invalid.
    """
    owner_id = _owner_id_from_config(config)
    from datetime import date as _date

    today = _date.today()
    s = _date.fromisoformat(start) if start else _date(today.year, 1, 1)
    e = _date.fromisoformat(end) if end else today
    if e < s:
        raise ToolException("BAD_DATE_RANGE: end must be >= start")
    with SessionLocal() as db:
        rows = list_owner_financial_rows(
            db=db,
            start=s,
            end=e,
            owner_user_id=owner_id,
            status=None,
            payment_status=None,
            client_account_id=None,
        )
        from datetime import datetime as _dt

        by_client: Dict[str, Dict[str, any]] = {}
        for r in rows:
            label = r.get("client_label") or "Client"
            key = str(r.get("client_account_id") or label)
            grp = by_client.setdefault(
                key,
                {
                    "client_account_id": r.get("client_account_id"),
                    "client_label": label,
                    "total_paid_cents": 0,
                    "lines": [],
                },
            )
            paid_cash = int(r.get("paid_cash_cents") or 0)
            bundle = int(r.get("bundle_applied_cents") or 0)
            grp["total_paid_cents"] += paid_cash + bundle
            dt: _dt = r.get("start_utc")
            grp["lines"].append(
                {
                    "date": (dt.date().isoformat() if hasattr(dt, "date") else str(dt)),
                    "appointment_id": r.get("id"),
                    "paid_cash_cents": paid_cash,
                    "bundle_applied_cents": bundle,
                }
            )
        customers = list(by_client.values())
        for c in customers:
            c["lines"].sort(key=lambda x: x["date"])
        customers.sort(key=lambda x: x["client_label"] or "")
        return CustomerPaymentsOut(customers=customers)


@tool("customer_balances", args_schema=ToolDateRangeIn, return_direct=False)
def customer_balances_tool(
    start: Optional[str] = None,
    end: Optional[str] = None,
    config: RunnableConfig = None,
) -> CustomerBalancesOut:
    """Return per-customer balances (owed) within an optional date window.

    Args:
        start: Optional inclusive start date (YYYY-MM-DD). Defaults to Jan 1.
        end: Optional inclusive end date (YYYY-MM-DD). Defaults to today.
        config: Runnable configuration providing the owner id.

    Returns:
        'CustomerBalancesOut' grouped by client, sorted by owed desc.

    Raises:
        ToolException: If the date window is invalid.
    """
    owner_id = _owner_id_from_config(config)
    from datetime import date as _date

    today = _date.today()
    s = _date.fromisoformat(start) if start else _date(today.year, 1, 1)
    e = _date.fromisoformat(end) if end else today
    if e < s:
        raise ToolException("BAD_DATE_RANGE: end must be >= start")
    with SessionLocal() as db:
        rows = list_owner_financial_rows(
            db=db,
            start=s,
            end=e,
            owner_user_id=owner_id,
            status=None,
            payment_status=None,
            client_account_id=None,
        )
        by_client: Dict[str, Dict[str, any]] = {}
        for r in rows:
            label = r.get("client_label") or "Client"
            key = str(r.get("client_account_id") or label)
            grp = by_client.setdefault(
                key,
                {
                    "client_account_id": r.get("client_account_id"),
                    "client_label": label,
                    "total_owed_cents": 0,
                },
            )
            grp["total_owed_cents"] += int(r.get("owed_cents") or 0)
        customers = list(by_client.values())
        customers.sort(
            key=lambda x: (
                -int(x.get("total_owed_cents") or 0),
                x.get("client_label") or "",
            )
        )
        return CustomerBalancesOut(customers=customers)


@tool("total_owed", args_schema=ToolDateRangeIn, return_direct=False)
def total_owed_tool(
    start: Optional[str] = None,
    end: Optional[str] = None,
    config: RunnableConfig = None,
) -> TotalOwedOut:
    """Return total amount owed within an optional inclusive date window.

    Defaults to YTD if no dates are provided.

    Args:
        start: Optional inclusive start date (YYYY-MM-DD).
        end: Optional inclusive end date (YYYY-MM-DD).
        config: Runnable configuration providing the owner id.

    Returns:
        'TotalOwedOut' with the computed sum in cents.

    Raises:
        ToolException: If the date window is invalid.
    """
    owner_id = _owner_id_from_config(config)
    from datetime import date as _date

    today = _date.today()
    s = _date.fromisoformat(start) if start else _date(today.year, 1, 1)
    e = _date.fromisoformat(end) if end else today
    if e < s:
        raise ToolException("BAD_DATE_RANGE: end must be >= start")
    with SessionLocal() as db:
        rows = list_owner_financial_rows(
            db=db,
            start=s,
            end=e,
            owner_user_id=owner_id,
            status=None,
            payment_status=None,
            client_account_id=None,
        )
        tot = sum(int(r.get("owed_cents") or 0) for r in rows)
        return TotalOwedOut(
            start=s.isoformat() if start else None,
            end=e.isoformat() if end else None,
            total_owed_cents=tot,
        )


@tool(
    "owner_financial_dashboard", args_schema=ToolOwnerDashboardIn, return_direct=False
)
def owner_financial_dashboard_tool(
    start: Optional[str] = None,
    end: Optional[str] = None,
    top_n: int = 5,
    config: RunnableConfig = None,
) -> ToolOwnerDashboardOut:
    """Return a financial dashboard summary for the owner.

    Args:
        start: Optional inclusive start date (YYYY-MM-DD).
        end: Optional inclusive end date (YYYY-MM-DD).
        top_n: Number of items to include in top lists.
        config: Runnable configuration providing the owner id.

    Returns:
        'ToolOwnerDashboardOut' with totals, revenue, owed, and top lists.

    Raises:
        ToolException: If the date window is invalid.
    """
    owner_id = _owner_id_from_config(config)
    from datetime import date as _date

    today = _date.today()
    s = _date.fromisoformat(start) if start else _date(today.year, 1, 1)
    e = _date.fromisoformat(end) if end else today
    if e < s:
        raise ToolException("BAD_DATE_RANGE: end must be >= start")
    with SessionLocal() as db:
        rows = list_owner_financial_rows(
            db=db,
            start=s,
            end=e,
            owner_user_id=owner_id,
            status=None,
            payment_status=None,
            client_account_id=None,
        )
        totals = summarize_financial_rows(rows)
        # group by client for owed
        owed_by_client: Dict[str, Dict[str, any]] = {}
        paid_by_client: Dict[str, Dict[str, any]] = {}
        for r in rows:
            label = r.get("client_label") or "Client"
            key = str(r.get("client_account_id") or label)
            ob = owed_by_client.setdefault(
                key,
                {
                    "client_account_id": r.get("client_account_id"),
                    "client_label": label,
                    "total_owed_cents": 0,
                },
            )
            ob["total_owed_cents"] += int(r.get("owed_cents") or 0)
            pb = paid_by_client.setdefault(
                key,
                {
                    "client_account_id": r.get("client_account_id"),
                    "client_label": label,
                    "total_paid_cents": 0,
                },
            )
            pb["total_paid_cents"] += int(r.get("paid_cash_cents") or 0) + int(
                r.get("bundle_applied_cents") or 0
            )

        top_debtors = sorted(
            owed_by_client.values(),
            key=lambda x: (
                -int(x.get("total_owed_cents") or 0),
                x.get("client_label") or "",
            ),
        )[: int(top_n)]
        top_payers = sorted(
            paid_by_client.values(),
            key=lambda x: (
                -int(x.get("total_paid_cents") or 0),
                x.get("client_label") or "",
            ),
        )[: int(top_n)]

        return ToolOwnerDashboardOut(
            start=s.isoformat() if start else None,
            end=e.isoformat() if end else None,
            totals=totals,
            revenue_paid_cents=int(totals.get("total_paid_cents") or 0),
            total_owed_cents=int(totals.get("total_owed_cents") or 0),
            top_debtors=top_debtors,
            top_payers=top_payers,
        )


@tool("explain_owner_dashboard", return_direct=True)
def explain_owner_dashboard_tool(
    dashboard: dict,
    style: str = "brief",
    currency: str = "$",
) -> str:
    """Format an owner financial dashboard into a concise human summary.

    Args:
        dashboard: Dict returned by 'owner_financial_dashboard'.
        style: "brief" or "detailed" (default "brief").
        currency: Currency symbol (default "$").

    Returns:
        A multi-line human-readable summary.
    """
    if not isinstance(dashboard, dict):
        return "No data to summarize."

    totals = dashboard.get("totals") or {}
    paid = int(
        dashboard.get("revenue_paid_cents") or totals.get("total_paid_cents") or 0
    )
    owed = int(dashboard.get("total_owed_cents") or totals.get("total_owed_cents") or 0)
    total_appts = int(totals.get("total_appointments") or 0)
    expected = int(totals.get("total_expected_cents") or 0)
    cash = int(totals.get("total_cash_cents") or 0)
    bundle = int(totals.get("total_bundle_cents") or 0)

    start = dashboard.get("start")
    end = dashboard.get("end")
    range_str = (
        f" for {start} to {end}"
        if start and end
        else (f" since {start}" if start else (f" up to {end}" if end else ""))
    )

    lines = []
    title = "Owner Financial Summary" + (f"{range_str}" if range_str else "")
    lines.append(title)
    lines.append("" if style == "brief" else "")
    lines += [
        f"- Revenue: {_fmt_cents(paid, currency)}",
        f"- Currently owed: {_fmt_cents(owed, currency)}",
    ]

    if style != "brief":
        lines += [
            f"- Expected (priced): {_fmt_cents(expected, currency)}",
            f"- Cash: {_fmt_cents(cash, currency)}  •  Bundle: {_fmt_cents(bundle, currency)}",
            f"- Appointments counted: {total_appts}",
        ]

    def _fmt_top(label: str, arr_key: str, amt_key: str, max_items: int = 5):
        arr = dashboard.get(arr_key) or []
        if not isinstance(arr, list) or not arr:
            return
        n = min(len(arr), max_items)
        lines.append("")
        lines.append(f"{label}")
        lines.append("")
        lines.append("| Client | Amount |")
        lines.append("|---|---:|")
        for i in range(n):
            row = arr[i] or {}
            name = (row.get("client_label") or "Client").replace("|", "/")
            amt = int(row.get(amt_key) or 0)
            lines.append(f"| {name} | {_fmt_cents(amt, currency)} |")

    _fmt_top("Top payers", "top_payers", "total_paid_cents")
    _fmt_top("Top debtors", "top_debtors", "total_owed_cents")

    return "\n".join(lines)


@tool("create_email_draft", args_schema=EmailDraftIn, return_direct=False)
def create_email_draft_tool(
    to: Optional[str] = None,
    subject: str = "",
    lines: List[str] = [],
    to_name: Optional[str] = None,
    recipients: Optional[List[Dict[str, Optional[str]]]] = None,
    config: RunnableConfig = None,
) -> str:
    """Create an email draft for later approval and sending.

    The UI will show an editor for the draft. For multiple recipients, do not
    comma-join into 'to'; instead provide 'recipients=[{email,name?}, ...]'.
    'to' may hold a single anchor recipient; the backend merges/dedupes.

    Args:
        to: Anchor recipient email (optional).
        subject: Draft subject.
        lines: Plaintext lines; rendered for preview.
        to_name: Anchor recipient name (optional).
        recipients: Optional list of additional recipients.
        config: Runnable configuration providing the owner id.

    Returns:
        A JSON string marker with the draft payload for preview by the graph.
    """
    owner_id = _owner_id_from_config(config)

    # Normalize recipients into plain dicts (works with pydantic or raw dicts)
    norm_recipients: List[Dict[str, Optional[str]]] = []
    for r in recipients or []:
        if isinstance(r, dict):
            email = r.get("email")
            name = r.get("name")
        else:
            email = getattr(r, "email", None)
            name = getattr(r, "name", None)
        if email:
            norm_recipients.append({"email": str(email), "name": name})

    # Anchor single 'to' if not provided
    anchor_to = to or (norm_recipients[0]["email"] if norm_recipients else None)
    anchor_name = to_name or (
        norm_recipients[0].get("name") if norm_recipients else None
    )

    payload = EmailDraftIn(
        to=anchor_to,
        to_name=anchor_name,
        subject=subject,
        lines=lines or [],
        recipients=norm_recipients or None,
    )

    with SessionLocal() as db:
        ob = _persist_draft(db, owner_id, payload)
        out = EmailDraftOut(
            draft_id=str(ob.id),
            to=ob.to_email,
            to_name=ob.to_name,
            subject=ob.subject,
            text=ob.text_body,
            html=ob.preview_html,
            status=ob.status,
            recipients=[
                {"email": r.email, "name": r.name} for r in (ob.recipients or [])
            ],
        )
        return json.dumps(
            {"marker": "email_draft", "payload": out.model_dump()}, ensure_ascii=False
        )


@tool("send_approved_email", args_schema=EmailApprovalIn, return_direct=False)
def send_approved_email_tool(
    draft_id: str | int,
    approve: bool,
    to: Optional[str] = None,
    to_name: Optional[str] = None,
    subject: Optional[str] = None,
    text: Optional[str] = None,
    recipients: Optional[List[Dict[str, Optional[str]]]] = None,
    replace_recipients: bool = False,
    config: RunnableConfig = None,
) -> Dict[str, object]:
    """Approve (optionally) and send an email draft.

    If 'approve' is True, the draft is approved before sending. Subject/text
    overrides are applied prior to approval, and recipients can be replaced or
    appended.

    Args:
        draft_id: OutboxEmail id of the draft.
        approve: Whether to approve before sending.
        to: Optional override for anchor recipient.
        to_name: Optional override for anchor name.
        subject: Optional subject override.
        text: Optional plaintext override.
        recipients: Optional recipient list to write.
        replace_recipients: If True, replace existing recipients; otherwise append.
        config: Runnable configuration providing the owner id.

    Returns:
        Dict with 'ok' and 'status' fields describing send outcome.
    """
    _ = _owner_id_from_config(config)
    with SessionLocal() as db:
        ob = db.query(OutboxEmail).filter(OutboxEmail.id == draft_id).one()

        # Apply overrides before approval/send
        changed_subject = changed_text = changed_any = False

        if subject and subject != ob.subject:
            ob.subject = subject
            changed_subject = changed_any = True

        if text and text != ob.text_body:
            ob.text_body = text
            changed_text = changed_any = True

        if to and to != ob.to_email:
            ob.to_email = to
            changed_any = True

        if to_name is not None and to_name != ob.to_name:
            ob.to_name = to_name
            changed_any = True

        if changed_subject or changed_text:
            ob.preview_html = render_basic_html(ob.subject, ob.text_body.splitlines())

        if changed_any:
            db.commit()
            db.refresh(ob)

        # Apply recipients (optional)
        if recipients is not None:
            if replace_recipients:
                db.query(OutboxEmailRecipient).filter(
                    OutboxEmailRecipient.outbox_id == ob.id
                ).delete()
            for r in recipients or []:
                email = (
                    r.get("email") if isinstance(r, dict) else getattr(r, "email", None)
                )
                name = (
                    r.get("name") if isinstance(r, dict) else getattr(r, "name", None)
                )
                if email:
                    db.add(
                        OutboxEmailRecipient(
                            outbox_id=ob.id, email=str(email), name=name
                        )
                    )
            db.commit()
            db.refresh(ob)

        # Approve if requested
        if approve and ob.status != OutboxEmailStatus.APPROVED.value:
            ob.status = OutboxEmailStatus.APPROVED.value
            ob.approved_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(ob)

        # Send if approved — to ALL recipients (anchor + list), deduped
        if ob.status == OutboxEmailStatus.APPROVED.value:
            all_emails = set()
            if ob.to_email:
                all_emails.add(ob.to_email)
            for r in ob.recipients or []:
                if r.email:
                    all_emails.add(r.email)

            for addr in sorted(all_emails):
                send_email(
                    to=addr, subject=ob.subject, text=ob.text_body, html=ob.preview_html
                )

            ob.status = OutboxEmailStatus.SENT.value
            ob.sent_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(ob)
            return {"ok": True, "status": ob.status}

        return {
            "ok": ob.status.lower() == OutboxEmailStatus.SENT.value,
            "status": ob.status,
        }
