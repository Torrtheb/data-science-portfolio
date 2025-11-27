from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime, timezone
from sqlalchemy.orm import Session, joinedload

from app.db import get_db
from app.core.auth import require_owner, TokenUser
from app.models import OutboxEmail, OutboxEmailStatus, OutboxEmailRecipient
from services.emailer import send_email, render_basic_html
import sqlalchemy as sa
from typing import Optional, List

router = APIRouter(prefix="/api/outbox", tags=["outbox"])


class RecipientIn(BaseModel):
    """Recipient input with email and optional display name."""

    email: EmailStr
    name: Optional[str] = None


class RecipientOut(BaseModel):
    """Recipient output model (email + optional name)."""

    email: EmailStr
    name: str | None = None


class SendBody(BaseModel):
    """Draft update/send payload.

    Fields:
      - approve: when true, mark draft approved before sending
      - to / to_name: legacy single-recipient fields (kept for compatibility)
      - subject / text: content overrides
      - recipients: list of new recipients to merge/replace
      - replace_recipients: if true, replace; otherwise merge distinct emails
    """

    approve: bool = True
    to: Optional[str] = None
    to_name: Optional[str] = None
    subject: Optional[str] = None
    text: Optional[str] = None
    recipients: Optional[List[RecipientIn]] = None
    replace_recipients: bool = False


class RejectBody(BaseModel):
    """Reject draft payload containing a reason for auditing/display."""

    reason: str


class DraftOut(BaseModel):
    """Outbox draft representation returned by the API."""

    id: str
    owner_user_id: str
    to: EmailStr | None
    to_name: str | None
    recipients: list[RecipientOut]
    subject: str
    text: str
    status: str
    created_at: datetime | None = None
    approved_at: datetime | None = None
    sent_at: datetime | None = None
    rejected_reason: str | None = None


def _ensure_owner(ob: OutboxEmail, me_id: str) -> None:
    """Ensure the outbox draft belongs to the current owner; raise 403 otherwise."""
    if ob.owner_user_id != me_id:
        raise HTTPException(status_code=403, detail="Forbidden")


def _as_out(ob: OutboxEmail) -> DraftOut:
    """Serialize ORM OutboxEmail row into the API DraftOut shape."""
    return DraftOut(
        id=str(ob.id),
        owner_user_id=ob.owner_user_id,
        to=ob.to_email,
        to_name=ob.to_name,
        recipients=[
            RecipientOut(email=r.email, name=r.name) for r in (ob.recipients or [])
        ],
        subject=ob.subject,
        text=ob.text_body,
        status=ob.status,
        created_at=ob.created_at,
        approved_at=ob.approved_at,
        sent_at=ob.sent_at,
        rejected_reason=ob.rejected_reason,
    )


def _apply_recipients(ob: OutboxEmail, body: SendBody) -> bool:
    """Apply recipient changes to a draft, returning True if any changes were made."""
    changed = False

    if body.recipients is not None:
        if body.replace_recipients:
            ob.recipients.clear()
            for r in body.recipients:
                ob.recipients.append(
                    OutboxEmailRecipient(email=str(r.email), name=r.name)
                )
            changed = True
        else:
            existing = {r.email.lower() for r in ob.recipients}
            for r in body.recipients:
                if r.email.lower() not in existing:
                    ob.recipients.append(
                        OutboxEmailRecipient(email=str(r.email), name=r.name)
                    )
                    existing.add(r.email.lower())
                    changed = True
    if body.to and body.to != ob.to_email:
        ob.to_email = body.to
        changed = True
    if body.to_name is not None and body.to_name != ob.to_name:
        ob.to_name = body.to_name
        changed = True

    return changed


# ---------- Routes ----------
@router.get("", response_model=list[DraftOut])
def list_outbox(
    status: str | None = Query(None, description="pending|approved|rejected|sent"),
    q: str | None = Query(None, description="search subject or recipient email"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> list[DraftOut]:
    """List drafts for the owner with optional status filter and search.

    Search matches subject and recipient email (case-insensitive). Results are
    ordered by creation time descending and support limit/offset pagination.
    """
    query = (
        db.query(OutboxEmail)
        .options(joinedload(OutboxEmail.recipients))
        .filter(OutboxEmail.owner_user_id == user.sub)
        .order_by(OutboxEmail.created_at.desc())
    )
    if status:
        query = query.filter(OutboxEmail.status == status)

    if q:
        pattern = f"%{q.lower()}%"
        query = query.outerjoin(
            OutboxEmailRecipient, OutboxEmailRecipient.outbox_id == OutboxEmail.id
        )
        query = query.filter(
            sa.or_(
                sa.func.lower(OutboxEmail.subject).like(pattern),
                sa.func.lower(OutboxEmailRecipient.email).like(pattern),
            )
        )

    rows = query.limit(limit).offset(offset).all()
    return [_as_out(ob) for ob in rows]


@router.get("/{draft_id}", response_model=DraftOut)
def get_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> DraftOut:
    """Return a single outbox draft by id for the owner."""
    ob = (
        db.query(OutboxEmail)
        .options(joinedload(OutboxEmail.recipients))
        .filter(OutboxEmail.id == draft_id)
        .first()
    )
    if not ob:
        raise HTTPException(status_code=404, detail="Draft not found")
    _ensure_owner(ob, user.sub)
    return _as_out(ob)


@router.put("/{draft_id}", response_model=DraftOut)
def update_draft(
    draft_id: UUID,
    body: SendBody,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> DraftOut:
    """Update draft fields (subject, text, recipients), without sending.
    Enforces conservative size limits and rebuilds HTML preview when content
    changes. Drafts already sent or rejected are immutable.
    """
    ob = (
        db.query(OutboxEmail)
        .options(joinedload(OutboxEmail.recipients))
        .filter(OutboxEmail.id == draft_id)
        .first()
    )
    if not ob:
        raise HTTPException(status_code=404, detail="Draft not found")
    _ensure_owner(ob, user.sub)

    if ob.status in (OutboxEmailStatus.SENT.value, OutboxEmailStatus.REJECTED.value):
        raise HTTPException(status_code=400, detail="Draft is not editable")
    import os

    MAX_SUBJECT = int(os.getenv("MAX_EMAIL_SUBJECT", "200") or 200)
    MAX_TEXT = int(os.getenv("MAX_EMAIL_TEXT", "10000") or 10000)
    MAX_RECIPIENTS = int(os.getenv("MAX_EMAIL_RECIPIENTS", "200") or 200)

    changed_subject = False
    changed_text = False
    changed_any = False

    if body.subject and body.subject != ob.subject:
        if len(body.subject) > MAX_SUBJECT:
            raise HTTPException(status_code=413, detail="Subject too long")
        ob.subject = body.subject
        changed_subject = True
        changed_any = True

    if body.text and body.text != ob.text_body:
        if len(body.text) > MAX_TEXT:
            raise HTTPException(status_code=413, detail="Body too long")
        ob.text_body = body.text
        changed_text = True
        changed_any = True
    if body.recipients is not None and len(body.recipients) > MAX_RECIPIENTS:
        raise HTTPException(status_code=413, detail="Too many recipients")
    if _apply_recipients(ob, body):
        changed_any = True

    if changed_subject or changed_text:
        ob.preview_html = render_basic_html(ob.subject, ob.text_body.splitlines())

    if changed_any:
        db.commit()
        db.refresh(ob)

    return _as_out(ob)


@router.post("/{draft_id}/send", response_model=DraftOut)
def send_outbox_email(
    draft_id: UUID,
    body: SendBody,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> DraftOut:
    """Approve (optional) and send a draft to all recipients.

    Applies subject/text/recipient updates subject to size limits. When
    approved, sends to the union of the anchor 'to_email' and the recipients
    list, then marks the draft as 'sent' with a timestamp.
    """
    ob = (
        db.query(OutboxEmail)
        .options(joinedload(OutboxEmail.recipients))
        .filter(OutboxEmail.id == draft_id)
        .first()
    )
    if not ob:
        raise HTTPException(status_code=404, detail="Draft not found")

    _ensure_owner(ob, user.sub)

    if ob.status == OutboxEmailStatus.SENT.value:
        return _as_out(ob)
    import os

    MAX_SUBJECT = int(os.getenv("MAX_EMAIL_SUBJECT", "200") or 200)
    MAX_TEXT = int(os.getenv("MAX_EMAIL_TEXT", "10000") or 10000)
    MAX_RECIPIENTS = int(os.getenv("MAX_EMAIL_RECIPIENTS", "200") or 200)

    changed_subject = False
    changed_text = False
    changed_any = False
    if body.subject and body.subject != ob.subject:
        if len(body.subject) > MAX_SUBJECT:
            raise HTTPException(status_code=413, detail="Subject too long")
        ob.subject = body.subject
        changed_subject = True
        changed_any = True

    if body.text and body.text != ob.text_body:
        if len(body.text) > MAX_TEXT:
            raise HTTPException(status_code=413, detail="Body too long")
        ob.text_body = body.text
        changed_text = True
        changed_any = True

    if body.to and body.to != ob.to_email:
        ob.to_email = body.to
        changed_any = True

    if body.to_name is not None and body.to_name != ob.to_name:
        ob.to_name = body.to_name
        changed_any = True
    if body.recipients is not None and len(body.recipients) > 0:
        if len(body.recipients) > MAX_RECIPIENTS:
            raise HTTPException(status_code=413, detail="Too many recipients")
        if body.replace_recipients:
            db.query(OutboxEmailRecipient).filter(
                OutboxEmailRecipient.outbox_id == ob.id
            ).delete()
        for r in body.recipients:
            db.add(
                OutboxEmailRecipient(outbox_id=ob.id, email=str(r.email), name=r.name)
            )
        changed_any = True
    if changed_subject or changed_text:
        ob.preview_html = render_basic_html(ob.subject, ob.text_body.splitlines())

    if changed_any:
        db.commit()
        db.refresh(ob)
    if body.approve and ob.status != OutboxEmailStatus.APPROVED.value:
        ob.status = OutboxEmailStatus.APPROVED.value
        ob.approved_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(ob)
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

    return _as_out(ob)


@router.post("/{draft_id}/reject", response_model=DraftOut)
def reject_outbox_email(
    draft_id: UUID,
    body: RejectBody,
    db: Session = Depends(get_db),
    user: TokenUser = Depends(require_owner),
) -> DraftOut:
    """Reject a draft with an audit reason; immutable after rejection."""
    ob = (
        db.query(OutboxEmail)
        .options(joinedload(OutboxEmail.recipients))
        .filter(OutboxEmail.id == draft_id)
        .first()
    )
    if not ob:
        raise HTTPException(status_code=404, detail="Draft not found")
    _ensure_owner(ob, user.sub)

    if ob.status == OutboxEmailStatus.SENT.value:
        raise HTTPException(status_code=400, detail="Already sent")

    ob.status = OutboxEmailStatus.REJECTED.value
    ob.rejected_reason = (body.reason or "").strip() or None
    db.commit()
    db.refresh(ob)
    return _as_out(ob)
