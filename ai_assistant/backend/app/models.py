from __future__ import annotations
from datetime import datetime, time
import enum
import uuid
from typing import Optional

from sqlalchemy import (
    Integer,
    String,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Time,
    Enum,
    func,
    Text,
    SmallInteger,
    CheckConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import text
from sqlalchemy import Enum as SAEnum
import sqlalchemy as sa


# ---------- Enums ----------
class RoleEnum(str, enum.Enum):
    """Application role for users within 'auth.User'.

    Values control access scopes across the application. Stored as a Postgres
    enum via Prisma in the 'auth' schema and referenced from SQLAlchemy.
    """

    OWNER = "OWNER"
    STAFF = "STAFF"
    CLIENT = "CLIENT"


AttendanceEnum = Enum("unknown", "attended", "late", "no_show", name="attendance_enum")
PaymentEnum = Enum("unpaid", "paid", "refunded", "waived", name="payment_enum")


# ---------- Prisma-managed auth user (read-only from SQLAlchemy side) ----------
class User(Base):
    __tablename__ = "User"
    __table_args__ = {"schema": "auth"}
    """Auth user (Prisma-managed) mirrored locally for joins and ownership checks.

    Notes:
    - 'role' maps to the Postgres enum 'auth."Role"' managed by Prisma
    - 'timezone' is the owner’s canonical timezone for local scheduling
    - 'appt_edge_buffer_min' sets the minimum edge buffer between appointments
    """
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    emailVerified: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    image: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    password: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    role: Mapped[RoleEnum] = mapped_column(
        SAEnum(
            RoleEnum,
            name="Role",
            schema="auth",
            native_enum=True,
            create_type=False,
            validate_strings=True,
        ),
        nullable=False,
    )
    timezone: Mapped[str] = mapped_column(
        String, nullable=False, default="America/Toronto"
    )
    createdAt: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updatedAt: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    appt_edge_buffer_min: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="5"
    )
    group_price_60_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


# ---------- NEW: Client accounts, emails & people ----------
class ClientAccount(Base):
    __tablename__ = "client_accounts"
    """Owner-scoped client account aggregating emails and people (household).
        Relationships:
        - people: list[Person]
        - emails: list[ClientEmail]

    Represents a top-level client container under an owner’s book. One account
    can hold multiple 'Person' rows (e.g., family members) and multiple
    'ClientEmail' rows, one of which may be primary for messaging.
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    owner_user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    client_user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )

    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    emergency_contact: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    people: Mapped[list["Person"]] = relationship(
        "Person", back_populates="account", cascade="all, delete-orphan"
    )
    emails: Mapped[list["ClientEmail"]] = relationship(
        "ClientEmail", back_populates="account", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("owner_user_id", "client_user_id", name="uq_owner_client"),
    )


class ClientEmail(Base):
    __tablename__ = "client_emails"
    """Email address on a client account; one may be marked primary.

    Primary flag helps pick a default contact address for messaging workflows.
    Uniqueness is enforced per account.
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    account_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("client_accounts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    email: Mapped[str] = mapped_column(String, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Integer, nullable=False, default=1)
    unsubscribed: Mapped[bool] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )

    account: Mapped["ClientAccount"] = relationship(
        "ClientAccount", back_populates="emails"
    )
    __table_args__ = (UniqueConstraint("account_id", "email", name="uq_account_email"),)


class Person(Base):
    __tablename__ = "people"
    """Individual under a client account who attends lessons/appointments."""
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    account_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("client_accounts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    account: Mapped["ClientAccount"] = relationship(
        "ClientAccount", back_populates="people"
    )


# ---------- Availability ----------
class AvailabilityRule(Base):
    __tablename__ = "availability_rules"
    """Weekly recurring availability window in owner-local time.

    Each rule defines a weekday, start/end (local), and the default slot and
    buffer lengths used to generate openings unless overridden by specials.
    """
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "weekday",
            "start_local",
            "end_local",
            "slot_minutes",
            "buffer_minutes",
            name="uq_rule_owner_day_window_len_buf",
        ),
    )
    id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    weekday: Mapped[int] = mapped_column(Integer, nullable=False)
    start_local: Mapped[time] = mapped_column(Time, nullable=False)
    end_local: Mapped[time] = mapped_column(Time, nullable=False)
    slot_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    buffer_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


# ---------- Time off ----------
class TimeOff(Base):
    __tablename__ = "timeoffs"
    """Owner time-off window in UTC that blocks bookings and generated openings."""
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(String)
    __table_args__ = (
        Index("ix_timeoff_owner_start_end", "owner_id", "start_utc", "end_utc"),
    )


# ---------- Appointments ----------
class Appointment(Base):
    __tablename__ = "appointments"
    """Appointment/lesson with identity, attendance, payment, and grouping state.

    Key invariants and fields:
    - Identity: either 'person_id' OR both ('client_name', 'client_email') required
    - Status: 'booked' | 'completed' | 'canceled' (guarded by constraint)
    - Attendance: tracks status and optional 'late_minutes'
    - Payment: 'payment_status', optional 'bundle_id', amounts and overrides
    - Grouping: optional 'group_id' to group attendees of the same session
    - Double-booking prevention via partial unique index on active appointments
    """

    id: Mapped["uuid.UUID"] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        nullable=False,
    )

    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    client_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )

    person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("people.id", ondelete="SET NULL"), index=True, nullable=True
    )

    client_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    client_email: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )

    start_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    status: Mapped[str] = mapped_column(String, nullable=False, default="booked")
    client_previsit_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    client_change_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cancel_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    owner_private_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attendance_status: Mapped[str] = mapped_column(
        AttendanceEnum,
        nullable=False,
        default="attended",
        server_default="attended",
    )
    late_minutes: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="0"
    )

    payment_status: Mapped[str] = mapped_column(
        PaymentEnum, nullable=False, server_default="unpaid"
    )
    paid_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    bundle_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("prepaid_bundles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    amount_paid_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    price_override_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Optional grouping for owner-managed group lessons (same id across attendees)
    group_id: Mapped[Optional["uuid.UUID"]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    person: Mapped[Optional["Person"]] = relationship(
        "Person", lazy="joined", foreign_keys=[person_id]
    )
    owner: Mapped["User"] = relationship("User", foreign_keys=[owner_id])
    client_user: Mapped[Optional["User"]] = relationship(
        "User", foreign_keys=[client_id]
    )

    __table_args__ = (
        Index(
            "ix_owner_start_person_active_unique",
            "owner_id",
            "start_utc",
            "person_id",
            unique=True,
            postgresql_where=text("status <> 'canceled'"),
        ),
        CheckConstraint(
            "status IN ('booked','completed','canceled')",
            name="appointments_status_check",
        ),
        CheckConstraint(
            "(person_id IS NOT NULL) OR (client_name IS NOT NULL AND client_email IS NOT NULL)",
            name="ck_appointments_identity",
        ),
        Index("ix_appt_owner_start_end", "owner_id", "start_utc", "end_utc"),
    )


# ---------- Special one-off openings ----------
class SpecialOpening(Base):
    __tablename__ = "special_openings"
    """One-off opening window for availability outside weekly rules."""
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    slot_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    buffer_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    note: Mapped[Optional[str]] = mapped_column(String)
    __table_args__ = (
        Index("ix_special_open_owner_start_end", "owner_id", "start_utc", "end_utc"),
    )


# ---------- Prepaid credit bundles ----------
class PrepaidBundle(Base):
    __tablename__ = "prepaid_bundles"
    """Credit pack or wallet for a client under an owner.

    'total_credits=0' indicates a wallet (store credit). 'remaining_credits'
    decreases when appointments are consumed and can be refilled via ledger.
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    client_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="Bundle")
    total_credits: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    remaining_credits: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USD")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("total_credits >= 0", name="ck_bundle_total_nonneg"),
        CheckConstraint("remaining_credits >= 0", name="ck_bundle_remaining_nonneg"),
        Index(
            "uq_wallet_active_owner_client",
            "owner_id",
            "client_id",
            unique=True,
            postgresql_where=sa.text("total_credits = 0 AND status = 'active'"),
        ),
    )


class PrepaidLedger(Base):
    __tablename__ = "prepaid_ledger"
    """Ledger for credit and wallet movements.

    Records positive deltas for purchases/deposits and negative deltas for
    consumption or refunds. Optionally links to an 'Appointment' when applied.
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bundle_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("prepaid_bundles.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    event: Mapped[str] = mapped_column(String(20), nullable=False)
    delta_credits: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    amount_cents: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    appointment_id: Mapped[Optional["uuid.UUID"]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("appointments.id", ondelete="SET NULL"),
        nullable=True,
    )
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class OwnerFeeSetting(Base):
    __tablename__ = "owner_fee_settings"
    """Per-owner settings for administration fee amount and metadata."""

    owner_id: Mapped[str] = mapped_column(
        String, ForeignKey("auth.User.id", ondelete="CASCADE"), primary_key=True
    )
    admin_fee_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=1500)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AdminFeeStatus(str, enum.Enum):
    """Admin fee lifecycle states used by 'AdminFeeCharge'."""

    UNPAID = "unpaid"
    BUNDLE = "bundle"
    REFUNDED = "refunded"
    WAIVED = "waived"
    PAID = "paid"


class AdminFeeCharge(Base):
    __tablename__ = "admin_fee_charges"
    """Standalone administration fee charge attached to a client account.

    Tracks payment state across cash and bundle applications with simple
    non-negative amount constraints and timestamps for auditing.
    """

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    client_account_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("client_accounts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    client_user_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[AdminFeeStatus] = mapped_column(
        SAEnum(
            AdminFeeStatus,
            name="admin_fee_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=AdminFeeStatus.UNPAID,
        server_default=AdminFeeStatus.UNPAID.value,
    )
    paid_cash_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bundle_applied_cents: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("amount_cents >= 0", name="ck_admin_fee_amount_nonneg"),
        CheckConstraint("paid_cash_cents >= 0", name="ck_admin_fee_cash_nonneg"),
        CheckConstraint("bundle_applied_cents >= 0", name="ck_admin_fee_bundle_nonneg"),
    )


# ---------- Sticky notes on a client (owner-only) ----------
class ClientNote(Base):
    __tablename__ = "client_notes"
    """Owner-private note attached to a client user profile."""
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    client_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    note: Mapped[str] = mapped_column(Text, nullable=False)
    pinned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ServiceOption(Base):
    __tablename__ = "service_options"
    """Owner-defined mapping from service duration to price.

    Used for pricing hints and UI suggestions; restricted to a fixed set of
    allowed durations, with currency and active flag for visibility.
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    duration_minutes: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USD")
    is_active: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("owner_id", "duration_minutes", name="uq_owner_duration"),
        CheckConstraint(
            "duration_minutes IN (15,30,45,60,120)", name="ck_duration_allowed"
        ),
        CheckConstraint("price_cents >= 0", name="ck_price_nonneg"),
    )


# ---------- Email Outbox (owner-approval for agent emails) ----------
class OutboxEmailStatus(str, enum.Enum):
    """Statuses for the outbound email approval pipeline."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SENT = "sent"


class OutboxEmail(Base):
    __tablename__ = "outbox_emails"
    """Owner-approval email draft stored for later sending and auditing.

    The agent proposes content and recipients; owners review and approve.
    Emails may have multiple recipients via 'OutboxEmailRecipient' rows and
    track lifecycle timestamps for auditing.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    owner_user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("auth.User.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    to_email: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    to_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    subject: Mapped[str] = mapped_column(String, nullable=False)
    recipients: Mapped[list["OutboxEmailRecipient"]] = relationship(
        "OutboxEmailRecipient",
        back_populates="outbox",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    text_body: Mapped[str] = mapped_column(Text, nullable=False)
    preview_html: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(
        SAEnum(
            OutboxEmailStatus,
            name="outboxemailstatus",
            native_enum=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        server_default=OutboxEmailStatus.PENDING.value,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approved_by: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("auth.User.id", ondelete="SET NULL"), nullable=True
    )
    sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    rejected_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class OutboxEmailRecipient(Base):
    __tablename__ = "outbox_email_recipients"
    """Recipient row belonging to a single 'OutboxEmail' (unique per email)."""

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True, autoincrement=True)
    outbox_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("outbox_emails.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    email: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    outbox: Mapped["OutboxEmail"] = relationship(
        "OutboxEmail", back_populates="recipients"
    )

    __table_args__ = (
        sa.UniqueConstraint("outbox_id", "email", name="uq_outbox_recipient_per_email"),
    )
