from __future__ import annotations
from datetime import date, time, datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Literal, Dict
from uuid import UUID
from enum import Enum


# ---- Owner (is a User with role=OWNER) --------------------------------------
class OwnerCreate(BaseModel):
    """Payload to create an owner account."""

    name: str
    email: EmailStr
    timezone: str = "America/Toronto"


class OwnerOut(BaseModel):
    """Owner profile returned from the API."""

    id: str
    name: str
    email: EmailStr
    timezone: str
    model_config = ConfigDict(from_attributes=True)


# ---- Client views (User with role=CLIENT) -----------------------------------
class ClientOut(BaseModel):
    """Minimal client identity used in responses.

    Notes:
    - Some owner-facing listings may include client name/email even when the
      appointment is not linked to an auth user yet. In those cases, `id` can
      be absent. Making `id` optional keeps responses consistent without
      forcing placeholder IDs.
    """

    id: str | None = None
    name: str | None
    email: EmailStr | None
    model_config = ConfigDict(from_attributes=True)


# ---- Availability & Time-off ------------------------------------------------
class AvailabilityCreate(BaseModel):
    """Create payload for a weekly recurring availability rule."""

    weekday: int = Field(ge=0, le=6)
    start_local: str
    end_local: str
    slot_minutes: int = 30
    buffer_minutes: int = 0

    def get_start_time(self) -> time:
        """Convert 'start_local' string (HH:MM) to a 'time' object."""
        h, m = self.start_local.split(":")
        return time(hour=int(h), minute=int(m))

    def get_end_time(self) -> time:
        """Convert 'end_local' string (HH:MM) to a 'time' object."""
        h, m = self.end_local.split(":")
        return time(hour=int(h), minute=int(m))


class AvailabilityOut(BaseModel):
    """Availability rule formatted for API responses."""

    id: str
    weekday: int
    start_local: str
    end_local: str
    slot_minutes: int
    buffer_minutes: int

    @classmethod
    def from_db_model(cls, rule):
        """Convert a DB model instance into an API schema."""
        return cls(
            id=rule.id,
            weekday=rule.weekday,
            start_local=rule.start_local.strftime("%H:%M"),
            end_local=rule.end_local.strftime("%H:%M"),
            slot_minutes=rule.slot_minutes,
            buffer_minutes=rule.buffer_minutes,
        )

    model_config = ConfigDict(from_attributes=True)


class TimeOffCreate(BaseModel):
    """Create payload for a time-off window in UTC."""

    start: datetime
    end: datetime
    note: str | None = None


class TimeOffOut(BaseModel):
    """Time-off window formatted for API responses (UTC and local)."""

    id: str
    start_utc: datetime
    end_utc: datetime
    start_local: datetime | None = None
    end_local: datetime | None = None
    timezone: str | None = None
    note: str | None

    model_config = ConfigDict(from_attributes=True)


# ---- Slots & Booking --------------------------------------------------------
class SlotsQuery(BaseModel):
    """Query parameters for listing available slots on a date."""

    date: date


class Slot(BaseModel):
    """An individual availability slot in owner-local time."""

    start: datetime
    end: datetime


class BookRequest(BaseModel):
    """Client request to book a slot in owner-local time."""

    client_name: str
    client_email: EmailStr
    start_local: datetime
    message: Optional[str] = None


class AppointmentOut(BaseModel):
    """Appointment response payload with attendance and payment details."""

    id: str
    start_utc: datetime
    end_utc: datetime
    status: str
    client: ClientOut | None
    client_previsit_note: Optional[str] = None
    cancel_reason: Optional[str] = None
    owner_private_note: Optional[str] = None
    attendance_status: Optional[str] = None
    late_minutes: Optional[int] = None
    payment_status: Optional[str] = None
    paid_at: Optional[datetime] = None
    bundle_id: Optional[int] = None
    amount_paid_cents: Optional[int] = None
    price_override_cents: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


Attendance = Literal["unknown", "attended", "late", "no_show"]
Payment = Literal["unpaid", "paid", "refunded", "waived"]


class AppointmentUpdateClient(BaseModel):
    """Client-editable fields for an existing appointment."""

    client_previsit_note: Optional[str] = None


class AppointmentCancel(BaseModel):
    """Client/owner cancellation request with a reason."""

    reason: str = Field(min_length=1, max_length=1000)


class AppointmentUpdateOwner(BaseModel):
    """Owner-editable fields for an existing appointment."""

    owner_private_note: Optional[str] = None
    attendance_status: Optional[str] = Field(
        None, pattern="^(unknown|attended|late|no_show)$"
    )
    late_minutes: Optional[int] = None
    payment_status: Optional[str] = Field(
        None, pattern="^(unpaid|paid|refunded|waived)$"
    )
    status: Optional[str] = Field(None, pattern="^(booked|completed|canceled)$")
    paid_at: Optional[datetime] = None
    bundle_id: Optional[int] = None
    amount_paid_cents: Optional[int] = Field(None, ge=0)
    price_override_cents: Optional[int] = Field(None, ge=0)


# --- People ---
class PersonCreate(BaseModel):
    """Create payload for a 'Person' within a client account."""

    full_name: str = Field(min_length=1, max_length=200)
    email: Optional[EmailStr] = None


class PersonOut(BaseModel):
    """Person record returned from the API."""

    id: int
    full_name: str
    email: Optional[str]
    model_config = ConfigDict(from_attributes=True)


# --- Owner client listings ---
class ClientAccountSummary(BaseModel):
    """Owner-facing summary for listing client accounts."""

    account_id: int
    client_user_id: Optional[str] = None
    client_email: Optional[str] = None
    client_name: Optional[str] = None
    people_count: int
    name: Optional[str] = None


class ClientAccountDetail(BaseModel):
    """Owner-facing detailed view of a client account, emails, and people."""

    account_id: int
    client_user_id: str
    client_email: Optional[str] = None
    client_name: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None
    emails: List["ClientEmailOut"] = Field(default_factory=list)
    people: List[PersonOut]


# --- Profile (client self-service) ---
class ProfileOut(BaseModel):
    """Client self-service profile including associated 'Person' entries."""

    user_id: str
    email: Optional[EmailStr]
    name: Optional[str]
    people: List[PersonOut]


# --- Payload to create a client under the owner ---
class ClientCreate(BaseModel):
    """Payload to create a client under an owner."""

    email: EmailStr
    name: Optional[str] = None


class SpecialOpeningCreate(BaseModel):
    """Create payload for a one-off opening window (owner-local time)."""

    start: datetime
    end: datetime
    slot_minutes: int = 30
    buffer_minutes: int = 0
    note: Optional[str] = None
    allow_overlap: bool = False


class SpecialOpeningOut(BaseModel):
    """One-off opening window formatted for API responses."""

    id: str
    start_utc: datetime
    end_utc: datetime
    start_local: datetime | None = None
    end_local: datetime | None = None
    timezone: str | None = None
    slot_minutes: int
    buffer_minutes: int
    note: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------- Owner calendar snapshot + filtered appointments ------------
class CalendarScope(str, Enum):
    """Predefined ranges used to scope calendar snapshots."""

    today = "today"
    week = "week"
    month = "month"


class SnapshotOut(BaseModel):
    """Calendar snapshot payload including computed event list."""

    tz: str
    start: datetime
    end: datetime
    events: list["CalendarEventOut"]


class CalendarEventOut(BaseModel):
    """Calendar event (appointment, opening, or time-off) for UI display."""

    id: str
    type: Literal["appointment", "opening", "time_off"]
    title: str
    start: datetime
    end: datetime
    status: Optional[str] = None
    meta: Optional[Dict[str, str]] = None


class AppointmentFilter(str, Enum):
    """Filter buckets for owner appointment lists and reports."""

    today = "today"
    this_week = "this_week"
    this_month = "this_month"
    cancelled = "cancelled"
    completed_last_week = "completed_last_week"
    completed_last_month = "completed_last_month"
    completed_all_time = "completed_all_time"


class AppointmentRow(BaseModel):
    """Compact row for appointment lists."""

    id: str
    title: str
    start: datetime
    end: datetime
    status: str
    client_name: Optional[str] = None
    client_email: Optional[str] = None
    needs_edit: bool = False


class AppointmentListOut(BaseModel):
    """List wrapper for appointment rows."""

    rows: list[AppointmentRow]


class AppointmentPersonOut(BaseModel):
    """Person details attached to an appointment (owner-facing)."""

    id: int
    name: Optional[str] = None
    email: Optional[str] = None


class OwnerAppointmentRowOut(BaseModel):
    """Rich owner-facing appointment row used by admin listing endpoints."""

    id: str
    start_utc: datetime
    end_utc: datetime
    group_id: Optional[str] = None
    start_local: datetime
    end_local: datetime
    timezone: str
    status: str
    client: Optional[ClientOut] = None
    client_account_id: Optional[int] = None
    person: Optional[AppointmentPersonOut] = None
    cancel_reason: Optional[str] = None
    owner_note: Optional[str] = None
    client_note: Optional[str] = None
    paid: bool
    late: bool
    no_show: bool
    amount_paid_cents: Optional[int] = None
    labels: Optional[Dict[str, str]] | None = None
    attendance_status: Optional[str] = None
    late_minutes: int = 0
    payment_status: Optional[str] = None
    paid_at: Optional[datetime] = None
    bundle_id: Optional[int] = None
    price_override_cents: Optional[int] = None
    effective_price_cents: Optional[int] = None


# --- Email DTOs ---
class ClientEmailOut(BaseModel):
    """Client email record with primary flag."""

    id: int
    email: EmailStr
    is_primary: bool
    model_config = ConfigDict(from_attributes=True)


class ClientEmailIn(BaseModel):
    """Client email input with optional primary flag."""

    email: EmailStr
    is_primary: bool = False


# --- Client Profile (self-service) ---
class ClientProfileOut(BaseModel):
    """Client-facing profile view with emails and people."""

    account_id: int
    name: Optional[str] = None
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None
    emails: List["ClientEmailOut"] = Field(default_factory=list)
    people: List["PersonOut"] = Field(default_factory=list)
    model_config = ConfigDict(from_attributes=True)


class ClientProfileUpdate(BaseModel):
    """Update payload for client-facing profile edits."""

    name: Optional[str] = None
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None
    emails: List[ClientEmailIn] = Field(default_factory=list)


class MyBookPayload(BaseModel):
    """Client booking payload for self-service scheduling."""

    start_local: datetime
    duration_minutes: Optional[int] = None
    message: Optional[str] = None


class MyUpdatePayload(BaseModel):
    """Client edits to a booked appointment (reschedule/cancel)."""

    start_local: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    status: Optional[Literal["canceled"]] = None
    message: Optional[str] = None


class AdminCreateAppt(BaseModel):
    """Admin path for creating a new appointment on behalf of a client."""

    client_name: str
    client_email: EmailStr
    start_local: datetime
    duration_minutes: int = 30
    status: Literal["booked", "completed", "canceled"] = "booked"
    allow_override: bool = False
    message: Optional[str] = None


# --- owner broadcast email request ---
class BroadcastEmailRequest(BaseModel):
    """Owner broadcast email request for multiple client recipients.

    Supports HTML or plain text body. When 'preview_only' is true, the server
    returns a dry-run summary without sending emails.
    """

    subject: str = Field(min_length=1)
    html: Optional[str] = None
    text: Optional[str] = None
    client_user_ids: Optional[List[str]] = None
    preview_only: bool = False


class PrepaidBundleCreate(BaseModel):
    """Create payload for a credit bundle or wallet deposit."""

    client_id: str
    name: str = "Bundle"
    total_credits: int = Field(ge=0, le=200)
    price_cents: int = Field(ge=0)
    currency: str = "USD"
    expires_at: Optional[datetime] = None


class PrepaidBundleOut(BaseModel):
    """Bundle/wallet representation returned after creation or fetch."""

    id: int
    client_id: str
    name: str
    total_credits: int
    remaining_credits: int
    remaining_balance_cents: Optional[int] = None
    price_cents: int
    currency: str
    status: str
    expires_at: Optional[datetime]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# --- Service options (duration → price mapping) ---


class PricingMap(BaseModel):
    """Map of duration (minutes as string) to price in cents."""

    prices: Dict[str, int] = Field(default_factory=dict)


class ServiceOptionCreate(BaseModel):
    """Create payload for a single service duration → price mapping."""

    duration_minutes: int
    price_cents: int = Field(ge=0)
    currency: str = "USD"
    is_active: bool = True


class ServiceOptionOut(BaseModel):
    """Service option returned from the API."""

    id: int
    duration_minutes: int
    price_cents: int
    currency: str
    is_active: bool
    model_config = ConfigDict(from_attributes=True)


class ClientApptRow(BaseModel):
    """Client-facing appointment row with payment and price context."""

    id: UUID
    start_utc: datetime
    end_utc: datetime
    status: Optional[Literal["booked", "completed", "canceled"]] = None
    payment_status: Literal[
        "unpaid", "paid", "refunded", "waived", "partial", "unknown", "bundle"
    ]
    amount_paid_cents: Optional[int] = None
    duration_minutes: Optional[int] = None
    price_cents: Optional[int] = None


class ClientApptListOut(BaseModel):
    """List wrapper for client-facing appointment rows."""

    rows: List[ClientApptRow]


# ---------------- Client payments (client-facing) ----------------
# Allow all statuses that can be produced by compute_financials(),
# including refunded/waived for explicit overrides.
PaymentStatusLiteral = Literal[
    "paid",
    "partial",
    "unpaid",
    "bundle",
    "unknown",
    "refunded",
    "waived",
]


class AppointmentPaymentRow(BaseModel):
    """Payment view row for an appointment including computed amounts."""

    id: str
    start_utc: str
    duration_minutes: int
    status: Literal["booked", "completed", "canceled"]
    attendance: Optional[
        Literal["unknown", "on_time", "late", "no_show", "attended"]
    ] = None
    lesson_person_name: Optional[str] = None
    is_group: Optional[bool] = None
    price_cents: Optional[int] = None
    amount_paid_cents: int = 0
    bundle_applied_cents: int = 0
    payment_status: PaymentStatusLiteral


class ClientPaymentsSummary(BaseModel):
    """Aggregate payment totals for a single client account."""

    total_appointments: int
    late_appointments: int
    paid_appointments: int
    unpaid_appointments: int
    total_expected_cents: int
    total_paid_cents: int
    total_owed_cents: int


class ClientPaymentsOut(BaseModel):
    """Client payments response containing summary and detailed rows."""

    summary: ClientPaymentsSummary
    rows: List[AppointmentPaymentRow]


class OwnerClientAggregate(BaseModel):
    """Per-client rollup used in owner payments analytics."""

    client_account_id: int
    client_name: str
    appointments: int
    late: int
    paid_appts: int
    unpaid_appts: int
    total_expected_cents: int
    total_paid_cents: int
    total_owed_cents: int


class OwnerPaymentsSummary(BaseModel):
    """Owner-wide payments summary result set and totals."""

    start: str
    end: str
    totals: dict
    results: List[OwnerClientAggregate]


# ---------------- Financial analytics (owner + client) ----------------

PaymentStatusFull = Literal[
    "unpaid", "partial", "paid", "refunded", "waived", "bundle", "unknown"
]


class FinancialFilters(BaseModel):
    """Filter parameters for financial reports and exports."""

    start: date
    end: date
    status: Optional[List[Literal["booked", "completed", "canceled"]]] = None
    payment_status: Optional[
        List[Literal["unpaid", "paid", "refunded", "waived", "partial", "bundle"]]
    ] = None
    client_account_id: Optional[int] = None


class AppointmentFinancialRow(BaseModel):
    """Financially enriched appointment row with computed amounts."""

    id: str
    start_utc: datetime
    end_utc: datetime
    client_account_id: Optional[int] = None
    client_label: Optional[str] = None
    lesson_person_name: Optional[str] = None
    lesson_person_email: Optional[str] = None
    is_group: Optional[bool] = None
    status: Literal["booked", "completed", "canceled"]
    duration_minutes: int
    attendance_status: Optional[Literal["unknown", "attended", "late", "no_show"]] = (
        None
    )
    price_cents: Optional[int] = None
    paid_cash_cents: int = 0
    bundle_applied_cents: int = 0
    owed_cents: int = 0
    payment_status: PaymentStatusFull = "unknown"


class FinancialSummary(BaseModel):
    """Topline financial metrics across a filtered window."""

    total_appointments: int
    total_expected_cents: int
    total_paid_cents: int
    total_cash_cents: int
    total_bundle_cents: int
    total_owed_cents: int
    total_wallet_balance_cents: int = 0
    total_no_show: int = 0


# ---------------- Administration fee (owner) ----------------

AdminFeeStatus = Literal["unpaid", "bundle", "refunded", "waived", "paid"]


class AdminFeeSettingsOut(BaseModel):
    """Owner setting for the flat administration fee amount (in cents)."""

    admin_fee_cents: int


class AdminFeeSettingsIn(BaseModel):
    """Input model for updating owner administration fee amount."""

    admin_fee_cents: int = Field(default=1500, ge=0, le=1_000_000)


class AdminFeeChargeCreate(BaseModel):
    """Create payload for a standalone administration fee charge."""

    client_account_id: int = Field(..., ge=1)
    amount_cents: Optional[int] = Field(None, ge=0, le=1_000_000)
    note: Optional[str] = None


class AdminFeeChargeUpdate(BaseModel):
    """Partial update for an administration fee charge."""

    status: Optional[AdminFeeStatus] = None
    paid_cash_cents: Optional[int] = Field(None, ge=0, le=1_000_000)
    note: Optional[str] = None
    apply_wallet: Optional[bool] = None


class AdminFeeChargeOut(BaseModel):
    """Administration fee record returned from the API with audit fields."""

    id: int
    owner_id: str
    client_account_id: int
    client_user_id: Optional[str] = None
    amount_cents: int
    status: AdminFeeStatus
    paid_cash_cents: int
    bundle_applied_cents: int
    note: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    client_label: Optional[str] = None
