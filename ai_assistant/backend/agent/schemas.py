from __future__ import annotations
from typing import Optional, Literal, List, Dict, Any, Union
from uuid import UUID
from datetime import datetime, date

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
    ConfigDict,
    model_validator,
)


class ChatInput(BaseModel):
    """Input payload for initiating or continuing a chat session."""

    owner_user_id: str
    session_id: str
    message: str


class ChatEvent(BaseModel):
    """Server-sent event emitted during chat streaming."""

    type: Literal["assistant_msg", "tool_call", "tool_result", "error", "final"]
    data: Dict


class ToolRememberIn(BaseModel):
    """Persist a small key/value memory associated with an entity."""

    owner_user_id: str
    subject_type: Literal["owner", "client", "task", "preference"]
    subject_id: Optional[str] = None
    key: str
    value_json: Dict


class ToolRememberOut(BaseModel):
    """Acknowledgement for a stored memory."""

    memory_id: int


class ToolFindSlotsIn(BaseModel):
    """Parameters for free-time slot search in the owner's local timezone."""

    owner_user_id: Optional[str] = None
    day: str = Field(
        ...,
        description="Owner-local day: 'YYYY-MM-DD', 'today', 'tomorrow', 'next friday', etc.",
    )
    duration_minutes: int = Field(
        ..., ge=15, le=240, description="Desired slot length in minutes"
    )


class ToolFindSlotsOut(BaseModel):
    """Resulting available slots for the requested day and duration."""

    slots: List[Dict]


class AppointmentIdentity(BaseModel):
    """Identity for the person an appointment is for.

    Provide at least one of:
    - 'person_id' (preferred)
    - 'client_email'
    - 'client_name' (case-insensitive exact match within the owner account)
    """

    person_id: Optional[int] = Field(None, description="Existing Person.id if known")
    client_email: Optional[EmailStr] = Field(
        None, description="Email to resolve or create a Person"
    )
    client_name: Optional[str] = Field(
        None, description="Exact full name to match/create if email is unknown"
    )

    def require_identity_any(self) -> None:
        """Validate that at least one identity field is present.

        Raises:
            ValueError: If none of 'person_id', 'client_email', or
                'client_name' is provided.
        """
        if not (self.person_id or self.client_email or self.client_name):
            raise ValueError(
                "Appointment identity required: provide person_id or client_email or client_name"
            )


class ToolCreateApptIn(AppointmentIdentity):
    """Create a one-off appointment in the owner's local timezone."""

    owner_user_id: Optional[str] = None
    start_local: str = Field(..., description="Owner-local ISO 'YYYY-MM-DDTHH:MM'")
    duration_minutes: int = Field(..., ge=15, le=240)
    note: Optional[str] = None
    service_option_id: Optional[int] = Field(
        None, description="Optional ServiceOption.id for pricing"
    )

    @field_validator("start_local")
    @classmethod
    def _iso_like(cls, v: str) -> str:
        if "T" not in v:
            raise ValueError("start_local must look like 'YYYY-MM-DDTHH:MM'")
        return v


class ToolCreateApptOut(BaseModel):
    """Confirmation payload returned after creating an appointment."""

    appointment_id: str
    start_utc: datetime
    end_utc: datetime
    status: str = "booked"


class ToolBookAppointmentIn(AppointmentIdentity):
    """Book an appointment (service-backed) with identity and pricing context."""

    model_config = ConfigDict(populate_by_name=True)  # allow alias population

    owner_user_id: Optional[str] = None
    start_local: str = Field(..., description="Owner-local ISO 'YYYY-MM-DDTHH:MM'")
    duration_min: int = Field(..., ge=5, le=240)
    client_query: Optional[str] = None
    price_cents: Optional[int] = None
    notes: Optional[str] = Field(default=None, alias="private_note")
    create_person_if_missing: bool = False

    @field_validator("start_local")
    @classmethod
    def _iso_like(cls, v: str) -> str:
        if "T" not in v:
            raise ValueError("start_local must look like 'YYYY-MM-DDTHH:MM'")
        return v


class ToolBookAppointmentOut(BaseModel):
    """Result of a booking attempt, including assigned identity if available."""

    appointment_id: str
    start_utc: datetime
    end_utc: datetime
    status: str
    person_id: Optional[int] = None
    client_name: Optional[str] = None
    client_email: Optional[EmailStr] = None


class ToolBookRecurringAppointmentsIn(BaseModel):
    """Create a series of recurring appointments starting from an anchor time."""

    start_local: str = Field(..., description="Owner-local ISO 'YYYY-MM-DDTHH:MM'")
    duration_min: int = Field(..., ge=5, le=240)
    repeat_every_weeks: int = Field(default=1, ge=1, le=52)
    occurrences: Optional[int] = Field(None, ge=1, le=104)
    until_date: Optional[str] = Field(
        None, description="Inclusive owner-local date 'YYYY-MM-DD'"
    )
    client_email: EmailStr
    client_name: Optional[str] = None
    confirm_if_conflicts: bool = False
    message: Optional[str] = None

    @field_validator("start_local")
    @classmethod
    def _iso_like(cls, v: str) -> str:
        if "T" not in v:
            raise ValueError("start_local must look like 'YYYY-MM-DDTHH:MM'")
        return v

    @model_validator(mode="after")
    def _validate_recurrence(self):
        if self.occurrences is None and self.until_date is None:
            raise ValueError("Provide occurrences or until_date for recurring booking")
        if self.until_date is not None:
            try:
                date.fromisoformat(self.until_date)
            except ValueError as e:
                raise ValueError(f"until_date must be YYYY-MM-DD: {e}")
        return self


class ToolBookRecurringAppointmentsOut(BaseModel):
    """Summary of created recurring appointments."""

    count: int
    appointments: List[Dict[str, Any]]


class ToolUpdateAppointmentIn(BaseModel):
    """Update appointment metadata such as attendance, payment, and notes."""

    appointment_id: str
    private_note: Optional[str] = Field(
        None, description="Owner-only notes before the visit"
    )
    attendance: Optional[Literal["unknown", "attended", "late", "no_show"]] = None
    late_minutes: Optional[int] = Field(None, ge=0)
    payment_status: Optional[Literal["unpaid", "paid", "refunded", "waived"]] = None
    amount_paid_cents: Optional[int] = Field(None, ge=0)
    price_override_cents: Optional[int] = Field(
        None, ge=0, description="Change price for this appointment only"
    )
    bundle_id: Optional[int] = Field(
        None, description="Attach bundle/pack to this visit"
    )
    apply_wallet_now: Optional[bool] = Field(
        False, description="If true, apply wallet funds up to owed"
    )
    restore_wallet_now: Optional[bool] = Field(
        False, description="If true, restore any wallet funds previously applied"
    )


class ToolUpdateAppointmentOut(BaseModel):
    """Confirmation of updated fields for an appointment."""

    ok: bool
    appointment_id: str
    updated: dict


class ToolUpdateApptDetailsIn(AppointmentIdentity):
    """Change or fix the identity attached to an existing appointment."""

    appointment_id: str
    note: Optional[str] = None
    price_override_cents: Optional[int] = Field(None, ge=0)


class ToolUpdateApptDetailsOut(BaseModel):
    """Result of identity update or attachment on an appointment."""

    ok: bool
    appointment_id: str
    person_id: Optional[int] = None
    client_name: Optional[str] = None
    client_email: Optional[EmailStr] = None


class ToolSendEmailIn(BaseModel):
    """Send a direct email (non-draft). Prefer drafts in most flows."""

    owner_user_id: Optional[str] = None
    to: EmailStr
    subject: str
    html: Optional[str] = None
    text: Optional[str] = None


class ToolSendEmailOut(BaseModel):
    """Acknowledgement for queued email send."""

    queued_id: str


class ToolListServiceOptionsOut(BaseModel):
    """Collection of service options available to the owner/account."""

    options: List[Dict]


class ToolFinancialSummaryIn(BaseModel):
    """Parameters for generating a financial summary report."""

    owner_user_id: Optional[str] = None
    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    client_account_id: Optional[int] = None
    status: Optional[List[str]] = None
    payment_status: Optional[List[str]] = None


class ToolFinancialSummaryOut(BaseModel):
    """Aggregated totals and detailed results for a financial report."""

    totals: Dict
    results: List[Dict]


class ToolDateRangeIn(BaseModel):
    """Optional date range window used by analytics endpoints."""

    owner_user_id: Optional[str] = None
    start: Optional[str] = Field(None, description="YYYY-MM-DD (inclusive)")
    end: Optional[str] = Field(None, description="YYYY-MM-DD (inclusive)")


class CustomerPaymentLine(BaseModel):
    """Line item representing payment details for a single visit."""

    date: str
    appointment_id: str
    paid_cash_cents: int = 0
    bundle_applied_cents: int = 0


class CustomerPaymentsOut(BaseModel):
    """Aggregated payments grouped by customer."""

    customers: List[Dict]


class CustomerBalancesOut(BaseModel):
    """Outstanding balances grouped by customer."""

    customers: List[Dict]


class TotalOwedOut(BaseModel):
    """Total amount owed within an optional window."""

    start: Optional[str] = None
    end: Optional[str] = None
    total_owed_cents: int


class ToolOwnerDashboardIn(BaseModel):
    """Inputs for the owner financial dashboard summary."""

    owner_user_id: Optional[str] = None
    start: Optional[str] = Field(None, description="YYYY-MM-DD (inclusive)")
    end: Optional[str] = Field(None, description="YYYY-MM-DD (inclusive)")
    top_n: int = Field(default=5, ge=1, le=50)


class ToolOwnerDashboardOut(BaseModel):
    """Owner dashboard dataset including totals and top lists."""

    start: Optional[str] = None
    end: Optional[str] = None
    totals: Dict
    revenue_paid_cents: int
    total_owed_cents: int
    top_debtors: List[Dict]
    top_payers: List[Dict]


class ToolCreateBundleIn(BaseModel):
    """Create a wallet (store credit) or a fixed-credit bundle for a client."""

    owner_user_id: Optional[str] = None
    client_user_id: str = Field(..., description="auth.User.id for the client")
    name: str = Field(default="Bundle", min_length=1, max_length=120)
    total_credits: int = Field(
        ge=0, le=200, description="0 for wallet (store credit); >0 for credit pack"
    )
    price_cents: int = Field(ge=0, description="Deposit for wallet or bundle price")
    currency: str = Field(default="USD", max_length=10)
    expires_at: Optional[datetime] = None


class ToolCreateBundleOut(BaseModel):
    """Details of the created bundle or wallet."""

    id: int
    client_id: str
    name: str
    total_credits: int
    remaining_credits: int
    price_cents: int
    currency: str
    status: str
    expires_at: Optional[datetime] = None
    created_at: datetime
    remaining_balance_cents: Optional[int] = None


class ToolAdminFeeChargeIn(BaseModel):
    """Issue an administrative fee charge to a client account/user."""

    owner_user_id: Optional[str] = None
    client_account_id: Optional[int] = Field(None, ge=1)
    client_user_id: Optional[str] = None
    client_email: Optional[EmailStr] = None
    amount_cents: Optional[int] = Field(None, ge=0, le=1_000_000)
    note: Optional[str] = None

    def require_target(self) -> None:
        """Validate that a client target is specified.

        Raises:
            ValueError: If none of 'client_account_id', 'client_user_id',
                or 'client_email' is provided.
        """
        if not (self.client_account_id or self.client_user_id or self.client_email):
            raise ValueError(
                "Provide client_account_id, client_user_id, or client_email"
            )


class ToolAdminFeeChargeOut(BaseModel):
    """Confirmation for an administrative fee charge."""

    charge_id: int
    client_account_id: int
    amount_cents: int
    status: str


class ToolAdjustWalletIn(BaseModel):
    """Adjust wallet funds outside an appointment context."""

    owner_user_id: Optional[str] = None
    bundle_id: int = Field(..., ge=1)
    amount_cents: int = Field(
        ..., description="Positive to add funds, negative to remove funds"
    )
    note: Optional[str] = None
    client_user_id: Optional[str] = None
    client_account_id: Optional[int] = None

    @field_validator("amount_cents")
    @classmethod
    def _non_zero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("amount_cents must be non-zero")
        return v


class ToolAdjustWalletOut(BaseModel):
    """Result of a wallet adjustment including new balance."""

    ok: bool
    bundle_id: int
    balance_cents: int


class ToolAddTimeOffIn(BaseModel):
    """Block off time in the owner's calendar (vacation, lunch, etc.)."""

    owner_user_id: Optional[str] = None
    start_local: str = Field(..., description="Owner-local ISO like 'YYYY-MM-DDTHH:MM'")
    end_local: str = Field(..., description="Owner-local ISO like 'YYYY-MM-DDTHH:MM'")
    note: Optional[str] = None


class ToolAddTimeOffOut(BaseModel):
    """Confirmation of the created time-off block."""

    id: str
    start_utc: datetime
    end_utc: datetime
    note: Optional[str] = None


class ToolCalendarSnapshotIn(BaseModel):
    """Inputs for generating a calendar snapshot over a scope."""

    scope: Literal["today", "week", "month", "tomorrow", "day"]
    anchor: Optional[str] = None


class ToolCalendarEvent(BaseModel):
    """Calendar event element returned by the snapshot tool."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["opening", "time_off", "appointment"]
    title: str
    start: datetime
    end: datetime
    status: Optional[str] = None

    start_local: Optional[str] = None
    end_local: Optional[str] = None
    start_local_pretty: Optional[str] = None
    end_local_pretty: Optional[str] = None
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None

    meta: Optional[Dict[str, Any]] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    client_name: Optional[str] = None
    client_email: Optional[str] = None


class ToolCalendarSnapshotOut(BaseModel):
    """Calendar snapshot in the owner's timezone with optional pretty lines."""

    tz: str
    start: datetime
    end: datetime
    events: List[ToolCalendarEvent]
    pretty_lines: Optional[List[str]] = None


class ToolFunImageIn(BaseModel):
    """Lightweight schema for a fun/cute image request."""

    source: Literal["cat", "dog", "fox", "random"] = "random"
    fresh: bool = False


class ToolFunImageOut(BaseModel):
    kind: Literal["image"] = "image"
    url: str
    alt: str
    source: Literal["cat", "dog", "fox", "random"]


# =========================
# Special Openings
# =========================
class ToolAddSpecialOpeningIn(BaseModel):
    owner_user_id: Optional[str] = None
    start_local: str = Field(..., description="Owner-local ISO like 'YYYY-MM-DDTHH:MM'")
    end_local: str = Field(..., description="Owner-local ISO like 'YYYY-MM-DDTHH:MM'")
    slot_minutes: int = Field(
        ge=5, description="Granularity of slots inside this opening"
    )
    buffer_minutes: int = Field(ge=0, default=0)
    note: Optional[str] = None


class ToolAddSpecialOpeningOut(BaseModel):
    id: str
    start_utc: datetime
    end_utc: datetime
    slot_minutes: int
    buffer_minutes: int
    note: Optional[str] = None


# =========================
# Cancel Appointment
# =========================
class ToolCancelAppointmentIn(BaseModel):
    owner_user_id: Optional[str] = None
    appointment_id: Optional[str] = None
    start_local: Optional[str] = Field(
        default=None,
        description="Owner-local ISO 'YYYY-MM-DDTHH:MM' if canceling by time",
    )
    duration_minutes: Optional[int] = Field(default=None, ge=5, le=240)
    reason: Optional[str] = Field(default=None, description="Short cancel reason")


class ToolCancelAppointmentOut(BaseModel):
    appointment_id: str
    status: str


# =========================
# Email Draft & Approval
# =========================


class RecipientIn(BaseModel):
    email: str
    name: Optional[str] = None


class EmailDraftIn(BaseModel):
    to: Optional[str] = None
    subject: str
    lines: List[str]
    to_name: Optional[str] = None
    recipients: Optional[List[RecipientIn]] = None


class EmailDraftOut(BaseModel):
    draft_id: str
    to: EmailStr
    to_name: Optional[str] = None
    subject: str
    text: str
    html: str
    status: Literal["pending"] = "pending"
    recipients: List[RecipientIn] | None = None


class EmailApprovalIn(BaseModel):
    draft_id: UUID
    approve: bool = True
    to: Optional[EmailStr] = None
    to_name: Optional[str] = None
    subject: Optional[str] = None
    text: Optional[str] = None


# =========================
# Convenience lookups
# =========================
class ToolGetNextApptIn(BaseModel):
    client_query: str = Field(..., description="name/email substring to match")
    include_canceled: bool = False


class ToolGetNextApptOut(BaseModel):
    found: bool
    appointment: Optional[dict] = None


class ToolListAppointmentsIn(BaseModel):
    day: str = Field(..., description="'today'/'tomorrow' or 'YYYY-MM-DD' in owner tz")
    include_canceled: bool = False
    client_query: Optional[str] = None


class ToolListAppointmentsOut(BaseModel):
    appointments: list[dict]


class ToolGetAppointmentDetailsIn(BaseModel):
    appointment_id: Optional[str] = None
    start_local: Optional[str] = Field(
        default=None, description="Owner-local 'YYYY-MM-DDTHH:MM'"
    )
    duration_minutes: Optional[int] = Field(default=None, ge=5, le=240)


class ToolGetAppointmentDetailsOut(BaseModel):
    appointment: Dict


# =========================
# Public Holidays
# =========================
class ToolGetPublicHolidaysIn(BaseModel):
    country_code: Optional[str] = Field(
        default=None,
        description="ISO-3166 alpha-2 code, e.g. 'US', 'GB', 'DE'. Defaults to CA if omitted.",
    )
    year: Optional[int] = Field(
        None, description="4-digit year; defaults to current year in owner tz"
    )
    region_code: Optional[str] = Field(
        default=None,
        description="Optional ISO-3166-2 region (e.g. 'CA-NB'). Defaults to CA-NB if omitted.",
    )


class ToolGetPublicHolidaysOut(BaseModel):
    holidays: List[Dict]


class ToolIsPublicHolidayIn(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD (owner-local date)")
    country_code: Optional[str] = Field(
        default=None,
        description="ISO-3166 alpha-2 code, e.g. 'US', 'GB', 'DE'. Defaults to CA if omitted.",
    )
    region_code: Optional[str] = Field(
        default=None,
        description="Optional ISO-3166-2 region (e.g. 'CA-NB') for province/state-specific holidays. Defaults to CA-NB if omitted.",
    )


class ToolIsPublicHolidayOut(BaseModel):
    is_holiday: bool
    name: Optional[str] = None


# =========================
# Reschedule Appointment
# =========================
class ToolRescheduleApptIn(BaseModel):
    owner_user_id: Optional[str] = None
    appointment_id: str = Field(..., description="Existing appointment id")
    start_local: str = Field(
        ..., description="New start time in owner-local ISO 'YYYY-MM-DDTHH:MM'"
    )
    duration_minutes: int = Field(..., ge=5, le=480)
    allow_override: bool = Field(
        default=False, description="If true, applies even if conflicts exist"
    )
    message: Optional[str] = Field(
        default=None, description="Optional message for notifications (if any)"
    )

    @field_validator("start_local")
    @classmethod
    def _iso_like_resched(cls, v: str) -> str:
        if "T" not in v:
            raise ValueError("start_local must look like 'YYYY-MM-DDTHH:MM'")
        return v


class ToolRescheduleApptOut(BaseModel):
    appointment_id: str
    start_utc: datetime
    end_utc: datetime
    status: str


# =========================
# Post-appointment actions
# =========================
class PostApptActionItem(BaseModel):
    id: str
    client_name: Optional[str] = None
    start_local: str
    end_local: str
    status: str
    needs_attendance: bool
    needs_payment: bool
    late_minutes: Optional[int] = None
    payment_status: Optional[str] = None


class ToolListPostApptActionsOut(BaseModel):
    items: List[PostApptActionItem]


# =========================
# Openings
# =========================


class ToolListOpeningsIn(BaseModel):
    day: str = Field(
        ..., description="Owner-local day like 'YYYY-MM-DD' or 'today'/'tomorrow'"
    )


class ToolListOpeningsOut(BaseModel):
    openings: List[Dict]


class ToolUpdateOpeningIn(BaseModel):
    opening_id: Union[int, str]
    start_local: Optional[str] = None
    end_local: Optional[str] = None
    slot_minutes: Optional[int] = Field(None, ge=5, le=360)
    buffer_minutes: Optional[int] = Field(None, ge=0)
    note: Optional[str] = None
    day: Optional[str] = None


class ToolDeleteOpeningIn(BaseModel):
    opening_id: Union[int, str]
    day: Optional[str] = None


class ToolDeleteOpeningOut(BaseModel):
    deleted: bool
    opening_id: str


class ToolCreateRecurringOpeningsIn(BaseModel):
    weekday: int
    start_hhmm: str
    end_hhmm: str
    weeks: int
    start_date: Optional[str] = None
    slot_minutes: int
    buffer_minutes: Optional[int] = None
    note: Optional[str] = None


class ToolCreateRecurringOpeningsOut(BaseModel):
    created: List[Dict]


class ToolTruncateAfterIn(BaseModel):
    local_hhmm: str
    day: Optional[str] = None
    scope: Optional[Literal["day", "weekly", "ask"]] = None


class ToolTruncateAfterOut(BaseModel):
    ok: bool
    updated: List[str] = []
    deleted: List[str] = []
    cutoff_local: str
    requires_cancellation: bool = False
    blocked_appointments: List[Dict[str, Any]] = []


# =========================
# Time Off
# =========================


class ToolListTimeOffIn(BaseModel):
    day: str = Field(
        ..., description="Owner-local day like 'YYYY-MM-DD' or 'today'/'tomorrow'"
    )


class ToolListTimeOffOut(BaseModel):
    timeoff: List[Dict]


class ToolNextTimeOffOut(BaseModel):
    """Earliest upcoming time-off block for the owner (if any)."""

    found: bool
    start_local: Optional[str] = None
    end_local: Optional[str] = None
    note: Optional[str] = None


class ToolDeleteTimeOffIn(BaseModel):
    mode: Literal["by_id", "by_day"] = "by_day"
    timeoff_id: Optional[str] = None
    day: Optional[str] = None


class ToolDeleteTimeOffOut(BaseModel):
    deleted_count: int
    deleted_ids: List[str]


class ToolUpdateTimeOffIn(BaseModel):
    timeoff_id: Optional[str] = Field(None, description="If omitted, resolve by 'day'")
    day: Optional[str] = Field(
        None, description="Owner-local day like 'YYYY-MM-DD' or 'today'/'tomorrow'"
    )
    start_local: Optional[str] = Field(
        None,
        description="New start (owner-local) 'YYYY-MM-DDTHH:MM' or 'HH:MM' if day provided",
    )
    end_local: Optional[str] = Field(
        None,
        description="New end (owner-local) 'YYYY-MM-DDTHH:MM' or 'HH:MM' if day provided",
    )
    note: Optional[str] = None


class ToolUpdateTimeOffOut(BaseModel):
    id: str
    start_utc: datetime
    end_utc: datetime
    note: Optional[str] = None


# =========================
# Weekly Rules
# =========================


class ToolAddAvailabilityIn(BaseModel):
    start_local: str
    end_local: str
    slot_minutes: int = Field(..., ge=5, le=360)
    buffer_minutes: int = Field(..., ge=0)
    note: Optional[str] = None
    confirm_if_conflicts: bool = False


class ToolAddAvailabilityOut(BaseModel):
    id: str
    start_utc: datetime
    end_utc: datetime
    slot_minutes: int
    buffer_minutes: int
    note: Optional[str] = None


class ToolListWeeklyRulesOut(BaseModel):
    rules: List[Dict[str, Any]]


class ToolCreateWeeklyRuleIn(BaseModel):
    weekday: int = Field(..., ge=0, le=6)
    start_hhmm: str
    end_hhmm: str
    slot_minutes: int = Field(..., ge=5, le=360)
    buffer_minutes: int = Field(0, ge=0)
    note: Optional[str] = None


class ToolWeeklyRuleOut(BaseModel):
    id: str
    day_of_week: int
    start_minute: int
    end_minute: int
    slot_minutes: int
    buffer_minutes: int
    note: Optional[str] = None
    start_hhmm: str
    end_hhmm: str


class ToolDeleteWeeklyRuleIn(BaseModel):
    rule_id: str


class ToolDeleteWeeklyRuleOut(BaseModel):
    deleted: bool
    rule_id: str


class ToolUpdateWeeklyRuleIn(BaseModel):
    rule_id: Optional[str] = None
    weekday: Optional[int] = Field(None, ge=0, le=6)
    start_local: Optional[str] = None
    end_local: Optional[str] = None
    slot_minutes: Optional[int] = Field(None, ge=5, le=360)
    buffer_minutes: Optional[int] = Field(None, ge=0)
    anchor_day: Optional[str] = None


class ToolUpdateWeeklyRuleOut(BaseModel):
    ok: bool
    rule: Optional[Dict[str, Any]] = None
    ambiguous: bool = False
    candidates: Optional[List[Dict[str, Any]]] = None
    requires_cancellation: bool = False
    blocked_appointments: List[Dict[str, Any]] = []
