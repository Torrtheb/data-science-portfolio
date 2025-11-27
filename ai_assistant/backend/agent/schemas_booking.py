from __future__ import annotations
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional


class BookAppointmentIn(BaseModel):
    """Input payload to book a single appointment.

    Fields:
        start_local: Owner‑local ISO datetime ("YYYY‑MM‑DDTHH:MM").
        duration_minutes: Appointment duration in minutes (15–240).
        person_id: Existing Person.id; preferred when known.
        client_email: Email to resolve/create a person when needed.
        client_name: Exact client name; used when email/person_id not provided.
        note: Optional owner‑private note.
        price_override_cents: Optional per‑appointment price override in cents.
    """

    start_local: str = Field(
        ..., description="Owner-local ISO datetime 'YYYY-MM-DDTHH:MM'"
    )
    duration_minutes: int = Field(..., ge=15, le=240)

    person_id: Optional[int] = None
    client_email: Optional[EmailStr] = None
    client_name: Optional[str] = None

    note: Optional[str] = None
    price_override_cents: Optional[int] = None


class BookAppointmentOut(BaseModel):
    """Response payload returned after successful booking.

    Fields:
        appointment_id: Identifier of the created appointment.
        start_utc: Start time in UTC.
        end_utc: End time in UTC.
        client_name: Denormalized client name attached to appointment.
        client_email: Denormalized client email, if known.
        person_id: Linked person identifier, if resolved.
        status: Appointment status (e.g., "booked").
    """

    appointment_id: str
    start_utc: datetime
    end_utc: datetime
    client_name: str
    client_email: Optional[str] = None
    person_id: Optional[int] = None
    status: str
