from typing import Final

ACTIVE_APPT_STATUSES: Final[frozenset[str]] = frozenset({"booked", "completed"})
CANCELLED_APPT_STATUSES: Final[frozenset[str]] = frozenset({"cancelled", "canceled"})
ALL_APPT_STATUSES: Final[frozenset[str]] = (
    ACTIVE_APPT_STATUSES | CANCELLED_APPT_STATUSES
)
BUSY_APPT_STATUSES: Final[frozenset[str]] = ACTIVE_APPT_STATUSES

__all__ = [
    "ACTIVE_APPT_STATUSES",
    "CANCELLED_APPT_STATUSES",
    "ALL_APPT_STATUSES",
    "BUSY_APPT_STATUSES",
]
