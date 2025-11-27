from __future__ import annotations
from typing import Any, Final, Dict, List

# -----------------------------
# Core scheduling / legacy tools
# -----------------------------
from agent.tools import (
    find_slots_tool,
    cancel_appointment_tool,
    get_next_appointment_for_client_tool,
    list_appointments_tool,
    get_appointment_details_tool,
    update_appointment_tool,
    reschedule_appointment_tool,
    update_appointment_details_tool,
    list_post_appointment_actions_tool,
    help_appointment_updates_tool,
)
from agent.tools_wallets import (
    create_bundle_tool,
    attach_wallet_tool,
    apply_wallet_tool,
    restore_wallet_tool,
    top_up_wallet_tool,
    list_wallets_tool,
    create_admin_fee_charge_tool,
    adjust_wallet_tool,
)

# -----------------------------
# New booking tool (service-backed)
# -----------------------------
from agent.tools_booking import (
    book_appointment_tool,
    book_recurring_appointments_tool,
    debug_slot_conflicts_tool,
)

# -----------------------------
# Calendar / availability tools
# -----------------------------
from agent.tools_calendar import (
    calendar_snapshot_tool,
    add_special_opening_tool,
    create_recurring_openings_tool,
    list_openings_tool,
    update_opening_tool,
    delete_opening_tool,
    add_availability_tool,
    list_weekly_rules_tool,
    create_weekly_rule_tool,
    update_weekly_rule_tool,
    delete_weekly_rule_tool,
    truncate_availability_after_tool,
    next_time_off_tool,
)

from agent.calendar.timeoff import (
    add_time_off_tool,
    update_time_off_tool,
    delete_time_off_tool,
    list_time_off_tool,
)

# -----------------------------
# External: Public Holidays
# -----------------------------
from agent.tools_holidays import (
    get_public_holidays_tool,
    is_public_holiday_tool,
)

from agent.tools_fun import fun_cute_image_tool

# -----------------------------
# Client management tools
# -----------------------------
from agent.tools_clients import (
    list_clients_tool,
    find_client_tool,
    update_client_tool,
    list_people_tool,
    update_person_tool,
    delete_person_tool,
)

# -----------------------------
# Ops / comms / finance tools
# -----------------------------
from agent.tool_ops import (
    send_email_tool,
    list_service_options_tool,
    financial_summary_tool,
    customer_payments_tool,
    customer_balances_tool,
    total_owed_tool,
    owner_financial_dashboard_tool,
    explain_owner_dashboard_tool,
    create_email_draft_tool,
    send_approved_email_tool,
)

ALL_TOOLS: Final[list[Any]] = [
    # 1) Client resolution (so the LLM learns to pick identity first)
    list_clients_tool,
    find_client_tool,
    update_client_tool,
    list_people_tool,
    update_person_tool,
    delete_person_tool,
    # 2) Availability → Booking (prefer the identity-safe booking tool)
    find_slots_tool,
    book_appointment_tool,
    book_recurring_appointments_tool,
    debug_slot_conflicts_tool,
    # 3) Appointment mgmt & lookups
    update_appointment_details_tool,
    reschedule_appointment_tool,
    get_appointment_details_tool,
    update_appointment_tool,
    cancel_appointment_tool,
    list_appointments_tool,
    get_next_appointment_for_client_tool,
    # 4) Calendar/availability maintenance
    calendar_snapshot_tool,
    add_time_off_tool,
    update_time_off_tool,
    delete_time_off_tool,
    list_time_off_tool,
    next_time_off_tool,
    add_special_opening_tool,
    create_recurring_openings_tool,
    list_openings_tool,
    update_opening_tool,
    delete_opening_tool,
    add_availability_tool,
    list_weekly_rules_tool,
    create_weekly_rule_tool,
    update_weekly_rule_tool,
    delete_weekly_rule_tool,
    truncate_availability_after_tool,
    get_public_holidays_tool,
    is_public_holiday_tool,
    # 5) Comms & finance
    list_service_options_tool,
    financial_summary_tool,
    customer_payments_tool,
    customer_balances_tool,
    total_owed_tool,
    owner_financial_dashboard_tool,
    explain_owner_dashboard_tool,
    create_email_draft_tool,
    send_approved_email_tool,
    create_bundle_tool,
    attach_wallet_tool,
    apply_wallet_tool,
    restore_wallet_tool,
    top_up_wallet_tool,
    list_wallets_tool,
    create_admin_fee_charge_tool,
    adjust_wallet_tool,
    send_email_tool,
    help_appointment_updates_tool,
    list_post_appointment_actions_tool,
    fun_cute_image_tool,
]


# -----------------------------
# Lightweight intent metadata
# -----------------------------
TOOL_INTENTS: Final[Dict[str, Dict[str, Any]]] = {
    # Booking and availability
    "book_appointment": {
        "keywords": ["book", "schedule", "lesson", "appointment", "reschedule"],
        "summary": "Book a single appointment/lesson at a specific time.",
    },
    "book_recurring_appointments": {
        "keywords": ["weekly", "every week", "recurring", "repeat"],
        "summary": "Book a series of recurring appointments on a weekly pattern.",
    },
    "find_slots": {
        "keywords": ["slot", "availability", "free time", "open time"],
        "summary": "Find open time slots of a given length on a day.",
    },
    "calendar_snapshot": {
        "keywords": ["schedule", "calendar", "what's my day", "what's my week"],
        "summary": "Overview of openings, time off, and appointments for today/week/month.",
    },
    "add_special_opening": {
        "keywords": ["opening", "availability", "open up", "add opening"],
        "summary": "Create a one-off opening window for bookings on a specific day.",
    },
    "add_time_off": {
        "keywords": ["time off", "vacation", "day off", "ooo", "out of office"],
        "summary": "Block time off so clients cannot book during that window.",
    },
    "cancel_appointment": {
        "keywords": ["cancel appointment", "cancel lesson", "cancel my lesson"],
        "summary": "Cancel an existing appointment/lesson for the owner.",
    },
    # Appointments / clients
    "list_appointments": {
        "keywords": ["appointments", "lessons", "today", "tomorrow", "on"],
        "summary": "List appointments/lessons on a given day.",
    },
    "get_next_appointment_for_client": {
        "keywords": ["next lesson", "next appointment", "when is", "upcoming"],
        "summary": "Find the next upcoming appointment for a specific client.",
    },
    "list_clients": {
        "keywords": ["clients", "who are my clients", "list clients"],
        "summary": "List clients and basic contact info.",
    },
    "find_client": {
        "keywords": ["client", "email", "phone", "find", "look up"],
        "summary": "Resolve a single client by name or email before other actions.",
    },
    # Email / comms
    "create_email_draft": {
        "keywords": ["email", "message", "notify", "write to", "draft"],
        "summary": "Draft an email based on the request without sending it yet.",
    },
    "send_approved_email": {
        "keywords": ["send email", "send it", "approve email"],
        "summary": "Send a previously approved email draft.",
    },
    # Wallets / finance
    "list_wallets": {
        "keywords": ["wallet", "store credit", "balance", "credit"],
        "summary": "List wallets and balances for a client.",
    },
    "customer_balances": {
        "keywords": ["owe", "owed", "outstanding", "who owes", "balances"],
        "summary": "Show how much each customer owes over a period.",
    },
    "total_owed": {
        "keywords": ["total owed", "owe in total", "outstanding balance"],
        "summary": "Return the total amount owed over a period.",
    },
    "owner_financial_dashboard": {
        "keywords": ["revenue", "owed", "dashboard", "how much did I make"],
        "summary": "Summarize revenue and amounts owed over a period.",
    },
    "explain_owner_dashboard": {
        "keywords": ["explain dashboard", "summarize finances", "explain revenue"],
        "summary": "Turn a dashboard dict into a concise summary for the owner.",
    },
    "truncate_after": {
        "keywords": [
            "not available after",
            "no availability after",
            "stop at",
            "cut off after",
        ],
        "summary": "Cut off availability after a specific local time today.",
    },
    "next_time_off": {
        "keywords": [
            "next time off",
            "upcoming time off",
            "time off",
            "pto",
            "vacation",
            "day off",
            "out of office",
            "ooo",
        ],
        "summary": "Find the next scheduled time-off block.",
    },
    # Holidays / fun
    "get_public_holidays": {
        "keywords": ["holiday", "holidays", "statutory", "public holiday"],
        "summary": "List public holidays for a country and year.",
    },
    "is_public_holiday": {
        "keywords": ["holiday", "public holiday", "statutory"],
        "summary": "Check if a specific date is a public holiday.",
    },
    "fun_cute_image": {
        "keywords": ["cute", "fun", "picture", "image"],
        "summary": "Generate a fun, cute image for light-hearted requests.",
    },
}


def get_tool_intents() -> Dict[str, Dict[str, Any]]:
    """Return a mapping of tool_name → intent metadata."""
    return TOOL_INTENTS


if False:
    import logging

    names = [
        getattr(t, "name", getattr(t, "name", None)) or getattr(t, "__name__", str(t))
        for t in ALL_TOOLS
    ]
    logging.getLogger(__name__).info("Registered tools (order): %s", names)
