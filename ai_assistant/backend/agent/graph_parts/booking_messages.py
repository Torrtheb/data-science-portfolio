from __future__ import annotations
import json
from typing import Any, Mapping
from langchain_core.messages import AIMessage, SystemMessage


def make_pending_book_failed_message(data: Mapping[str, Any]) -> AIMessage:
    """Build a deterministic AI message for a booking failure state.

    Arguments
    - 'data': Mapping containing failure context (e.g., 'reason', 'start_local',
      or other machine-readable fields).

    Behavior
    - Serializes 'data' to JSON; on serialization error, falls back to a
      minimal subset ('reason', 'start_local').
    - Prefixes content with 'PENDING_BOOK_FAILED:' for downstream parsing.

    Returns
    - 'AIMessage' with deterministic content for the post-tools stage.

    Raises
    - None
    """
    try:
        payload = json.dumps(data)
    except Exception:
        payload = json.dumps(
            {
                k: data.get(k)
                for k in ("reason", "start_local")
                if data.get(k) is not None
            }
        )
    return AIMessage(content="PENDING_BOOK_FAILED:" + payload)


def make_pending_booked_message(data: Mapping[str, Any]) -> AIMessage:
    """Build a deterministic AI message for a successful booking state.

    Arguments
    - 'data': Mapping containing success context (e.g., 'appointment_id',
      'start_local', etc.).

    Behavior
    - Serializes 'data' to JSON; on serialization error, falls back to include
      just the 'appointment_id' when available.
    - Prefixes content with 'PENDING_BOOKED:' for downstream parsing.

    Returns
    - 'AIMessage' with deterministic content for the post-tools stage.

    Raises
    - None
    """
    try:
        payload = json.dumps(data)
    except Exception:
        payload = json.dumps({"appointment_id": data.get("appointment_id")})
    return AIMessage(content="PENDING_BOOKED:" + payload)


def build_booking_instruction_for_payload(
    payload: Mapping[str, Any], duration_min: int | None
) -> str:
    """Compose deterministic instructions for single booking with resolved identity.

    Arguments
    - 'payload': Already-resolved identity and metadata fields to include
      verbatim when calling 'book_appointment' (e.g., 'person_id', 'client_name',
      'client_email', 'client_query').
    - 'duration_min': If provided, include this explicit duration; otherwise
      instruct the model to ask the user for duration before calling the tool.

    Returns
    - A concise, step-like instruction string for the LLM to follow exactly.

    Raises
    - None
    """
    identity_order = ["person_id", "client_name", "client_email", "client_query"]
    identity_lines = []
    for key in identity_order:
        val = payload.get(key)
        if val is not None:
            identity_lines.append(f"    - {key}: {json.dumps(val)}")
    for key, value in payload.items():
        if key not in identity_order and value is not None:
            identity_lines.append(f"    - {key}: {json.dumps(value)}")
    if not identity_lines:
        identity_lines.append("    - (no additional identity fields)")

    if duration_min is None:
        duration_line = (
            "  - duration_min: ask the user for the duration before calling the tool."
        )
    else:
        duration_line = f"  - duration_min: {int(duration_min)}"

    return (
        "The client has already been resolved. Do not call 'find_client' again.\n"
        "Call 'book_appointment' exactly once with:\n"
        "  - start_local: parse from the user's request in the owner's local timezone (format 'YYYY-MM-DDTHH:MM')\n"
        f"{duration_line}\n"
        "  - Include these identity fields exactly as written:\n"
        + "\n".join(identity_lines)
        + "\n"
        "If the requested time is unavailable, rely on the tool response."
    )


def build_recurring_instruction_for_payload(
    payload: Mapping[str, Any], duration_min: int | None
) -> str:
    """Compose deterministic instructions for recurring bookings with identity.

    Arguments
    - 'payload': Identity/metadata fields to include when calling
      'book_recurring_appointments'.
    - 'duration_min': If provided, include explicitly; otherwise instruct the
      model to ask the user first.

    Returns
    - A concise instruction string that specifies cadence parameters and the
      identity fields to pass to the tool.

    Raises
    - None
    """
    identity_order = ["person_id", "client_name", "client_email", "client_query"]
    identity_lines = []
    for key in identity_order:
        val = payload.get(key)
        if val is not None:
            identity_lines.append(f"    - {key}: {json.dumps(val)}")
    for key, value in payload.items():
        if key not in identity_order and value is not None:
            identity_lines.append(f"    - {key}: {json.dumps(value)}")
    if not identity_lines:
        identity_lines.append("    - (no additional identity fields)")

    if duration_min is None:
        duration_line = (
            "  - duration_min: ask the user for the duration before calling the tool."
        )
    else:
        duration_line = f"  - duration_min: {int(duration_min)}"

    return (
        "The client has already been resolved. Do not call 'find_client' again.\n"
        "This is a recurring booking request. Call 'book_recurring_appointments' exactly once with:\n"
        "  - start_local: parse the first occurrence in the owner's local timezone ('YYYY-MM-DDTHH:MM')\n"
        f"{duration_line}\n"
        "  - repeat_every_weeks: parse from the cadence (default 1 if simply weekly)\n"
        "  - Provide either 'occurrences' (count) or 'until_date' ('YYYY-MM-DD') based on the user's request.\n"
        "  - Include these identity fields exactly as written:\n"
        + "\n".join(identity_lines)
        + "\n"
        "If conflicts occur, rely on the tool response before proceeding."
    )


def make_booking_llm_instruction(data: Mapping[str, Any], status: str) -> SystemMessage:
    """Produce a short instruction prompt for confirmations or failure notices.

    Arguments
    - 'data': Mapping with booking context (e.g., client name, start time, reason).
    - 'status': Either 'booked' or any other string indicating failure.

    Behavior
    - When 'status == 'booked', asks for a friendly 1–2 sentence confirmation.
    - Otherwise, asks for a brief empathetic explanation referencing the reason
      and suggesting alternatives.

    Returns
    - 'SystemMessage' containing the instruction and inlined JSON 'DATA' for
      the model to reference.

    Raises
    - None
    """
    safe = {k: v for k, v in data.items() if v is not None}
    safe["status"] = status
    SafeJSON = json.dumps(safe)
    if status == "booked":
        prompt = (
            "Craft a short, upbeat confirmation for the owner letting them know the appointment is booked. "
            "Mention the client name and start time, and invite them to ask for anything else. "
            "Keep it under 2 sentences.\n"
            f"DATA: {SafeJSON}"
        )
    else:
        prompt = (
            "Explain briefly that the appointment was not booked. "
            "Use the reason provided and suggest checking other slots if available. Keep it warm and concise.\n"
            f"DATA: {SafeJSON}"
        )
    return SystemMessage(content=prompt)
