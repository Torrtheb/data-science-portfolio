from __future__ import annotations

import json
from typing import Any, Mapping, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage


def _tc(
    name: str, args: Mapping[str, Any] | None, id: str | None = None
) -> dict[str, Any]:
    """Build a uniform 'tool_call' dictionary for messages and tests.

    Arguments
    - 'name': The tool/function name to invoke.
    - 'args': Arguments to pass to the tool. If 'None', an empty dict is used.
    - 'id': Optional call identifier to keep tool calls addressable.

    Returns
    - A dictionary shaped as '{"name": <name>, "args": <dict>, "id": <optional>}'.

    Raises
    - None
    """
    return {"name": name, "args": dict(args or {}), **({"id": id} if id else {})}


def _tc_name(tc: Mapping[str, Any]) -> str | None:
    """Extract the tool/function name from a 'tool_call' mapping.

    Supports both shapes:
    - OpenAI function calls: '{"function": {"name": "...", "arguments": "..."}}'
    - Simple shape: '{"name": "...", "args": {...}}'

    Arguments
    - 'tc': The tool call payload as a mapping.

    Returns
    - The name string if present, otherwise 'None'.

    Raises
    - None
    """
    func = tc.get("function")
    if isinstance(func, Mapping):
        nm = func.get("name")
        if isinstance(nm, str):
            return nm
    nm2 = tc.get("name")
    return nm2 if isinstance(nm2, str) else None


def _tc_args(tc: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the arguments dictionary from a 'tool_call' mapping.

    Supports the following sources in precedence order:
    - '{"args": {...}}' (already a dict)
    - '{"function": {"arguments": "<json string>"}}'
    - '{"arguments": "<json string>"}'

    Arguments
    - 'tc': The tool call payload as a mapping.

    Returns
    - A dictionary of arguments. If no arguments are available or decoding fails,
      an empty dictionary is returned.

    Raises
    - None
    """
    if "args" in tc and isinstance(tc["args"], dict):
        return tc["args"] or {}

    raw: Any = (
        (tc.get("function") or {}).get("arguments")
        or tc.get("arguments")
        or tc.get("args")
        or {}
    )

    if isinstance(raw, dict):
        return raw or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except Exception:
            return {}
    return {}


def _last_tool_call_and_args(
    messages: Sequence[BaseMessage],
) -> Tuple[str | None, dict[str, Any] | None]:
    """Find the most recent assistant tool call and its arguments.

    Supports both OpenAI function-call shape and the simple '{"name","args"}' shape.

    Arguments
    - 'messages': The conversation history as a sequence of 'BaseMessage'.

    Returns
    - A tuple of '(tool_name, args_dict)' if found; otherwise '(None, None)'.

    Raises
    - None
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            try:
                tc = m.tool_calls[-1]
                name = _tc_name(tc)
                args = _tc_args(tc)
                return name, args
            except Exception:
                return None, None
    return None, None
