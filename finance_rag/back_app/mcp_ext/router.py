from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from .client import list_tools as mcp_list_tools, call_tool as mcp_call_tool, MAX_YEAR
from .client import get_indicator_value_for_year_via_mcp_server
from .registry import resolve_server
from .countries import resolve_country
from .types import WorldBankValueArgs, WorldBankValueResult

from ..analytics.routes_analytics import ingest as _analytics_ingest

router = APIRouter(tags=["mcp"])

_INDICATOR_ALIASES = {
    "gdp per capita, ppp (constant 2021 international $)": "NY.GDP.PCAP.PP.KD",
    "gdp per capita ppp constant 2021": "NY.GDP.PCAP.PP.KD",
    "gdp per capita ppp constant": "NY.GDP.PCAP.PP.KD",
    "gdp per capita ppp": "NY.GDP.PCAP.PP.KD",
    "gdp per capita, ppp (current international $)": "NY.GDP.PCAP.PP.CD",
}

# ------------------------------- helpers -------------------------------------


def _iso_utc_now() -> str:
    """
    Get the current UTC timestamp as an ISO-8601 formatted string.

    Returns:
        str: Current time in UTC with timezone info.
    """
    return datetime.now(timezone.utc).isoformat()


def _normalize_country_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and enrich country-related arguments for MCP tool calls.

    Strategy:
      - Resolve free-text 'country' → canonical ISO-3 + normalized name.
      - Rewrite known input keys ('country_id', 'countryCode', 'country_code', 'iso3')
        to ISO-3 if possible.
      - Always set 'iso3' and 'country' in the returned dict when resolution succeeds.

    Args:
        args: Arbitrary arguments dict from client/tool call.

    Returns:
        Dict with possibly rewritten/augmented fields.
    """

    if not args:
        return {}

    out: Dict[str, Any] = dict(args)
    id_keys = ["country_id", "countryCode", "country_code"]
    any_changed = False

    raw_country = out.get("country")
    if isinstance(raw_country, str) and raw_country.strip():
        try:
            iso3, name = resolve_country(raw_country)
            out["iso3"] = iso3
            out["country"] = name
            for k in id_keys:
                if isinstance(out.get(k), str):
                    out[k] = iso3
            any_changed = True
        except Exception:
            pass

    if not any_changed:
        for k in ["iso3"] + id_keys:
            raw = out.get(k)
            if isinstance(raw, str) and raw.strip():
                try:
                    iso3, name = resolve_country(raw)
                    out[k] = iso3
                    out["iso3"] = iso3
                    out.setdefault("country", name)
                    any_changed = True
                    break
                except Exception:
                    continue

    return out


def _norm_indicator_id(raw: str) -> str:
    """
    Normalize a human-friendly World Bank indicator string into a canonical ID.

    Uses a small alias map for common GDP per capita PPP phrases.
    Falls back to the input if no alias is found.

    Args:
        raw: Indicator ID or alias string.

    Returns:
        Canonical indicator ID or unchanged input.
    """
    key = (raw or "").strip().lower()
    return _INDICATOR_ALIASES.get(key, raw)


def _prefer_constant_ppp(ind_code: str, prefer_constant: bool) -> str:
    """
    Canonicalize PPP GDP per capita indicators when requested.

    - If indicator is one of {NY.GDP.PCAP.PP.CD, NY.GDP.PCAP.PP.KD}:
        - prefer_constant=True → force constant 2021 series (KD).
        - prefer_constant=False → keep whichever form was given.
    - All other indicators are returned unchanged.

    Args:
        ind_code: Candidate indicator code.
        prefer_constant: Flag controlling normalization.

    Returns:
        Canonicalized indicator ID.
    """
    u = (ind_code or "").strip().upper()
    if u in {"NY.GDP.PCAP.PP.CD", "NY.GDP.PCAP.PP.KD"}:
        return "NY.GDP.PCAP.PP.KD" if prefer_constant else u
    return ind_code


# ------------------------------- models --------------------------------------


class ToolCallIn(BaseModel):
    """
    Input model for invoking an MCP tool via the REST API.

    Fields:
        server_key (str): MCP server key from registry, e.g. "world_bank".
        tool (str): Tool name as published by the server.
        arguments (dict|None): Tool-specific arguments.
        session_id (str|None): Optional analytics session identifier.
        turn_id (int|None): Optional analytics turn identifier.
        timeout_ms (int): Timeout for the tool call (100–120000 ms).
    """

    server_key: str = Field(..., description="Key from registry (e.g., 'world_bank')")
    tool: str = Field(..., min_length=1, description="Tool name on server")
    arguments: Dict[str, Any] | None = Field(default=None, description="Tool arguments")
    session_id: Optional[str] = Field(default=None, description="Analytics session id")
    turn_id: Optional[int] = Field(default=None, description="Analytics turn id")
    timeout_ms: Optional[int] = Field(
        default=10_000, ge=100, le=120_000, description="Tool timeout"
    )

    @field_validator("tool")
    @classmethod
    def _safe_tool_name(cls, v: str) -> str:
        import re

        if not re.match(r"^[A-Za-z0-9_.:-]+$", v or ""):
            raise ValueError("invalid tool name")
        return v


# ------------------------------- endpoints -----------------------------------


@router.get("/tools")
async def get_tools(server_key: str):
    """
    List tools exposed by a named MCP server.

    Args:
        server_key: MCP server key (e.g. "world_bank").

    Returns:
        JSON with key "tools" mapping to a list of tool descriptors.

    Raises:
        HTTPException 400: Invalid server key.
        HTTPException 502: Upstream MCP error.
    """
    try:
        url = resolve_server(server_key)
        tools = await mcp_list_tools(url)
        return {"tools": tools}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MCP upstream error: {e}")


@router.post("/call")
async def call_tool(body: ToolCallIn):
    """
    Invoke a tool on a named MCP server.

    - Normalizes country arguments for consistency.
    - Wraps the call with analytics ingestion (best-effort).
    - Returns result payload plus a sources entry.

    Args:
        body: ToolCallIn request body.

    Returns:
        JSON with keys:
          - result: raw tool output (dict)
          - sources: list containing a single MCP tool source descriptor

    Raises:
        HTTPException 400: Invalid server key.
        HTTPException 502: MCP call timed out/failed.
        HTTPException 500: Unexpected error.
    """
    try:
        url = resolve_server(body.server_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    norm_args = _normalize_country_args(body.arguments or {})
    try:
        result = await mcp_call_tool(
            url, body.tool, norm_args, timeout_ms=body.timeout_ms or 10_000
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP call failed: {e}")

    try:
        if body.session_id:
            await _analytics_ingest(
                {
                    "type": "tool",
                    "session_id": body.session_id,
                    "turn_id": body.turn_id,
                    "tool_name": f"mcp:{body.server_key}:{body.tool}",
                    "args": norm_args,
                    "ok": True,
                    "latency_ms": 0,
                    "created_at": _iso_utc_now(),
                }
            )
    except Exception:
        pass

    return {
        "result": result,
        "sources": [
            {
                "kind": "tool",
                "type": "tool",
                "id": f"tool:mcp:{body.server_key}:{body.tool}",
                "title": f"MCP:{body.server_key}/{body.tool}",
                "display": f"MCP:{body.server_key}/{body.tool}",
            }
        ],
    }


@router.post("/worldbank/value", response_model=WorldBankValueResult)
async def worldbank_value(body: WorldBankValueArgs):
    """
    Fetch a single-year World Bank indicator value.

    Behavior:
      - Resolves free-text or coded 'country_id' → ISO-3 + canonical name.
      - Normalizes indicator IDs and enforces PPP constant series by default.
      - Clamps year into [1960, MAX_YEAR].
      - Calls the MCP server to fetch the exact value.

    Args:
        body: WorldBankValueArgs containing country, indicator, year, etc.

    Returns:
        WorldBankValueResult with enriched metadata.

    Raises:
        HTTPException 400: Invalid input or country not recognized.
        HTTPException 502: MCP error.
        HTTPException 500: Unexpected runtime error.
    """
    try:
        server_url = resolve_server("world_bank")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        iso3, canonical = resolve_country(body.country_id)
    except Exception:
        raise HTTPException(status_code=400, detail="country not recognized")

    year = max(1960, min(int(body.year), MAX_YEAR))
    prefer_constant = (
        True if body.prefer_constant is None else bool(body.prefer_constant)
    )
    indicator_id = _prefer_constant_ppp(
        _norm_indicator_id(body.indicator_id), prefer_constant
    )
    try:
        out = await get_indicator_value_for_year_via_mcp_server(
            server_url=server_url,
            country_id=iso3,
            indicator_id=indicator_id,
            year=year,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    out["country"] = canonical
    out["country_iso3"] = iso3
    out["year"] = year
    out.setdefault("ok", True)

    return out



@router.get("/tools/list")
async def tools_list(server_key: str = "world_bank"):
    """
    Alias endpoint for '/tools'.

    Args:
        server_key: MCP server key, defaults to "world_bank".

    Returns:
        Output from 'get_tools'.
    """
    return await get_tools(server_key)

@router.post("/tools/call")
async def tools_call(payload: dict):
    """
    Alias endpoint for '/call'.

    Accepts flexible payload formats:
      - { server_key, tool, arguments, timeout_ms? }
      - { server|server_key, name|tool, arguments|args, timeout_ms? }

    Returns:
        Same result shape as '/call'.

    Raises:
        HTTPException 400: Missing or invalid tool name / payload.
    """
    from pydantic import ValidationError

    server_key = payload.get("server_key") or payload.get("server")
    tool = payload.get("tool") or payload.get("name")
    arguments = payload.get("arguments") or payload.get("args")
    timeout_ms = payload.get("timeout_ms", 10_000)

    if not tool:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Missing 'tool'/'name'.")

    try:
        body = ToolCallIn(
            server_key=server_key or "world_bank",
            tool=tool,
            arguments=arguments,
            timeout_ms=timeout_ms,
        )
    except ValidationError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    return await call_tool(body)
