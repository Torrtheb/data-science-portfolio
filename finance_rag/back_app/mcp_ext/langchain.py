from __future__ import annotations
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from .client import (
    list_tools,
    call_tool,
    get_indicator_value_for_year_via_mcp_server,
    MAX_YEAR,
)
from .registry import resolve_server
from .router import _norm_indicator_id, _prefer_constant_ppp
from .countries import resolve_country
from ..analytics.tokenlog import track_tool

# ----------------------------- Schema helpers --------------------------------
_JSON_TYPE_TO_PYDANTIC: Dict[str, Any] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": Dict[str, Any],
    "array": List[Any],
}


def _pydantic_type_for_property(prop_schema: Dict[str, Any]) -> Any:
    """
    Map a JSON-Schema property definition to an equivalent Python type.

    Supported mappings:
      - "string"   → str
      - "number"   → float
      - "integer"  → int
      - "boolean"  → bool
      - "object"   → Dict[str, Any]
      - "array"    → List[Any] (or List[inner_type] if 'items' is known)

    Notes:
      - If "enum" is present, the base type is preserved (Literal not used).
      - Falls back to Any if schema type is missing or unrecognized.

    Args:
        prop_schema: JSON-Schema-like dict describing a property.

    Returns:
        Corresponding Python type usable in a Pydantic model.
    """
    if not isinstance(prop_schema, dict):
        return Any

    t = prop_schema.get("type")
    if t is None:
        return Any

    if t == "array":
        items = prop_schema.get("items") or {}
        inner = _pydantic_type_for_property(items) if isinstance(items, dict) else Any
        return List[inner]

    return _JSON_TYPE_TO_PYDANTIC.get(t, Any)


def _build_args_model_from_schema(
    server_key: str, tool_name: str, schema: Dict[str, Any]
) -> type[BaseModel]:
    """
    Build a dynamic Pydantic model for MCP tool arguments from a JSON-Schema.

    Example schema:
        {
          "required": ["foo"],
          "properties": {
             "foo": {"type": "string", "description": "..."},
             "bar": {"type": "integer", "description": "...", "default": 10}
          }
        }

    Behavior:
      - Required fields → Field(...).
      - Optional fields → Field(default=...).
      - If no properties, falls back to a single dict field 'payload'.

    Args:
        server_key: Identifier for the MCP server (used in model name).
        tool_name: MCP tool name (used in model name).
        schema: JSON-Schema-like dict.

    Returns:
        A Pydantic BaseModel subclass for validating tool arguments.
    """

    required = set(schema.get("required") or [])
    props: Dict[str, Any] = schema.get("properties") or {}

    if not props:
        return create_model(
            f"MCP_{server_key}_{tool_name}_Args",
            payload=(
                Dict[str, Any],
                Field(
                    default_factory=dict, description="Opaque arguments for the tool"
                ),
            ),
        )

    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, ps in props.items():
        py_type = _pydantic_type_for_property(ps)
        desc = ps.get("description")
        default = ps.get("default", None)
        if name in required:
            fields[name] = (py_type, Field(..., description=desc))
        else:
            fields[name] = (py_type, Field(default=default, description=desc))

    return create_model(
        f"MCP_{server_key}_{tool_name}_Args",
        **fields,
    )


# ----------------------------- Tool runners ----------------------------------


def _runner_factory(server_url: str, tool_name: str):
    """
    Create an async runner function bound to a specific MCP tool.

    The runner:
      - Forwards keyword arguments to the remote MCP tool.
      - Wraps the call in 'track_tool(...)' for analytics/telemetry.
      - Returns the MCP tool’s response unchanged.

    Args:
        server_url: Full URL of the MCP server.
        tool_name: Name of the MCP tool to invoke.

    Returns:
        An async function **_runner(**kwargs) → dict**.
    """

    async def _runner(**kwargs):
        with track_tool(
            session_id="", turn_id=None, name=f"mcp:{tool_name}", args=kwargs
        ):
            return await call_tool(server_url, tool_name, kwargs)

    return _runner


# --------------------------- Public API: builder ------------------------------


async def build_langchain_tools(server_key: str) -> List[StructuredTool]:
    """
    Discover MCP tools and wrap them as LangChain StructuredTools.

    Workflow:
      1. Resolve the server URL from 'server_key'.
      2. Query MCP server for available tools via 'list_tools'.
      3. For each tool, generate a Pydantic args schema and async runner.
      4. Add each wrapped tool to a LangChain-compatible list.
      5. If 'server_key == "world_bank"', also add a local
         'worldbank_value_for_year' tool that:
           - Normalizes indicator IDs and country codes.
           - Clamps year to [1960, MAX_YEAR].
           - Calls 'get_indicator_value_for_year_via_mcp_server'.

    Args:
        server_key: Logical server identifier (e.g., "world_bank").

    Returns:
        List of LangChain 'StructuredTool' objects ready for agent/toolchain use.
    """
    server_url = resolve_server(server_key)
    discovered = await list_tools(server_url)
    lc_tools: List[StructuredTool] = []

    for t in discovered:
        name = t.get("name")
        if not name:
            continue

        desc = t.get("description") or f"Remote MCP tool '{name}'"
        schema = t.get("inputSchema") or {}
        args_model = _build_args_model_from_schema(server_key, name, schema)

        runner = _runner_factory(server_url, name)

        lc_tools.append(
            StructuredTool.from_function(
                coroutine=runner,
                name=f"mcp:{server_key}:{name}",
                description=desc,
                args_schema=args_model,
            )
        )

    if server_key == "world_bank":
        server_url_local = server_url

        async def _wb_exact_runner(
            country_id: str,
            indicator_id: str,
            year: int,
            prefer_constant: bool = True,
        ):
            """
            Resolve natural indicator names -> canonical IDs; normalize country -> ISO3;
            clamp year to 1960–2024; then fetch the exact WB value for that year.
            """
            indicator = _prefer_constant_ppp(
                _norm_indicator_id(indicator_id), prefer_constant
            )

            try:
                iso3, _ = resolve_country(country_id)
            except Exception:
                iso3 = country_id

            y = max(1960, min(int(year), MAX_YEAR))

            with track_tool(
                session_id="",
                turn_id=None,
                name="worldbank_value_for_year",
                args={
                    "country_id": country_id,
                    "indicator_id": indicator_id,
                    "year": y,
                    "prefer_constant": prefer_constant,
                },
            ):
                return await get_indicator_value_for_year_via_mcp_server(
                    server_url=server_url_local,
                    country_id=iso3,
                    indicator_id=indicator,
                    year=y,
                )

        lc_tools.append(
            StructuredTool.from_function(
                coroutine=_wb_exact_runner,
                name="worldbank_value_for_year",
                description=(
                    "World Bank (exact value, single year). "
                    "Use for questions like “<indicator> in <year> for <country>”. "
                    "If the indicator is PPP GDP per capita, this tool DEFAULTS to the constant 2021 series "
                    "(NY.GDP.PCAP.PP.KD). To explicitly request current PPP, set prefer_constant=false."
                ),
                args_schema=_WBExactArgs,
            )
        )

    return lc_tools


# ----------------------------- Explicit args model ---------------------------


class _WBExactArgs(BaseModel):
    """
    Pydantic argument schema for the local
    'worldbank_value_for_year' convenience tool.

    Fields:
        country_id (str): Country name or ISO3 code, e.g., "Albania" or "ALB".
        indicator_id (str): World Bank indicator ID or human-readable name.
        year (int): Year in range [1960, MAX_YEAR].
        prefer_constant (bool): For PPP GDP per capita indicators,
            - True → use constant 2021 international $ (NY.GDP.PCAP.PP.KD).
            - False → allow current PPP series (NY.GDP.PCAP.PP.CD).
    """
    country_id: str = Field(
        ..., description="Country name or code, e.g., 'Albania' or 'ALB'"
    )
    indicator_id: str = Field(
        ..., description="World Bank indicator code or natural name"
    )
    year: int = Field(..., ge=1960, le=MAX_YEAR, description="Year (1960–2024)")
    prefer_constant: bool = Field(
        True,
        description="If PPP GDP per capita, default to constant 2021 (KD). Set false to get current (CD).",
    )
