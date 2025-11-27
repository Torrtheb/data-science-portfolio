from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import asyncio
import csv
import io
import os
import sys
import shlex
from shutil import which
from urllib.parse import urlparse
import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters


DEFAULT_TIMEOUT = 25.0
MIN_YEAR = 1960
MAX_YEAR = 2024


# ------------------------------- Utilities -----------------------------------


def _schema_dump(schema: Any) -> Any:
    """
    Safely convert an arbitrary schema-like object into a JSON-serializable dict.

    Attempts:
      1. Call '.model_dump()' if available (e.g. pydantic BaseModel).
      2. Return the object itself if already a dict.
      3. Attempt to cast to 'dict()'.

    Args:
        schema: Input object to convert.

    Returns:
        dict representation of the object, or None if conversion fails.
    """
    if schema is None:
        return None
    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump()
        except Exception:
            pass
    if isinstance(schema, dict):
        return schema
    try:
        return dict(schema)
    except Exception:
        return None


def _to_seconds(ms: Optional[int], default_seconds: float) -> float:
    """
    Convert a millisecond value to seconds, with fallback defaults.

    Args:
        ms: Timeout in milliseconds, or None to use the default.
        default_seconds: Default fallback value in seconds.

    Returns:
        Seconds as a float, clamped to at least 0.1s.
    """
    if ms is None:
        return default_seconds
    try:
        return max(0.1, float(ms) / 1000.0)
    except Exception:
        return default_seconds


def _collect_text_blocks(result) -> list[str]:
    """
    Extract plain-text blocks from an MCP call result.

    Iterates over the '.content' list and collects all 'TextContent.text' values.

    Args:
        result: MCP tool call result with a 'content' attribute.

    Returns:
        List of extracted strings (may be empty).
    """
    out = []
    try:
        for c in getattr(result, "content", []) or []:
            if isinstance(c, types.TextContent):
                if c.text:
                    out.append(c.text)
    except Exception:
        pass
    return out


def _best_structured(result):
    """
    Extract the most structured representation from an MCP call result.

    Priority:
      1. 'result.structuredContent' if present.
      2. A single 'JsonContent.json' block in 'result.content'.
      3. None if neither is available.

    Args:
        result: MCP call result.

    Returns:
        Structured object (dict/JSON-like) or None.
    """
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        return sc
    try:
        for c in getattr(result, "content", []) or []:
            if isinstance(c, types.JsonContent):
                return c.json
    except Exception:
        pass
    return None


# ----------------------------- Session handling ------------------------------


@asynccontextmanager
async def connect(server_url: str, timeout: float = DEFAULT_TIMEOUT):
    """
    Open an initialized MCP ClientSession with automatic transport selection.

    Supports:
      - 'stdio://...' → Launch World Bank MCP server as subprocess.
      - 'http(s)://.../mcp' → Streamable HTTP client.
      - 'http(s)://.../sse' → Server-Sent Events.
      - 'http(s)://' (root) → Try streamable HTTP, then SSE.

    Args:
        server_url: URL string specifying transport type.
        timeout: Initialization timeout in seconds.

    Yields:
        An active 'ClientSession'.

    Raises:
        RuntimeError: If initialization fails, times out, or transport unsupported.
    """
    u = urlparse(server_url)

    try:
        if u.scheme == "stdio":
            extra_args = shlex.split(os.getenv("MCP_WORLD_BANK_ARGS", ""))
            env_cmd = (
                os.getenv("MCP_WORLD_BANK_CMD") or "world-bank-mcp-server"
            ).strip()
            env_parts = shlex.split(env_cmd) if env_cmd else []

            candidates: list[tuple[str, list[str], str]] = []
            if env_parts:
                candidates.append(
                    (
                        env_parts[0],
                        env_parts[1:] + extra_args,
                        "env:$MCP_WORLD_BANK_CMD",
                    )
                )
            for exe in ("world-bank-mcp-server", "world_bank_mcp_server"):
                candidates.append((exe, list(extra_args), f"console:{exe}"))

            py = sys.executable or "python"
            for mod in (
                "world_bank_mcp_server.server",
                "world_bank_mcp_server.cli",
                "world_bank_mcp_server.app",
                "world_bank_mcp_server",
            ):
                candidates.append((py, ["-m", mod, *extra_args], f"module:{mod}"))

            code_snips = [
                "from world_bank_mcp_server.cli import main; main()",
                "from world_bank_mcp_server.server import main; main()",
            ]
            for i, snip in enumerate(code_snips, start=1):
                candidates.append((py, ["-c", snip, *extra_args], f"python:-c#{i}"))

            attempts_log: list[str] = []
            last_err: Exception | None = None

            for exe, args, label in candidates:
                cmd = exe
                if exe not in (sys.executable, "python", "python3"):
                    path = which(exe)
                    if not path:
                        attempts_log.append(f"{label}: NOT FOUND on PATH")
                        continue
                    cmd = path

                try:
                    params = StdioServerParameters(command=cmd, args=args)
                    async with stdio_client(params) as (read, write, _sid):
                        async with ClientSession(read, write) as session:
                            await asyncio.wait_for(
                                session.initialize(), timeout=timeout
                            )
                            try:
                                yield session
                            finally:
                                pass
                            return
                except Exception as e:
                    last_err = e
                    attempts_log.append(f"{label}: {type(e).__name__}: {e}")
                    continue

            raise RuntimeError(
                "Could not launch World Bank MCP server via stdio.\n"
                + "Attempts:\n  - "
                + "\n  - ".join(attempts_log)
                + (
                    f"\nLast error: {type(last_err).__name__}: {last_err}"
                    if last_err
                    else ""
                )
            )

        # --------------------- HTTP / HTTPS transports ---------------------
        if u.scheme in ("http", "https"):
            path = (u.path or "").rstrip("/")

            if path in ("", "/"):
                try:
                    async with streamablehttp_client(server_url) as (read, write, _sid):
                        async with ClientSession(read, write) as session:
                            await asyncio.wait_for(
                                session.initialize(), timeout=timeout
                            )
                            try:
                                yield session
                            finally:
                                pass
                            return
                except Exception:
                    async with sse_client(server_url) as (read, write, _sid):
                        async with ClientSession(read, write) as session:
                            await asyncio.wait_for(
                                session.initialize(), timeout=timeout
                            )
                            try:
                                yield session
                            finally:
                                pass
                            return

            if path.endswith("/mcp"):
                async with streamablehttp_client(server_url) as (read, write, _sid):
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=timeout)
                        try:
                            yield session
                        finally:
                            pass
                        return

            if path.endswith("/sse"):
                async with sse_client(server_url) as (read, write, _sid):
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=timeout)
                        try:
                            yield session
                        finally:
                            pass
                        return

        raise RuntimeError(f"Unsupported MCP URL: {server_url}")

    except asyncio.TimeoutError:
        raise RuntimeError(
            f"MCP initialize timed out after {timeout:.1f}s ({server_url})"
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise RuntimeError(f"MCP endpoint unreachable: {server_url} ({e})")
    except Exception as e:
        raise RuntimeError(
            f"MCP endpoint unreachable or failed handshake: {server_url} "
            f"({type(e).__name__}: {e})"
        )


# --------------------------------- Remote Procedure Calls --------------------------------------


async def list_tools(
    server_url: str, *, timeout_ms: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve and normalize the list of tools exposed by the MCP server.

    Args:
        server_url: MCP server endpoint (stdio/http/https).
        timeout_ms: Optional RPC timeout in milliseconds.

    Returns:
        List of dicts with keys:
          - name (str)
          - description (str)
          - inputSchema (dict or None)

    Raises:
        RuntimeError: On timeout, connection failure, or handshake errors.
    """

    call_timeout = _to_seconds(timeout_ms, DEFAULT_TIMEOUT)
    async with connect(server_url) as s:
        try:
            resp = await asyncio.wait_for(s.list_tools(), timeout=call_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCP list_tools timed out after {call_timeout:.1f}s ({server_url})"
            )

        try:
            items = resp.tools
        except Exception:
            items = getattr(resp, "get", lambda *_: [])("tools")

        out: List[Dict[str, Any]] = []
        for t in items or []:
            if isinstance(t, dict):
                out.append(
                    {
                        "name": t.get("name", ""),
                        "description": t.get("description") or "",
                        "inputSchema": _schema_dump(t.get("inputSchema")),
                    }
                )
            else:
                out.append(
                    {
                        "name": getattr(t, "name", "") or "",
                        "description": getattr(t, "description", "") or "",
                        "inputSchema": _schema_dump(getattr(t, "inputSchema", None)),
                    }
                )
        return out


async def call_tool(
    server_url: str,
    tool: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    timeout_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a tool remotely via the MCP server.

    Args:
        server_url: MCP server endpoint.
        tool: Tool name to invoke.
        args: Arguments for the tool as a dict.
        timeout_ms: Optional RPC timeout in milliseconds.

    Returns:
        Dict with keys:
          - text (str|None): Concatenated plain-text output blocks.
          - structured (Any|None): Best structured JSON-like result.
          - raw (Any): Full raw response object or its JSON dump.

    Raises:
        RuntimeError: On timeout or connection errors.
    """
    args = args or {}
    call_timeout = _to_seconds(timeout_ms, DEFAULT_TIMEOUT)

    async with connect(server_url) as s:
        try:
            result = await asyncio.wait_for(
                s.call_tool(tool, arguments=args), timeout=call_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCP call_tool('{tool}') timed out after {call_timeout:.1f}s ({server_url})"
            )

        text_blocks = _collect_text_blocks(result)

        return {
            "text": ("\n".join(t for t in text_blocks if t).strip() or None),
            "structured": _best_structured(result),
            "raw": (
                result.model_dump(mode="json")
                if hasattr(result, "model_dump")
                else result
            ),
        }


# --------------------------- World Bank convenience ---------------------------


def _parse_worldbank_csv(text: str) -> List[Dict[str, Any]]:
    """
    Parse a World Bank CSV payload into structured rows.

    Expected columns:
      - date (year, e.g. "2022")
      - value (numeric string or blank)
      - country.value
      - countryiso3code
      - indicator.value

    Args:
        text: CSV-formatted string.

    Returns:
        Sorted list of dict rows:
        [
          {"date": int, "value": float|None,
           "country": str, "iso3": str, "indicator": str}, ...
        ]
    """
    if not text:
        return []

    rdr = csv.DictReader(io.StringIO(text))
    out: List[Dict[str, Any]] = []
    for r in rdr:
        try:
            y = int((r.get("date") or "").strip())
        except Exception:
            continue
        raw = (r.get("value") or "").strip()
        val = float(raw) if raw not in ("", None) else None
        out.append(
            {
                "date": y,
                "value": val,
                "country": r.get("country.value") or "",
                "iso3": r.get("countryiso3code") or "",
                "indicator": r.get("indicator.value") or "",
            }
        )
    out.sort(key=lambda x: x["date"])
    return out


async def get_indicator_value_for_year_via_mcp_server(
    server_url: str,
    country_id: str,
    indicator_id: str,
    year: int,
    *,
    timeout_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fetch a single indicator value for a given country-year via the World Bank MCP server.

    Workflow:
      1. Calls 'get_indicator_for_country' tool remotely.
      2. Parses returned CSV into structured rows.
      3. Looks up the requested year, clamped to [1960, 2024].

    Args:
        server_url: MCP server endpoint.
        country_id: ISO-3 country code (preferred) or name.
        indicator_id: World Bank indicator ID (e.g., "NY.GDP.PCAP.PP.KD").
        year: Target year.
        timeout_ms: Optional per-call timeout.

    Returns:
        Dict with fields:
          - ok (bool)
          - country (str)
          - country_iso3 (str|None)
          - indicator_id (str)
          - indicator_name (str|None)
          - year (int)
          - value (float|None)
          - unit (str|None, special-cased for GDP PPP)
          - source_url (str)
          - message (str|None, only if ok=False)
    """
    data = await call_tool(
        server_url,
        "get_indicator_for_country",
        {"country_id": country_id, "indicator_id": indicator_id},
        timeout_ms=timeout_ms,
    )
    csv_text = (data or {}).get("text") or ""
    series = _parse_worldbank_csv(csv_text)
    y = max(MIN_YEAR, min(int(year), MAX_YEAR))
    row = next((r for r in series if r["date"] == y and r["value"] is not None), None)
    source_url = f"https://api.worldbank.org/v2/country/{country_id}/indicator/{indicator_id}?format=csv"

    if not row:
        return {
            "ok": False,
            "country": country_id,
            "indicator_id": indicator_id,
            "year": y,
            "message": "No World Bank value for the requested year.",
            "source_url": source_url,
        }

    return {
        "ok": True,
        "country": row["country"] or country_id,
        "country_iso3": row["iso3"] or None,
        "indicator_id": indicator_id,
        "indicator_name": row["indicator"] or None,
        "year": row["date"],
        "value": row["value"],
        "unit": (
            "2021 international $ (PPP, constant)"
            if indicator_id == "NY.GDP.PCAP.PP.KD"
            else None
        ),
        "source_url": source_url,
    }
