from __future__ import annotations
import os
from typing import Dict
from urllib.parse import urlparse, urlunparse


MCP_SERVERS: Dict[str, str] = {
    "world_bank": os.getenv("MCP_SERVER_WORLD_BANK", "stdio://world_bank"),
}


def resolve_server(server_key: str) -> str:
    """
    Resolve the endpoint URL for a named MCP server, applying environment
    variable overrides and transport normalization.

    Supported server keys:
        - "world_bank" (only key currently supported)

    Resolution priority:
        1. MCP_PROXY_ENDPOINT        → Explicit proxy base URL (http/https).
        2. MCP_SERVER_WORLD_BANK     → Explicit override (may be stdio://).
        3. WORLD_BANK_MCP_SERVER     → Legacy/alternate override.
        4. Default                   → "stdio://world_bank".

    Transport handling:
        - stdio://world_bank  → Use stdio client to spawn the WB MCP server.
        - ws:// or wss://...  → WebSocket transport; returned unchanged.
        - http(s)://.../sse   → SSE client.
        - http(s)://.../mcp   → Streamable HTTP client.
        - http(s)://host[:port] → Normalized by appending path depending on
          MCP_TRANSPORT:
              - "sse" (default) → /sse
              - "mcp"           → /mcp

    Args:
        server_key: Logical MCP server key ("world_bank").

    Returns:
        Fully-qualified MCP endpoint URL as a string.

    Raises:
        ValueError: If server_key is unsupported, or if the resolved URL is invalid.
    """
    if server_key != "world_bank":
        raise ValueError(f"Unknown MCP server key: {server_key}")

    transport = (os.getenv("MCP_TRANSPORT", "sse") or "sse").lower()
    desired_path = "/sse" if transport == "sse" else "/mcp"

    raw_url_env = os.getenv("MCP_PROXY_ENDPOINT")
    raw_url = (
        raw_url_env
        or os.getenv("MCP_SERVER_WORLD_BANK")
        or os.getenv("WORLD_BANK_MCP_SERVER")
        or "stdio://world_bank"
    ).strip()

    p = urlparse(raw_url)
    if raw_url_env:
        return raw_url.rstrip("/")

    if p.scheme == "stdio":
        return raw_url.rstrip("/")

    if p.scheme in ("ws", "wss"):
        if not p.netloc:
            raise ValueError(f"Invalid MCP server URL: {raw_url}")
        return raw_url.rstrip("/")

    if p.scheme not in ("http", "https") or not p.netloc:
        raise ValueError(f"Invalid MCP server URL: {raw_url}")

    path = (p.path or "").rstrip("/")
    if path in ("", "/"):
        path = desired_path
    elif path not in ("/sse", "/mcp"):
        pass
    else:
        if path != desired_path:
            path = desired_path
    return urlunparse((p.scheme, p.netloc, path, "", "", "")).rstrip("/")