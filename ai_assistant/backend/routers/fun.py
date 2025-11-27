from __future__ import annotations
from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, Literal, Optional
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json
import random

router = APIRouter(prefix="/api/fun", tags=["fun"])

_CACHE: Dict[str, tuple[datetime, Dict[str, Any]]] = {}
_TTL = timedelta(seconds=10)


def _get_json(
    url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 3.0
) -> Any:
    """Fetch JSON from a public HTTP endpoint with minimal validation.

    Raises HTTPException on non-200 status or invalid JSON.
    Intended for simple, optional fun endpoints; not mission critical.
    """
    req = Request(
        url,
        headers=headers
        or {"User-Agent": "fun-welcome/1.0", "Accept": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise HTTPException(502, f"Upstream error {resp.status}")
        try:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            raise HTTPException(502, "Invalid JSON from upstream")


def _cat() -> Dict[str, Any]:
    """Return a small random cat image payload suitable for the UI."""
    data = _get_json(
        "https://api.thecatapi.com/v1/images/search?mime_types=jpg,png&size=small&limit=1&order=Rand"
    )
    if isinstance(data, list) and data:
        url = data[0].get("url")
        if isinstance(url, str) and url.startswith("http"):
            return {"kind": "image", "url": url, "alt": "Cute cat", "source": "cat"}
    raise HTTPException(502, "CAT_EMPTY")


def _dog() -> Dict[str, Any]:
    """Return a small random dog image payload suitable for the UI."""
    data = _get_json(
        "https://api.thedogapi.com/v1/images/search?mime_types=jpg,png&size=small&limit=1&order=Rand"
    )
    if isinstance(data, list) and data:
        url = data[0].get("url")
        if isinstance(url, str) and url.startswith("http"):
            return {"kind": "image", "url": url, "alt": "Cute dog", "source": "dog"}
    raise HTTPException(502, "DOG_EMPTY")


def _fox() -> Dict[str, Any]:
    """Return a small random fox image payload suitable for the UI."""
    data = _get_json("https://randomfox.ca/floof")
    url = data.get("image")
    if isinstance(url, str) and url.startswith("http"):
        return {"kind": "image", "url": url, "alt": "Cute fox", "source": "fox"}
    raise HTTPException(502, "FOX_EMPTY")


_SOURCES = {"cat": _cat, "dog": _dog, "fox": _fox}


def _order(src: Literal["cat", "dog", "fox", "random"]) -> list[str]:
    """Return a preference-ordered list of sources to try."""
    if src == "random":
        keys = list(_SOURCES.keys())
        random.shuffle(keys)
        return keys
    return [src] + [k for k in _SOURCES if k != src]


@router.get("/welcome")
def welcome(
    source: Literal["cat", "dog", "fox", "random"] = "random",
    fresh: bool = Query(
        False, description="If true, bypass server cache and fetch a new image"
    ),
) -> Dict[str, Any]:
    """Return a single cute animal image for the welcome panel.

    Uses a short, in-memory cache per source to reduce upstream calls.
    If all sources fail, returns a safe fox fallback.
    """
    now = datetime.utcnow()
    key = f"welcome:{source}"
    if not fresh and key in _CACHE and _CACHE[key][0] > now:
        return _CACHE[key][1]

    for name in _order(source):
        try:
            payload = _SOURCES[name]()
            _CACHE[key] = (now + _TTL, payload)
            return payload
        except (HTTPException, URLError, HTTPError, Exception):
            continue

    return {
        "kind": "image",
        "url": "https://randomfox.ca/images/1.jpg",
        "alt": "Cute fox",
        "source": "fox",
    }
