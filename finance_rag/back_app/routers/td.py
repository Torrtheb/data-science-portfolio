from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
from datetime import datetime, timezone
import httpx
from fastapi import APIRouter, HTTPException, Query
from collections import deque
import time, asyncio

try:
    from ..core.settings import settings
except Exception:
    settings = None

td_router = APIRouter(tags=["twelve-data"])
_TRANSPORT = httpx.AsyncHTTPTransport(retries=2)
_TD_RPS = int(os.getenv("TD_RPS", "8"))
_TD_CONCURRENCY = int(os.getenv("TD_CONCURRENCY", "4"))
_TD_CACHE_TTL = float(os.getenv("TD_CACHE_TTL", "20.0"))
_sem = asyncio.Semaphore(_TD_CONCURRENCY)
_req_times: deque[float] = deque()
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_TD_RETRIES = int(os.getenv("TD_RETRIES", "1"))
_ALLOWED_INTERVALS = {
    "1min",
    "5min",
    "15min",
    "30min",
    "45min",
    "1h",
    "2h",
    "4h",
    "8h",
    "1day",
    "1week",
    "1month",
}

# ---------------------------- HTTP client ------------------------------------


def _client(timeout: float = 12.0) -> httpx.AsyncClient:
    """
    Create and return a reusable asynchronous HTTP client for Twelve Data API requests.

    Args:
        timeout (float, optional): Request timeout in seconds. Defaults to 12.0.

    Returns:
        httpx.AsyncClient: Configured async client with retry transport,
        custom user-agent, HTTP/2 support, and connection pooling.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        transport=_TRANSPORT,
        headers={"User-Agent": "finance-chatbot/td/1.0"},
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
        http2=True,
    )


# ---------------------------- Key loading ------------------------------------


def _get_td_key() -> str:
    """
    Load the Twelve Data API key from settings or environment variables.

    Priority:
        1. 'settings.twelvedata_api_key' if available
        2. 'TWELVEDATA_API_KEY' environment variable
        3. 'TWELVE_DATA_KEY' environment variable

    Returns:
        str: The Twelve Data API key string, or an empty string if not set.
    """
    if settings and getattr(settings, "twelvedata_api_key", None):
        try:
            return settings.twelvedata_api_key.get_secret_value()
        except Exception:
            return str(settings.twelvedata_api_key)
    return os.getenv("TWELVEDATA_API_KEY") or os.getenv("TWELVE_DATA_KEY") or ""


# ---------------------------- Utils -----------------------------------------


def _mk_cache_key(url: str, params: Dict[str, Any]) -> str:
    """
    Construct a stable cache key for Twelve Data requests.

    Args:
        url (str): Request base URL.
        params (dict): Query parameters for the request.

    Returns:
        str: Unique cache key combining URL and sorted parameters.
    """
    return url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))


async def _td_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a Twelve Data API request with caching, rate-limiting, and retries.

    Features:
        * Uses in-memory cache with TTL ('_TD_CACHE_TTL')
        * Enforces request rate limit ('_TD_RPS')
        * Applies concurrency limit via semaphore ('_TD_CONCURRENCY')
        * Retries failed requests up to '_TD_RETRIES'
        * Converts API errors into FastAPI HTTPExceptions

    Args:
        url (str): API endpoint URL.
        params (dict): Request parameters.

    Returns:
        dict: Parsed JSON response from Twelve Data API.

    Raises:
        HTTPException: If Twelve Data responds with an error or after retries fail.
    """
    ckey = _mk_cache_key(url, params)
    now = time.monotonic()
    hit = _cache.get(ckey)
    if hit and (now - hit[0]) < _TD_CACHE_TTL:
        return hit[1]

    async with _sem:
        now = time.monotonic()
        while _req_times and (now - _req_times[0]) > 1.0:
            _req_times.popleft()
        if len(_req_times) >= _TD_RPS:
            sleep_for = 1.0 - (now - _req_times[0])
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        _req_times.append(time.monotonic())

        attempt = 0
        while True:
            async with _client() as client:
                try:
                    r = await client.get(url, params=params)
                except httpx.HTTPError as e:
                    if attempt < _TD_RETRIES:
                        attempt += 1
                        await asyncio.sleep(0.2 * attempt)
                        continue
                    raise HTTPException(502, detail=f"Twelve Data network error: {e!s}")

            text = r.text
            try:
                js = r.json()
            except Exception:
                js = {}

            print(
                f"[TD] {r.status_code} {params.get('symbol')} {params.get('interval')} -> {js or text[:200]}"
            )
            if isinstance(js, dict) and (js.get("status") == "error" or js.get("code")):
                msg = js.get("message") or js.get("status") or "error"
                code = r.status_code or 400
                if 500 <= code < 600 and attempt < _TD_RETRIES:
                    attempt += 1
                    await asyncio.sleep(0.2 * attempt)
                    continue
                if code in (400, 401, 403, 429):
                    raise HTTPException(code, detail=f"Twelve Data: {msg}")
                raise HTTPException(502, detail=f"Twelve Data: {msg}")
            if r.status_code >= 400:
                if 500 <= r.status_code < 600 and attempt < _TD_RETRIES:
                    attempt += 1
                    await asyncio.sleep(0.2 * attempt)
                    continue
                if r.status_code in (400, 401, 403, 429):
                    raise HTTPException(
                        r.status_code,
                        detail=f"Twelve Data HTTP {r.status_code}: {text[:200]}",
                    )
                raise HTTPException(
                    502, detail=f"Twelve Data HTTP {r.status_code}: {text[:200]}"
                )
            break

        _cache[ckey] = (time.monotonic(), js)
        return js


def _to_ts(dt_str: str) -> int:
    """
    Convert a Twelve Data datetime string into a UTC epoch timestamp (seconds).

    Accepts formats like:
        - '"YYYY-MM-DD HH:MM:SS"'
        - '"YYYY-MM-DDTHH:MM:SS"'
        - '"YYYY-MM-DDTHH:MM:SSZ"'

    Args:
        dt_str (str): Input datetime string.

    Returns:
        int: UTC epoch timestamp in seconds.
    """
    s = (dt_str or "").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        dt = datetime.fromisoformat(s.replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _validate_interval(interval: str) -> None:
    """
    Validate that the requested interval is supported.

    Args:
        interval (str): Interval string, e.g., "1day", "1h", "5min".

    Raises:
        HTTPException: If interval is not in '_ALLOWED_INTERVALS'.
    """
    if interval not in _ALLOWED_INTERVALS:
        raise HTTPException(
            400,
            detail=f"Unsupported interval '{interval}'. Allowed: {sorted(_ALLOWED_INTERVALS)}",
        )


# ---------------------------- Endpoints --------------------------------------


@td_router.get("/api/td/candles")
async def td_candles(
    symbol: str = Query(..., min_length=1, description="Ticker, e.g., AAPL or SU.TO"),
    interval: str = Query(
        "1day",
        description="One of: 1min,5min,15min,30min,45min,1h,2h,4h,8h,1day,1week,1month",
    ),
    outputsize: int = Query(500, ge=1, le=5000, description="Max points to return"),
    exchange: Optional[str] = Query(None, description="Optional exchange hint"),
    country: Optional[str] = Query(None, description="Optional country hint"),
) -> Dict[str, Any]:
    """
    Retrieve historical OHLCV (candlestick) data from the Twelve Data API.

    Args:
        symbol (str): Ticker symbol (e.g., "AAPL").
        interval (str, optional): Candle interval (default "1day").
        outputsize (int, optional): Maximum number of data points (default 500).
        exchange (str, optional): Exchange hint for ambiguous tickers.
        country (str, optional): Country hint for ticker resolution.

    Returns:
        dict: JSON response containing:
            - s (str): Status ("ok" or error)
            - t (list[int]): Timestamps (epoch seconds)
            - o, h, l, c (list[float]): Open, high, low, close prices
            - symbol (str): Uppercased ticker symbol

    Raises:
        HTTPException: If API key is missing, interval unsupported, or no data found.
    """
    _validate_interval(interval)

    key = _get_td_key()
    if not key:
        raise HTTPException(
            500,
            detail="Twelve Data API key not set. Configure TWELVEDATA_API_KEY (or TWELVE_DATA_KEY).",
        )

    params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": key,
        "format": "JSON",
        "order": "ASC",
        "timezone": "UTC",
    }
    if exchange:
        params["exchange"] = exchange
    if country:
        params["country"] = country

    js = await _td_get_json("https://api.twelvedata.com/time_series", params)

    values: List[Dict[str, Any]] = (js or {}).get("values") or []
    if values and values[0].get("datetime") > values[-1].get("datetime"):
        values = list(reversed(values))

    t: List[int]
    o: List[float]
    h: List[float]
    l: List[float]
    c: List[float]
    t, o, h, l, c = [], [], [], [], []
    for v in values:
        dt = v.get("datetime")
        if not dt:
            continue
        try:
            t.append(_to_ts(dt))
            o.append(float(v["open"]))
            h.append(float(v["high"]))
            l.append(float(v["low"]))
            c.append(float(v["close"]))
        except Exception:
            continue

    if not t:
        raise HTTPException(
            404, detail=f"No candle data for {symbol} at interval {interval}"
        )

    return {"s": "ok", "t": t, "c": c, "o": o, "h": h, "l": l, "symbol": symbol.upper()}


@td_router.get("/api/td/sparkline12")
async def td_sparkline12(
    symbol: str = Query(..., min_length=1, description="Ticker, e.g., AAPL"),
    points: int = Query(12, ge=2, le=200, description="Number of monthly closes"),
) -> Dict[str, Any]:
    """
    Retrieve a simplified monthly sparkline series (last N closing prices).

    Args:
        symbol (str): Ticker symbol (e.g., "AAPL").
        points (int, optional): Number of months to include (default 12).

    Returns:
        dict: JSON response containing:
            - s (str): Status ("ok" or "error")
            - symbol (str): Uppercased ticker symbol
            - t (list[int]): Monthly timestamps
            - c (list[float]): Monthly close prices
            - points (int): Actual number of returned points
    """
    key = _get_td_key()
    if not key:
        return {
            "s": "error",
            "symbol": symbol.upper(),
            "t": [],
            "c": [],
            "hint": "No Twelve Data key",
        }

    params = {
        "symbol": symbol,
        "interval": "1month",
        "outputsize": points,
        "apikey": key,
        "format": "JSON",
        "order": "ASC",
        "timezone": "UTC",
    }
    js = await _td_get_json("https://api.twelvedata.com/time_series", params)

    values: List[Dict[str, Any]] = (js or {}).get("values") or []
    if not values:
        return {"s": "ok", "symbol": symbol.upper(), "t": [], "c": []}

    if values and values[0].get("datetime") > values[-1].get("datetime"):
        values = list(reversed(values))

    t: List[int]
    c_: List[float]
    t, c_ = [], []
    for v in values[-points:]:
        try:
            t.append(_to_ts(v["datetime"]))
            c_.append(float(v["close"]))
        except Exception:
            continue

    if len(t) > points:
        t, c_ = t[-points:], c_[-points:]

    return {"s": "ok", "symbol": symbol.upper(), "t": t, "c": c_, "points": len(t)}
