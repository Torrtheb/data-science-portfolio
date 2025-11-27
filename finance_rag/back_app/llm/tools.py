from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal, Sequence, Union
import os
import time
import httpx
import asyncio
import random
import re
from loguru import logger
from decimal import Decimal, getcontext, ROUND_HALF_UP
from datetime import date, timedelta
from dotenv import load_dotenv
from ..core.settings import settings

load_dotenv()

getcontext().prec = 28

_ALLOWED_CANDLE_RES = {"1", "5", "15", "30", "60", "D", "W", "M"}

FREQUENCY_MAP: Dict[str, int] = {
    "annually": 1,
    "semiannually": 2,
    "quarterly": 4,
    "bimonthly": 6,
    "monthly": 12,
    "semimonthly": 24,
    "biweekly": 26,
    "weekly": 52,
    "daily": 365,
}

_SYNONYMS: Dict[str, str] = {
    "yearly": "annually",
    "annual": "annually",
    "semiannual": "semiannually",
    "semi-annual": "semiannually",
    "semi annually": "semiannually",
    "quarter": "quarterly",
    "twicemonthly": "semimonthly",
    "twiceamonth": "semimonthly",
    "twice-a-month": "semimonthly",
    "bi-weekly": "biweekly",
    "fortnightly": "biweekly",
    "month": "monthly",
    "week": "weekly",
    "day": "daily",
}


def _finnhub_key() -> Optional[str]:
    """Return Finnhub API key as a plain string (unwrap SecretStr if needed)."""
    raw = os.getenv("FINNHUB_API_KEY") or getattr(settings, "finnhub_api_key", None)
    if raw is None:
        return None
    if hasattr(raw, "get_secret_value"):
        return raw.get_secret_value()
    return str(raw)


def _finnhub_base() -> str:
    """Return the base URL for the Finnhub API (sandbox if the key starts with 'sandbox_')."""
    key = _finnhub_key() or ""
    return (
        "https://sandbox.finnhub.io/api/v1"
        if key.startswith("sandbox_")
        else "https://finnhub.io/api/v1"
    )


# --------- HTTP helper ---------
async def _fh_get(
    path: str, params: Dict[str, Any]
) -> Union[Dict[str, Any], List[Any]]:
    """
    Perform a robust GET request to the Finnhub API.

    This helper function:
    - Ensures the Finnhub API key is attached to the request (?token=...).
    - Normalizes and returns JSON responses (dict or list) for successful calls.
    - Retries transient failures (timeouts, network errors, 429 rate limits)
      with exponential backoff (0.5s, 1.0s).
    - Maps common HTTP status codes to structured error responses.
    - Special-cases Finnhub candle responses where {"s": "no_data"} is returned.

    Args:
        path (str): Endpoint path (e.g., "/quote", "/stock/candle").
        params (Dict[str, Any]): Query parameters to include in the request.

    Returns:
        Dict[str, Any]:
            - On success: Parsed JSON response from Finnhub (dict or list).
            - On error: A standardized error dict with keys:
                {
                    "error": <error_type>,   # e.g., "config", "forbidden", "rate_limited"
                    "detail": <str>,         # descriptive message or server response
                    "status": <int>,         # HTTP status (if available)
                    "path": <str>            # API path used
                }
              For candle endpoints with no data:
                {"error": "no_data", "note": "No candles for that range/symbol."}
    """
    key = _finnhub_key()
    if not key:
        return {"error": "config", "detail": "FINNHUB_API_KEY is not set", "path": path}

    if not path.startswith("/"):
        path = "/" + path

    url = f"{_finnhub_base()}{path}"
    logger.debug("Finnhub GET {} {} params={}", _finnhub_base(), path, params)
    attempts = 3
    base_backoff = 0.5

    try:
        async with httpx.AsyncClient(
            timeout=getattr(settings, "external_timeout", 15),
            headers={"User-Agent": "FinAssist/1.0"},
        ) as client:
            for attempt in range(attempts):
                try:
                    r = await client.get(url, params={**params, "token": key})
                    r.raise_for_status()
                    data = r.json()
                    if (
                        path == "/stock/candle"
                        and isinstance(data, dict)
                        and data.get("s") == "no_data"
                    ):
                        return {
                            "error": "no_data",
                            "note": "No candles for that range/symbol.",
                        }

                    if isinstance(data, dict) and "error" in data and data.get("error"):
                        return {
                            "error": "api_error",
                            "detail": data.get("error", ""),
                            "path": path,
                        }

                    return data

                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response else "?"
                    body = e.response.text if e.response is not None else ""

                    if status == 429 and attempt < attempts - 1:
                        await asyncio.sleep(base_backoff * (attempt + 1))
                        continue

                    if status == 401:
                        msg = "unauthorized"
                    elif status == 403:
                        msg = "forbidden"
                    elif status == 404:
                        msg = "not_found"
                    elif status == 429:
                        msg = "rate_limited"
                    else:
                        msg = "http_error"

                    logger.warning("Finnhub HTTP error {} {}: {}", status, path, body)
                    return {
                        "error": msg,
                        "status": status,
                        "detail": body,
                        "path": path,
                    }

                except (httpx.TimeoutException, httpx.RequestError) as e:
                    if attempt < attempts - 1:
                        await asyncio.sleep(base_backoff * (attempt + 1))
                        continue
                    logger.exception("Finnhub network error on {}: {}", path, e)
                    return {"error": "network_error", "detail": str(e), "path": path}

                except Exception as e:
                    logger.exception("Finnhub request failed on {}: {}", path, e)
                    return {"error": "unknown_error", "detail": str(e), "path": path}

    except Exception as e:
        logger.exception("Finnhub client failure on {}: {}", path, e)
        return {"error": "client_error", "detail": str(e), "path": path}


# --------- Public tools ---------
# 1) Live price (/quote)
async def get_price(symbol: str) -> Dict[str, Any]:
    """
    Fetch the latest market quote for a given stock symbol from Finnhub.

    This function:
    - Validates and normalizes the input symbol (strips spaces, uppercases).
    - Calls the Finnhub '/quote' endpoint via the _fh_get helper.
    - Returns a standardized dictionary containing the latest quote details.
    - Propagates standardized error dictionaries from _fh_get if the request fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "TSLA").

    Returns:
        Dict[str, Any]:
            On success:
                {
                    "symbol": str,        # validated ticker symbol
                    "price": float,       # current price (c)
                    "open": float,        # today's open price (o)
                    "high": float,        # today's high price (h)
                    "low": float,         # today's low price (l)
                    "prev_close": float,  # previous close price (pc)
                    "change": float,      # absolute change (d)
                    "change_pct": float,  # percent change (dp)
                    "ts": int             # Unix timestamp of quote (t)
                }
            On error:
                A standardized error dict from _fh_get, or:
                {"error": "validation", "detail": "Symbol is required"}
    """
    s = (symbol or "").strip().upper()
    if not s:
        return {"error": "validation", "detail": "Symbol is required"}

    q = await _fh_get("/quote", {"symbol": s})
    if (
        isinstance(q, dict)
        and "error" in q
        and (":" in s or s.endswith(("USD", "USDT")))
    ):
        now = int(time.time())
        c = await _fh_get(
            "/crypto/candle",
            {"symbol": s, "resolution": "1", "from": now - 3600, "to": now},
        )
        if isinstance(c, dict) and c.get("s") == "ok" and c.get("c"):
            price = c["c"][-1]
            return {
                "symbol": s,
                "price": price,
                "open": None,
                "high": None,
                "low": None,
                "prev_close": None,
                "change": None,
                "change_pct": None,
                "changePercent": None,
                "ts": c["t"][-1] if c.get("t") else now,
            }
        return q

    if isinstance(q, dict) and "error" in q:
        return q

    price = q.get("c")
    open_ = q.get("o")
    high = q.get("h")
    low = q.get("l")
    prev_close = q.get("pc")
    change = q.get("d")
    change_pct = q.get("dp")
    ts = q.get("t")

    return {
        "symbol": s,
        "price": price,
        "open": open_,
        "high": high,
        "low": low,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "changePercent": change_pct,
        "ts": ts,
    }


# 2) Screener (basic financials)
async def screen_equities(
    symbols: List[str],
    min_market_cap: float = 10e9,
    min_dividend_yield: float = 0.0,
    max_pe: float = 40.0,
) -> List[Dict[str, Any]]:
    """
    Screen a list of stock symbols by fundamental criteria using Finnhub's basic financials.

    For each symbol, this function fetches '/company-basic-financials?metric=all' and
    filters the result by:
      - market capitalization >= min_market_cap
      - dividend yield (TTM) >= min_dividend_yield
      - P/E (TTM) <= max_pe

    The function is resilient to per-symbol failures: symbols with API errors,
    missing metrics, or parsing issues are logged and skipped rather than raising.

    Args:
        symbols (List[str]): Ticker symbols to evaluate (e.g., ["AAPL", "MSFT"]).
        min_market_cap (float): Minimum market cap threshold in base currency units.
            Default is 10e9 (i.e., 10,000,000,000).
        min_dividend_yield (float): Minimum trailing-twelve-month dividend yield (e.g., 0.02 for 2%).
            Default is 0.0 (no minimum).
        max_pe (float): Maximum trailing-twelve-month price-to-earnings ratio. Default is 40.0.

    Returns:
        List[Dict[str, Any]]: A list of screened symbols with normalized fields:
            [
              {
                "symbol": str,       # uppercased ticker
                "pe": float,         # P/E (peBasicExclExtraTTM if available, else peTTM)
                "market_cap": float, # marketCapitalization
                "div_yield": float   # dividendYieldTTM (0 if not provided)
              },
              ...
            ]

    Notes:
        - Symbols are uppercased and blanks are ignored.
        - API errors from _fh_get are logged (warning) and the symbol is skipped.
        - Missing/NaN metrics or cast failures cause the symbol to be skipped.
        - Dividend yield is expected as a fraction (e.g., 0.025 = 2.5%).
    """
    rows: List[Dict[str, Any]] = []
    for raw in symbols:
        s = (raw or "").strip().upper()
        if not s:
            continue
        bf = await _fh_get("/company-basic-financials", {"symbol": s, "metric": "all"})
        if "error" in bf:
            logger.warning("{}: {}", s, bf.get("error"))
            continue
        m = (bf or {}).get("metric", {}) or {}
        mcap = m.get("marketCapitalization")
        pe = m.get("peBasicExclExtraTTM") or m.get("peTTM")
        dy = m.get("dividendYieldTTM") or 0
        try:
            if (
                mcap is not None
                and pe is not None
                and float(mcap) >= float(min_market_cap)
                and float(dy) >= float(min_dividend_yield)
                and float(pe) <= float(max_pe)
            ):
                rows.append(
                    {
                        "symbol": s,
                        "pe": float(pe),
                        "market_cap": float(mcap),
                        "div_yield": float(dy),
                    }
                )
        except Exception as e:
            logger.warning("{}: filtering error: {}", s, e)
            continue
    return rows


# 3) Candles (for charts)
async def get_candles(
    symbol: str,
    resolution: Literal["1", "5", "15", "30", "60", "D", "W", "M"],
    _from: int,
    to: int,
) -> Dict[str, Any]:
    """
    Retrieve historical candlestick (OHLCV) data for a given symbol from Finnhub.

    This function:
    - Validates and normalizes the input symbol.
    - Ensures the resolution and time range are valid.
    - Calls the Finnhub '/stock/candle' endpoint via the _fh_get helper.
    - Returns raw candle data or a standardized error dictionary.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL").
        resolution (Literal): Granularity of candles.
            - "1", "5", "15", "30", "60" → minute intervals
            - "D" → daily
            - "W" → weekly
            - "M" → monthly
        _from (int): Start time in Unix epoch seconds.
        to (int): End time in Unix epoch seconds. Must be greater than '_from'.

    Returns:
        Dict[str, Any]:
            On success (from Finnhub):
                {
                  "c": [float],  # close prices
                  "o": [float],  # open prices
                  "h": [float],  # high prices
                  "l": [float],  # low prices
                  "v": [float],  # volumes
                  "t": [int],    # timestamps (epoch seconds)
                  "s": "ok"
                }
            On no data:
                {"error": "no_data", "note": "No candles for that range/symbol."}
            On validation error:
                {"error": "validation", "detail": "..."}
            On API/HTTP error:
                Standardized error dict from _fh_get.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return {"error": "validation", "detail": "Symbol is required"}
    if resolution not in _ALLOWED_CANDLE_RES:
        return {"error": "validation", "detail": f"Invalid resolution '{resolution}'"}
    if not isinstance(_from, int) or not isinstance(to, int) or to <= _from:
        return {"error": "validation", "detail": "'to' must be > '_from' (seconds)"}
    endpoint = (
        "/crypto/candle"
        if (":" in s or s.endswith(("USD", "USDT")))
        else "/stock/candle"
    )
    data = await _fh_get(
        endpoint,
        {"symbol": s, "resolution": resolution, "from": _from, "to": to},
    )
    return data


# 4) Symbol search (/search)
async def search_symbol(query: str) -> List[Dict[str, Any]]:
    """
    Search for tradable instruments (tickers) matching a free-text query via Finnhub.

    This helper:
    - Trims and validates the input query (returns [] if empty).
    - Calls Finnhub's '/search' endpoint using the _fh_get helper.
    - Normalizes the response into a compact list of dictionaries.

    Args:
        query (str): Free-text input such as a company name or ticker
            (e.g., "apple", "AAPL", "tesla", "Air Canada").

    Returns:
        List[Dict[str, Any]]:
            A list of matches with the following fields (if available):
            [
              {
                "symbol": str,        # e.g., "AAPL"
                "description": str,   # e.g., "Apple Inc"
                "type": str,          # e.g., "Common Stock", "ETF"
                "mic": str            # Exchange MIC, e.g., "XNAS"
              },
              ...
            ]

            If the query is empty or the Finnhub call fails, returns an empty list.

    Notes:
        - Uses _fh_get for API key injection, retries, and error normalization.
        - This function intentionally swallows errors and returns [] to keep
          the search UX simple; log upstream if you need diagnostics.
    """
    q = (query or "").strip()
    if not q:
        return []
    data = await _fh_get("/search", {"q": q})
    if "error" in data:
        return []
    out: List[Dict[str, Any]] = []
    for r in data.get("result", []) or []:
        out.append(
            {
                "symbol": r.get("symbol"),
                "description": r.get("description"),
                "type": r.get("type"),
                "mic": r.get("mic"),
            }
        )
    return out


# 5) Company profile (/stock/profile2)
async def get_profile(symbol: str) -> Dict[str, Any]:
    """
    Retrieve basic company profile information for a given stock symbol from Finnhub.

    This function:
    - Validates and normalizes the input symbol (uppercase, strip whitespace).
    - Calls the Finnhub '/stock/profile2' endpoint via the _fh_get helper.
    - Returns a standardized dictionary with selected company profile fields.
    - Propagates standardized error dictionaries from _fh_get on failure.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "TSLA").

    Returns:
        Dict[str, Any]:
            On success:
                {
                  "symbol": str,         # normalized ticker symbol
                  "name": str,           # company name
                  "exchange": str,       # exchange name
                  "currency": str,       # reporting currency
                  "ticker": str,         # ticker symbol (redundant with 'symbol')
                  "market_cap": float,   # market capitalization
                  "ipo": str,            # IPO date (YYYY-MM-DD)
                  "logo": str            # company logo URL
                }
            On validation error:
                {"error": "validation", "detail": "Symbol is required"}
            On API/HTTP error:
                Standardized error dict from _fh_get.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return {"error": "validation", "detail": "Symbol is required"}
    data = await _fh_get("/stock/profile2", {"symbol": s})
    if "error" in data:
        return data
    return {
        "symbol": s,
        "name": data.get("name"),
        "exchange": data.get("exchange"),
        "currency": data.get("currency"),
        "ticker": data.get("ticker"),
        "market_cap": data.get("marketCapitalization"),
        "ipo": data.get("ipo"),
        "logo": data.get("logo"),
    }


# 6) Analyst recommendation trends (/stock/recommendation)
async def get_recommendation_trends(symbol: str) -> Dict[str, Any]:
    """
    Retrieve the most recent analyst recommendation trends for a stock symbol from Finnhub.

    This function:
    - Validates and normalizes the input symbol.
    - Calls the Finnhub '/stock/recommendation' endpoint via the _fh_get helper.
    - Returns the most recent recommendation snapshot (first element of the list).
    - Adds the 'symbol' field to the returned dictionary.
    - Propagates standardized error dictionaries from _fh_get if the request fails.

    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL", "TSLA").

    Returns:
        Dict[str, Any]:
            On success (latest recommendation snapshot):
                {
                  "symbol": str,       # normalized ticker symbol
                  "period": str,       # reporting month, e.g., "2023-07-01"
                  "strongBuy": int,    # number of Strong Buy recommendations
                  "buy": int,          # number of Buy recommendations
                  "hold": int,         # number of Hold recommendations
                  "sell": int,         # number of Sell recommendations
                  "strongSell": int    # number of Strong Sell recommendations
                }

            On validation error:
                {"error": "validation", "detail": "Symbol is required"}

            On no data:
                {"error": "no_data", "detail": "No recommendation data"}

            On API/HTTP error:
                Standardized error dict from _fh_get.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return {"error": "validation", "detail": "Symbol is required"}
    arr = await _fh_get("/stock/recommendation", {"symbol": s})
    if "error" in arr:
        return arr
    if not isinstance(arr, list) or not arr:
        return {"error": "no_data", "detail": "No recommendation data"}
    latest = arr[0]
    latest["symbol"] = s
    return latest


# 7) Recent news window (/company-news)
_NEWS_TTL_SEC = int(os.getenv("NEWS_TTL_SEC", "180"))
_NEWS_CACHE: dict[tuple[str, int, int], tuple[float, Dict[str, Any]]] = {}

_FINNHUB_NEWS_ID_RE = re.compile(
    r"^https?://(?:www\.)?finnhub\.io/api/news\?id=([A-Za-z0-9]+)$"
)


def _best_url(x: Dict[str, Any]) -> str | None:
    """
    Prefer the publisher URL. Some feeds/users accidentally pass the finnhub
    redirector (https://finnhub.io/api/news?id=...). If that happens, we still
    return it (it's clickable), but try publisher first.
    """
    url = (x.get("url") or "").strip() or None
    if url:
        if not _FINNHUB_NEWS_ID_RE.match(url):
            return url
    return url


def _normalize_items(raw: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """
    Normalize and de-duplicate raw news results.

    For each raw item, selects a best URL via ''_best_url'' and maps fields into a
    compact schema:
        {
            "headline": <str|None>,
            "title": <str>,          # headline > title > "Article"
            "url": <str|None>,       # result of _best_url(...)
            "source": <str|None>,
            "datetime": <int|str|None>
        }

    Then de-duplicates items by:
      1) canonicalized URL (case/space-insensitive), if present; otherwise
      2) canonicalized title.

    Finally, returns at most ''limit'' items.

    Args:
        raw: Iterable of dicts from the upstream API (Finnhub-ish shape).
        limit: Maximum number of normalized items to return (<= 0 → empty list).

    Returns:
        A de-duplicated list of normalized item dicts, truncated to ''limit''.
    """
    norm: List[Dict[str, Any]] = []
    for x in raw or []:
        u = _best_url(x)
        item = {
            "headline": x.get("headline"),
            "title": x.get("headline") or x.get("title") or "Article",
            "url": u,
            "source": x.get("source"),
            "datetime": x.get("datetime"),
        }
        norm.append(item)

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for it in norm:
        k_url = (it.get("url") or "").strip().lower()
        k_ttl = (it.get("title") or "").strip().lower()
        if k_url and k_url in seen_urls:
            continue
        if (not k_url) and k_ttl and k_ttl in seen_titles:
            continue
        if k_url:
            seen_urls.add(k_url)
        if k_ttl:
            seen_titles.add(k_ttl)
        deduped.append(it)

    return deduped[: max(0, int(limit))]


async def get_company_news(
    symbol: str, days: int = 7, limit: int = 10
) -> Dict[str, Any]:
    """
    Fetch recent company news for a ticker with local caching and retry.

    Behavior:
      - Validates inputs (non-empty symbol, positive ''days'').
      - Queries upstream (''/company-news'') over the date range
        ''[today - days, today]'' (inclusive).
      - On success, normalizes & de-dupes items via ''_normalize_items'' and
        returns a payload of the form:
          { "symbol": "<SYM>", "items": [...], "news": [...] }
        (''news'' mirrors ''items'' for downstream compatibility).
      - Results are cached in ''_NEWS_CACHE'' under key ''(symbol, days, limit)''
        for ''_NEWS_TTL_SEC'' seconds.
      - Retries up to 3 times with small jittered delays on exceptions.
      - If the upstream returns an error dict, it is passed through unchanged.

    Args:
        symbol: Stock ticker (e.g., "AAPL"). Case-insensitive; normalized to upper.
        days: Lookback window in days (must be > 0). Computed from today's date.
        limit: Maximum number of items to return after normalization/deduping.

    Returns:
        Dict with either:
          - Success: {"symbol": str, "items": List[dict], "news": List[dict]}
          - Validation error: {"error": "validation", "detail": str}
          - Upstream error passthrough: {"error": ..., "detail"?: ...}
          - Timeout after retries: {"error": "timeout", "detail": str}
    """
    s = (symbol or "").strip().upper()
    if not s:
        return {"error": "validation", "detail": "Symbol is required"}
    if days <= 0:
        return {"error": "validation", "detail": "days must be positive"}

    end = date.today()
    start = end - timedelta(days=days)

    key = (s, int(days), int(limit))
    import time

    now = time.time()
    hit = _NEWS_CACHE.get(key)
    if hit and (now - hit[0]) < _NEWS_TTL_SEC:
        return hit[1]

    max_attempts = 3
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            data = await _fh_get(
                "/company-news",
                {"symbol": s, "from": start.isoformat(), "to": end.isoformat()},
            )
            if isinstance(data, dict) and "error" in data:
                return data

            items = _normalize_items(list(data or []), limit)
            payload = {"symbol": s, "items": items, "news": items}
            _NEWS_CACHE[key] = (now, payload)
            return payload
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.25 + random.random() * 0.5)

    return {"error": "timeout", "detail": "Upstream did not respond"}


# ---------------------------------------------------------------------
# Financial calculation helpers
# ---------------------------------------------------------------------


def calculate_simple_interest(
    principal: float,
    rate_percent: float,
    years: float,
    inflation_rate_percent: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate simple interest and adjust for inflation (optional).

    This function computes the simple interest on an investment or loan using:
        Interest = Principal × (Rate/100) × Years

    It also adjusts the total amount for inflation if an inflation rate is provided,
    giving the real value of the investment in today's purchasing power.

    Args:
        principal (float): Initial amount of money (must be > 0).
        rate_percent (float): Annual interest rate in percent (must be >= 0).
        years (float): Time period in years (must be > 0).
        inflation_rate_percent (float, optional): Annual inflation rate in percent.
            Defaults to 0.0 (no inflation adjustment). Must be >= 0.

    Returns:
        Dict[str, Any]:
            {
              "principal": float,               # initial principal
              "rate": float,                    # interest rate (%)
              "years": float,                   # time period in years
              "interest": float,                # total interest earned
              "total_amount": float,            # principal + interest
              "inflation_rate": float,          # inflation rate (%)
              "real_value": float,              # total adjusted for inflation
              "purchasing_power_loss": float    # difference between nominal total and real value
            }

    Raises:
        ValueError: If principal <= 0, rate < 0, years <= 0, or inflation_rate < 0.
    """
    if principal <= 0:
        raise ValueError("Principal must be positive")
    if rate_percent < 0:
        raise ValueError("Interest rate cannot be negative")
    if years <= 0:
        raise ValueError("Time period must be positive")
    if inflation_rate_percent < 0:
        raise ValueError("Inflation rate cannot be negative")

    r = rate_percent / 100.0
    inf = inflation_rate_percent / 100.0

    interest = principal * r * years
    total = principal + interest
    real_value = total / ((1 + inf) ** years) if inflation_rate_percent > 0 else total
    purchasing_power_loss = (total - real_value) if inflation_rate_percent > 0 else 0.0

    return {
        "principal": principal,
        "rate": rate_percent,
        "years": years,
        "interest": interest,
        "total_amount": total,
        "inflation_rate": inflation_rate_percent,
        "real_value": real_value,
        "purchasing_power_loss": purchasing_power_loss,
    }


def calculate_compound_interest(
    principal: float,
    rate_percent: float,
    years: float,
    compounding_per_year: int = 1,
    inflation_rate_percent: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate compound interest and adjust for inflation (optional).

    This function computes compound interest using:
        Total = Principal × (1 + Rate/(100 × n))^(n × Years)

    where 'n' is the number of compounding periods per year.
    It also adjusts the total amount for inflation if an inflation rate is provided,
    giving the real value of the investment in today's purchasing power.

    Args:
        principal (float): Initial amount of money (must be > 0).
        rate_percent (float): Annual interest rate in percent (must be >= 0).
        years (float): Time period in years (must be > 0).
        compounding_per_year (int, optional): Number of times interest is compounded
            per year (e.g., 1 = yearly, 4 = quarterly, 12 = monthly). Default is 1.
        inflation_rate_percent (float, optional): Annual inflation rate in percent.
            Defaults to 0.0 (no inflation adjustment). Must be >= 0.

    Returns:
        Dict[str, Any]:
            {
              "principal": float,               # initial principal
              "rate": float,                    # interest rate (%)
              "years": float,                   # time period in years
              "compounding_frequency": int,     # compounding per year (n)
              "interest": float,                # total interest earned
              "total_amount": float,            # principal + interest
              "inflation_rate": float,          # inflation rate (%)
              "real_value": float,              # total adjusted for inflation
              "purchasing_power_loss": float    # difference between nominal total and real value
            }

    Raises:
        ValueError: If principal <= 0, rate < 0, years <= 0,
                    compounding_per_year <= 0, or inflation_rate < 0.

    """
    if principal <= 0:
        raise ValueError("Principal must be positive")
    if rate_percent < 0:
        raise ValueError("Interest rate cannot be negative")
    if years <= 0:
        raise ValueError("Time period must be positive")
    if compounding_per_year <= 0:
        raise ValueError("Compounding frequency must be positive")
    if inflation_rate_percent < 0:
        raise ValueError("Inflation rate cannot be negative")

    r = rate_percent / 100.0
    inf = inflation_rate_percent / 100.0
    n = float(compounding_per_year)

    total = principal * ((1 + r / n) ** (n * years))
    interest = total - principal
    real_value = total / ((1 + inf) ** years) if inflation_rate_percent > 0 else total
    purchasing_power_loss = (total - real_value) if inflation_rate_percent > 0 else 0.0

    return {
        "principal": principal,
        "rate": rate_percent,
        "years": years,
        "compounding_frequency": compounding_per_year,
        "interest": interest,
        "total_amount": total,
        "inflation_rate": inflation_rate_percent,
        "real_value": real_value,
        "purchasing_power_loss": purchasing_power_loss,
    }


def calculate_investment_return(
    principal: float,
    rate_percent: float,
    years: float,
    *,
    compounds_per_year: int = 12,
    contribution_per_period: float = 0.0,
    contribution_frequency_per_year: int = 12,
    contribution_timing: Literal["end", "begin"] = "end",
    inflation_rate_percent: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate the future value of an investment with optional recurring contributions
    and inflation adjustment.

    Args:
        principal (float): Initial investment amount (must be >= 0).
        rate_percent (float): Annual interest rate in percent (must be >= 0).
        years (float): Investment duration in years (must be > 0).
        compounds_per_year (int, optional): Number of compounding periods per year
            (e.g., 12 = monthly, 4 = quarterly). Default is 12.
        contribution_per_period (float, optional): Contribution amount per period.
            Must be >= 0. Default is 0.0 (no contributions).
        contribution_frequency_per_year (int, optional): Number of contribution
            periods per year (e.g., 12 = monthly, 1 = yearly). Default is 12.
        contribution_timing (Literal["end", "begin"], optional): Whether contributions
            are made at the "end" (ordinary annuity) or "begin" (annuity due) of each
            contribution period. Default is "end".
        inflation_rate_percent (float, optional): Annual inflation rate in percent.
            Defaults to 0.0 (no inflation adjustment). Must be >= 0.

    Returns:
        Dict[str, Any]:
            {
              "principal": float,                      # initial principal
              "rate": float,                           # annual interest rate (%)
              "years": float,                          # investment horizon
              "compounds_per_year": int,               # compounding frequency
              "contribution_per_period": float,        # contribution per period
              "contribution_frequency_per_year": int,  # contribution frequency
              "contribution_timing": str,              # "end" or "begin"
              "number_of_contributions": int,          # total number of deposits
              "total_contributions": float,            # sum of all contributions
              "total_invested": float,                 # principal + contributions
              "future_value": float,                   # projected value at end of period
              "total_return": float,                   # growth beyond invested capital
              "return_percentage": float,              # % return relative to invested
              "inflation_rate": float,                 # inflation rate (%)
              "real_value": float,                     # inflation-adjusted future value
              "purchasing_power_loss": float           # difference between nominal FV and real value
            }

    Raises:
        ValueError: If any of the following are invalid:
            - principal <= 0
            - rate_percent < 0
            - years <= 0
            - contribution_per_period < 0
            - compounds_per_year <= 0
            - contribution_frequency_per_year <= 0
            - inflation_rate_percent < 0
            - contribution_timing not in {"end", "begin"}
        - contribution_growth_mode = "effective_from_annual" uses i_eff = (1+r)^(1/k) - 1
        for contributions (k = contribution_frequency_per_year). This matches the
        interpretation “compounded annually with monthly deposits” producing 88,092.21
        in your example.
      - contribution_growth_mode = "align_with_compounding" keeps prior behavior,
        growing each deposit using the compounding periods (m = compounds_per_year).
    """

    if principal <= 0:
        raise ValueError("Initial investment must be positive")
    if rate_percent < 0:
        raise ValueError("Interest rate cannot be negative")
    if years <= 0:
        raise ValueError("Investment duration must be positive")
    if contribution_per_period < 0:
        raise ValueError("Contribution amount cannot be negative")
    if compounds_per_year <= 0:
        raise ValueError("Compounding frequency must be positive")
    if contribution_frequency_per_year <= 0:
        raise ValueError("Contribution frequency must be positive")
    if inflation_rate_percent < 0:
        raise ValueError("Inflation rate cannot be negative")
    if contribution_timing not in ("end", "begin"):
        raise ValueError("contribution_timing must be 'end' or 'begin'")

    r_annual = rate_percent / 100.0
    m = float(compounds_per_year)
    k = float(contribution_frequency_per_year)
    g_comp = 1.0 + (r_annual / m)
    fv_principal = principal * (g_comp ** (m * years)) if principal > 0 else 0.0
    j = (g_comp ** (m / k)) - 1
    n = int(round(years * k))
    fv_contrib = 0.0
    total_contrib = 0.0
    if contribution_per_period > 0 and n > 0:
        if j == 0.0:
            fv_contrib = contribution_per_period * n
        else:
            fv_contrib = contribution_per_period * (((1.0 + j) ** n - 1.0) / j)
            if contribution_timing == "begin":
                fv_contrib *= 1.0 + j
        total_contrib = contribution_per_period * n
    fv = fv_principal + fv_contrib
    total_invested = principal + total_contrib
    total_return = fv - total_invested
    return_pct = (total_return / total_invested * 100.0) if total_invested > 0 else 0.0

    inf = inflation_rate_percent / 100.0
    if inf > 0:
        real_value = fv / ((1 + inf) ** years)
        purchasing_power_loss = fv - real_value
    else:
        real_value = fv
        purchasing_power_loss = 0.0

    return {
        "principal": principal,
        "rate": rate_percent,
        "years": years,
        "compounds_per_year": int(m),
        "contribution_per_period": contribution_per_period,
        "contribution_frequency_per_year": int(k),
        "contribution_timing": contribution_timing,
        "number_of_contributions": n,
        "total_contributions": total_contrib,
        "total_invested": total_invested,
        "future_value": fv,
        "total_return": total_return,
        "return_percentage": return_pct,
        "inflation_rate": inflation_rate_percent,
        "real_value": real_value,
        "purchasing_power_loss": purchasing_power_loss,
        "effective_periodic_rate": j,
    }


def _normalize_freq_key(name: str) -> str:
    """
    Normalize a frequency string into a canonical key for lookup.

    This helper cleans and standardizes user-provided frequency names so they
    can be matched against known entries in FREQUENCY_MAP or synonyms in _SYNONYMS.

    Normalization steps:
      - Convert to lowercase.
      - Strip leading/trailing whitespace.
      - Remove spaces, hyphens, and underscores to form a compact key.
      - Check for matches in _SYNONYMS (variant → canonical).
      - Check for matches in FREQUENCY_MAP using compact or raw form.

    Args:
        name (str): Frequency string to normalize (e.g., "Monthly", "month-ly", "WEEKLY").

    Returns:
        str: Canonical frequency key if recognized, otherwise "" (empty string).
    """
    raw = (name or "").strip().lower()
    compact = raw.replace("-", "").replace("_", "").replace(" ", "")
    if compact in _SYNONYMS:
        return _SYNONYMS[compact]
    if compact in FREQUENCY_MAP:
        return compact
    if raw in FREQUENCY_MAP:
        return raw
    return ""


def _validate_and_get_frequency(name: str, field: str) -> int:
    """
    Validate a frequency string and return its numeric equivalent.

    This function ensures that a provided frequency label (e.g., "monthly",
    "bi-weekly", "semi-annual") is recognized, and maps it to the corresponding
    integer value from FREQUENCY_MAP (e.g., 12 for monthly, 26 for bi-weekly).

    Args:
        name (str): Frequency string to validate (e.g., "monthly", "weekly").
        field (str): Field name for context in error messages (e.g., "contribution frequency").

    Returns:
        int: The numeric frequency value associated with the given frequency
             (e.g., number of periods per year).

    Raises:
        ValueError: If the frequency is not recognized. The error message will list
                    allowed values and accepted synonyms.

    """
    key = _normalize_freq_key(name)
    if not key:
        allowed = ", ".join(sorted(FREQUENCY_MAP.keys()))
        raise ValueError(
            f"Unknown {field} '{name}'. Allowed values: {allowed}. "
            "Synonyms like 'semi-annual', 'bi-weekly', and 'twice a month' are also accepted."
        )
    return FREQUENCY_MAP[key]


def investment_return_from_strings(
    principal: float,
    rate_percent: float,
    years: float,
    *,
    compound: str = "monthly",
    regular_addition: float = 0.0,
    regular_addition_every: str = "monthly",
    addition_timing: Literal["end", "begin"] = "end",
    inflation_rate_percent: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate investment returns using human-readable frequency strings.

    This function is a wrapper around 'calculate_investment_return' that accepts
    compounding and contribution frequencies as strings (e.g., "monthly", "weekly",
    "semi-annual") instead of numeric values. Frequencies are validated and converted
    to their numeric equivalents via '_validate_and_get_frequency'.

    Args:
        principal (float): Initial investment amount (must be >= 0).
        rate_percent (float): Annual interest rate in percent (must be >= 0).
        years (float): Investment duration in years (must be > 0).
        compound (str, optional): Compounding frequency as a string
            (e.g., "annual", "quarterly", "monthly", "weekly").
            Default is "monthly".
        regular_addition (float, optional): Contribution amount per period (must be >= 0).
            Default is 0.0 (no contributions).
        regular_addition_every (str, optional): Frequency of contributions as a string
            (e.g., "monthly", "bi-weekly"). Default is "monthly".
        addition_timing (Literal["end", "begin"], optional): Whether contributions are
            made at the "end" (ordinary annuity) or "begin" (annuity due) of each period.
            Default is "end".
        inflation_rate_percent (float, optional): Annual inflation rate in percent.
            Default is 0.0 (no inflation adjustment). Must be >= 0.

    Returns:
        Dict[str, Any]: Same structure as 'calculate_investment_return', including:
            {
              "principal": float,
              "rate": float,
              "years": float,
              "compounds_per_year": int,
              "contribution_per_period": float,
              "contribution_frequency_per_year": int,
              "contribution_timing": str,
              "number_of_contributions": int,
              "total_contributions": float,
              "total_invested": float,
              "future_value": float,
              "total_return": float,
              "return_percentage": float,
              "inflation_rate": float,
              "real_value": float,
              "purchasing_power_loss": float
            }

    Raises:
        ValueError: If any frequency string is invalid, or if parameters
                    violate constraints enforced by 'calculate_investment_return'.
    """
    m = _validate_and_get_frequency(compound, "compound frequency")
    k = _validate_and_get_frequency(regular_addition_every, "contribution frequency")

    return calculate_investment_return(
        principal=principal,
        rate_percent=rate_percent,
        years=years,
        compounds_per_year=m,
        contribution_per_period=regular_addition,
        contribution_frequency_per_year=k,
        contribution_timing=addition_timing,
        inflation_rate_percent=inflation_rate_percent,
    )


def calculate_loan_amortization(
    principal: Decimal,
    annual_rate_percent: float,
    years: int,
    payments_per_year: int = 12,
) -> Dict[str, Decimal]:
    """
    Calculate the fixed payment and total interest for a standard amortizing loan.

    This function computes the level payment (PMT) required to fully amortize a loan
    over its term using the standard annuity formula. It also calculates the total
    interest paid over the life of the loan. If the annual interest rate is zero,
    payments are simply principal divided evenly across all periods.

    Args:
        principal (Decimal): Loan principal amount (must be > 0).
        annual_rate_percent (float): Annual nominal interest rate as a percentage
            (must be >= 0).
        years (int): Loan duration in years (must be > 0).
        payments_per_year (int, optional): Number of payments per year
            (e.g., 12 = monthly, 26 = bi-weekly). Default is 12.

    Returns:
        Dict[str, Decimal]:
            {
              "payment": Decimal,         # fixed periodic payment amount
              "total_interest": Decimal   # total interest paid over loan term
            }

    Raises:
        ValueError: If any input is invalid (principal <= 0, rate < 0, years <= 0,
                    or payments_per_year <= 0).

    """
    if principal <= 0:
        raise ValueError("Loan principal must be positive")
    if annual_rate_percent < 0:
        raise ValueError("Interest rate cannot be negative")
    if years <= 0:
        raise ValueError("Loan duration must be positive")
    if payments_per_year <= 0:
        raise ValueError("Payment frequency must be positive")

    n = years * payments_per_year
    r = Decimal(annual_rate_percent) / Decimal("100") / Decimal(payments_per_year)
    if r == 0:
        return {
            "payment": _money(principal / Decimal(n)),
            "total_interest": _money(Decimal("0")),
        }
    one_plus_r_pow_n = (Decimal("1") + r) ** n
    pmt = principal * (r * one_plus_r_pow_n) / (one_plus_r_pow_n - Decimal("1"))
    total_interest = (pmt * n) - principal
    return {"payment": _money(pmt), "total_interest": _money(total_interest)}


def _money(x: Decimal) -> Decimal:
    """Quantize a Decimal to cents using half-up rounding"""
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def npv(rate_percent_per_period: float, cashflows: Sequence[Decimal]) -> Decimal:
    """
    Calculate the Net Present Value (NPV) of a series of cash flows.

    NPV discounts each cash flow to its present value using the formula:
        NPV = Σ [ CF_t / (1 + r)^t ]
    where:
        CF_t = cash flow at time t
        r    = discount rate per period (decimal)
        t    = time index (0 = present)

    Args:
        rate_percent_per_period (float): Discount rate per period, in percent
            (e.g., 10 means 10% per period).
        cashflows (Sequence[Decimal]): Cash flow amounts by period, where index 0
            is time 0, index 1 is the next period, and so on. Can include positive
            inflows and negative outflows.

    Returns:
        Decimal: The net present value of the cash flows, rounded/formatted
        via '_money(...)'.
    """
    if rate_percent_per_period < 0:
        raise ValueError("Rate percent cannot be negative")
    r = Decimal(rate_percent_per_period) / Decimal("100")
    total = Decimal("0")
    one = Decimal("1")
    for t, cf in enumerate(cashflows):
        total += cf / ((one + r) ** t)
    return _money(total)


def cagr(start_value: float, end_value: float, years: float) -> Dict[str, Any]:
    """
    Calculate the Compound Annual Growth Rate (CAGR).

    CAGR is the constant annual growth rate that would take an investment from
    its beginning value to its ending value over a specified number of years,
    assuming compounding once per year.

    Formula:
        CAGR = (end_value / start_value)^(1 / years) - 1

    Args:
        start_value (float): Beginning value of the investment (must be > 0).
        end_value (float): Ending value of the investment (must be > 0).
        years (float): Number of years between start and end (must be > 0).

    Returns:
        {"type":"cagr","start":...,"end":...,"years":...,"cagr": <percent float>}
    Raises:
        ValueError: If start_value, end_value, or years are not positive.
    """
    if start_value <= 0 or end_value <= 0 or years <= 0:
        raise ValueError("start, end, years must be positive")
    cagr_decimal = (end_value / start_value) ** (1.0 / years) - 1.0
    return {
        "type": "cagr",
        "start": start_value,
        "end": end_value,
        "years": years,
        "cagr": cagr_decimal,
    }


def as_tool_source(name: str, args: dict, summary: str):
    """
    Normalized 'source' object front-end can show in SourcesDrawer.
    """
    return {
        "kind": "tool",
        "name": name,
        "args": args or {},
        "summary": summary,
        "id": f"tool:{name}",
        "url": None,
        "title": name,
    }
