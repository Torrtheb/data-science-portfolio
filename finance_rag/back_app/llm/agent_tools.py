from __future__ import annotations
import os
import re
import time
import asyncio
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Literal, Coroutine, TypeVar, Union

import httpx
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

from .tools import (
    calculate_simple_interest,
    calculate_compound_interest,
    calculate_investment_return,
    calculate_loan_amortization,
    investment_return_from_strings,
    cagr,
    npv,
    get_price,
    get_candles,
    search_symbol,
    get_profile,
    get_recommendation_trends,
    get_company_news,
)
from .rag import cached_doc_search_with_sources
from .worldbank import get_indicator


T = TypeVar("T")
FINNHUB_RPS = int(os.getenv("FINNHUB_RPS", "20"))
FINNHUB_PERIOD = float(os.getenv("FINNHUB_PERIOD", "1"))
EXTERNAL_TIMEOUT = float(os.getenv("EXTERNAL_TIMEOUT", "15"))
_WS = re.compile(r"\s+")
_FINNHUB_NEWS_ID = re.compile(
    r"https?://(?:www\.)?finnhub\.io/api/news\?id=([A-Za-z0-9]+)", re.I
)


# --- Rate limiting, upstream timeouts, and API key hygiene --------------
class RateLimiter:
    """Simple sliding-window rate limiter using an in-memory timestamp deque."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._calls = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            while self._calls and (now - self._calls[0]) > self.period:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                raise RuntimeError("rate_limited")
            self._calls.append(now)


def _sync_guard(*_args: Any, **_kwargs: Any) -> None:
    """Synchronous path should never be invoked for async tools; raises if called."""
    raise RuntimeError("Sync tool path used unexpectedly; tool must run via coroutine.")


def _tool_log(name: str, msg: str, **kv: Any) -> None:
    """Compact, structured tool log for observability."""
    kvs = " ".join(f"{k}={v}" for k, v in kv.items() if v is not None)
    logger.info(f"[TOOL] {name}: {msg}" + (f" | {kvs}" if kvs else ""))


_finnhub_limiter = RateLimiter(FINNHUB_RPS, FINNHUB_PERIOD)


async def _guarded_call_dict(
    op_name: str, coro: Coroutine[Any, Any, Any]
) -> dict | list:
    """
    Run an async upstream call behind a rate limiter and timeout.
    Returns:
        - successful upstream result (dict or list) as-is
        - on failure: {"error": <str>, "detail": <str?>}
    """
    try:
        await _finnhub_limiter.acquire()
        result = await asyncio.wait_for(coro, timeout=EXTERNAL_TIMEOUT)
        return result
    except asyncio.TimeoutError:
        logger.warning(f"{op_name} timeout")
        return {"error": "timeout", "detail": "Upstream did not respond."}
    except RuntimeError as e:
        if str(e) == "rate_limited":
            logger.warning(f"{op_name} rate-limited")
            return {"error": "rate_limited", "detail": "Please try again in a moment."}
        logger.exception(f"{op_name} runtime error: {e!r}")
        return {"error": "unexpected"}
    except Exception as e:
        logger.exception(f"{op_name} crashed: {e!r}")
        return {"error": "unexpected"}


if not (
    os.getenv("OPENAI_API_KEY")
    or getattr(globals().get("settings", object()), "openai_api_key", None)
):
    logger.error("OPENAI_API_KEY is not set. LLM calls will fail.")

if not os.getenv("FINNHUB_API_KEY"):
    logger.warning("FINNHUB_API_KEY is not set. Market data tools will fail.")


# ---- Parsers for percent/money (validators for tool input models) ----


def _parse_percent_like(v: float | str) -> float:
    """
    Normalize a percent-like input into percentage points (as a float).

    Accepted inputs:
      - A string with a trailing '%' (e.g., "7%", "  0.5% ")
      - A numeric string without '%' (e.g., "0.07", "7")
      - A float (e.g., 0.07, 7.0)

    Rules:
      - If the input ends with '%', return the numeric part as-is in percentage points.
        e.g., "7%" -> 7.0, "-5%" -> -5.0
      - Otherwise, parse to float 'x'. If '0 <= x < 1', treat as a fraction and scale by 100.
        e.g., 0.07 -> 7.0, "0.07" -> 7.0
      - For values < 0 or >= 1, treat as already in percentage points and return as-is.
        e.g., 7 -> 7.0, 1.5 -> 1.5, -0.05 -> -0.05

    Args:
        v: Percent-like value to parse.

    Returns:
        float: Percentage points (e.g., 7.0 means 7%).

    Raises:
        ValueError: If the string cannot be parsed as a float.
    """
    if isinstance(v, str):
        s = v.strip().replace(" ", "")
        if s.endswith("%"):
            return float(s[:-1])
        x = float(s)
    else:
        x = float(v)
    return x * 100.0 if 0 <= x < 1 else x


def _parse_money_like(v: float | str) -> float:
    """
    Parse a money-like value into a float number of currency units.

    Accepted inputs:
      - Strings with optional '$' and thousands separators (commas), e.g., "$1,234.56", "1234.56"
      - Floats/ints, which are returned as float

    Behavior:
      - Strips whitespace, removes '$' and ',' then parses as float.
      - Does not perform currency conversion or locale-aware parsing.

    Args:
        v: Money-like value to parse.

    Returns:
        float: The numeric amount, e.g., "$1,234.56" -> 1234.56.

    Raises:
        ValueError: If the string cannot be parsed as a float.
    """
    if isinstance(v, str):
        s = v.replace("$", "").replace(",", "").strip()
        return float(s)
    return float(v)


# ---------------- URL helpers -----------------


def _fmt_date(ts: Any) -> str:
    """Format UNIX ts -> YYYY-MM-DD (UTC) if numeric; else str(ts)."""
    try:
        if isinstance(ts, (int, float)) and ts > 0:
            return str(datetime.fromtimestamp(int(ts), tz=timezone.utc).date())
    except Exception:
        pass
    return str(ts) if ts else ""


def _clean_url(u: Optional[str]) -> str:
    """
    Sanitize URLs for storage (not necessarily for rendering):
    - collapse whitespace
    - add https:// for bare 'www.' links
    """
    if not u:
        return ""
    s = _WS.sub("", str(u))
    if not s:
        return ""
    if not re.match(r"^https?://", s, flags=re.I) and s.startswith("www."):
        s = "https://" + s
    return s


async def _resolve_publisher_url(
    raw: Optional[str], item_id: Optional[str] = None
) -> Optional[str]:
    """
    If Finnhub redirect (api/news?id=...), fetch to resolve publisher 'url'.
    Else returns cleaned raw URL. Returns None if nothing usable.
    """
    if not raw and not item_id:
        return None

    u = (raw or "").strip()
    m = _FINNHUB_NEWS_ID.match(u)
    if not m and u:
        return _clean_url(u) or None

    fid = item_id or (m.group(1) if m else None)
    if not fid:
        return _clean_url(u) or None

    params = {"id": fid}
    token = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_TOKEN") or ""
    if token:
        params["token"] = token

    try:
        async with httpx.AsyncClient(timeout=8) as h:
            r = await h.get("https://finnhub.io/api/news", params=params)
            r.raise_for_status()
            jd = r.json()
            item = (
                jd
                if isinstance(jd, dict)
                else (jd[0] if isinstance(jd, list) and jd else None)
            )
            url = (item or {}).get("url")
            return _clean_url(url) if url else None
    except Exception:
        return None


# ---------- Calculation tools input schemas ----------
class SimpleInterestInput(BaseModel):
    principal: float = Field(gt=0, description="Initial amount (> 0)")
    rate_percent: float | str = Field(
        ..., description="Annual simple rate; accepts 6, '6%', or 0.06"
    )
    years: float = Field(gt=0, description="Years (> 0, may be fractional)")
    inflation_rate_percent: Optional[float | str] = Field(
        0.0, ge=0, description="Annual inflation %"
    )

    @field_validator("principal", mode="before")
    def _v_principal(cls, v):
        return _parse_money_like(v)

    @field_validator("rate_percent", "inflation_rate_percent", mode="before")
    def _v_rate(cls, v):
        return _parse_percent_like(v)


class CompoundInterestInput(BaseModel):
    principal: float = Field(gt=0)
    rate_percent: float | str = Field(...)
    years: float = Field(gt=0)
    compounding_per_year: int = Field(1, gt=0, le=365)
    inflation_rate_percent: Optional[float | str] = Field(0.0, ge=0)

    @field_validator("principal", mode="before")
    def _v_principal(cls, v):
        return _parse_money_like(v)

    @field_validator("rate_percent", "inflation_rate_percent", mode="before")
    def _v_rate(cls, v):
        return _parse_percent_like(v)


class InvestmentReturnInput(BaseModel):
    principal: float = Field(ge=0)
    rate_percent: float | str = Field(ge=0)
    years: float = Field(gt=0)
    compounds_per_year: int = Field(12, gt=0, le=365)
    contribution_per_period: float = Field(0.0, ge=0)
    contribution_frequency_per_year: int = Field(12, gt=0, le=365)
    contribution_timing: Literal["end", "begin"] = "end"
    inflation_rate_percent: Optional[float | str] = Field(0.0, ge=0)

    @field_validator("principal", "contribution_per_period", mode="before")
    def _v_money(cls, v):
        return _parse_money_like(v)

    @field_validator("rate_percent", "inflation_rate_percent", mode="before")
    def _v_rate(cls, v):
        return _parse_percent_like(v)


class InvestmentReturnStringsInput(BaseModel):
    principal: float = Field(ge=0)
    rate_percent: float | str = Field(ge=0)
    years: float = Field(gt=0)
    compound: str = Field("monthly", description="e.g., monthly, quarterly, weekly")
    regular_addition: float = Field(0.0, ge=0)
    regular_addition_every: str = Field(
        "monthly", description="e.g., monthly, weekly, biweekly"
    )
    addition_timing: Literal["end", "begin"] = "end"
    inflation_rate_percent: Optional[float | str] = Field(0.0, ge=0)

    @field_validator("principal", "regular_addition", mode="before")
    def _v_money(cls, v):
        return _parse_money_like(v)

    @field_validator("rate_percent", "inflation_rate_percent", mode="before")
    def _v_rate(cls, v):
        return _parse_percent_like(v)


class LoanAmortizationInput(BaseModel):
    principal: float = Field(gt=0)
    annual_rate_percent: float | str = Field(ge=0)
    years: int = Field(gt=0)
    payments_per_year: int = Field(12, gt=0, le=365)

    @field_validator("principal", mode="before")
    def _v_principal(cls, v):
        return _parse_money_like(v)

    @field_validator("annual_rate_percent", mode="before")
    def _v_rate(cls, v):
        return _parse_percent_like(v)


class NPVInput(BaseModel):
    rate_percent_per_period: float | str = Field(
        ..., description="Discount rate per period (%, accepts '6%' or 0.06)"
    )
    cashflows: List[float] = Field(
        description="Cashflows t=0..N; negatives for investments"
    )

    @field_validator("rate_percent_per_period", mode="before")
    def _v_rate(cls, v):
        return _parse_percent_like(v)

    @field_validator("cashflows", mode="before")
    def _v_cfs(cls, v):
        return [_parse_money_like(x) for x in v]


class CAGRInput(BaseModel):
    start_value: float = Field(gt=0)
    end_value: float = Field(gt=0)
    years: float = Field(gt=0)

    @field_validator("start_value", "end_value", mode="before")
    def _v_money(cls, v):
        return _parse_money_like(v)


# ---------- Input Schemas for Finnhub API----------
class PriceInput(BaseModel):
    symbol: str = Field(min_length=1, description="Ticker, e.g., AAPL")


class CandlesInput(BaseModel):
    symbol: str = Field(min_length=1)
    resolution: Literal["1", "5", "15", "30", "60", "D", "W", "M"] = "D"
    from_ts: int = Field(gt=0, description="UNIX seconds (start, > 0)")
    to_ts: int = Field(gt=0, description="UNIX seconds (end; must be > from_ts)")

    @field_validator("symbol", mode="before")
    def _v_symbol(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("symbol is required")
        return s

    @field_validator("to_ts")
    def _v_time_order(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure to_ts > from_ts to avoid empty/invalid upstream windows."""
        start_raw = values.get("from_ts")
        try:
            start = int(start_raw) if start_raw is not None else None
            end = int(v)
            if start is not None and start > 0 and end > 0 and end <= start:
                raise ValueError("to_ts must be greater than from_ts")
        except Exception as exc:
            raise ValueError("invalid timestamp values") from exc
        return v


class SearchSymbolInput(BaseModel):
    query: str = Field(min_length=1, max_length=50)


class ProfileInput(BaseModel):
    symbol: str = Field(min_length=1)


class RecoTrendsInput(BaseModel):
    symbol: str = Field(min_length=1)


class CompanyNewsInput(BaseModel):
    symbol: str = Field(min_length=1)
    days: int = Field(default=7, gt=0, le=30)
    limit: int = Field(default=10, gt=0, le=50)


class DocumentSearchInput(BaseModel):
    query: str = Field(min_length=3, max_length=500)


# ---- async tool wrappers ------------------
def document_search_tool(query: str) -> dict:
    """Search the local finance KB and return markdown + structured 'source' payload."""
    q = query.strip()
    if len(q) < 3:
        return {
            "text": "❌ Please provide a more specific search query (at least 3 characters)."
        }
    md, sources = cached_doc_search_with_sources(q)
    return {
        "text": md,
        "source": {
            "kind": "tool",
            "name": "document_search",
            "args": {"query": q},
            "summary": "Local finance KB search with citations",
            "id": "tool:document_search",
            "title": "document_search",
            "url": None,
            "citations": sources,
        },
    }


async def _get_price_tool(symbol: str) -> str:
    """
    Fetch a live quote and render it as Markdown.

    Calls the upstream 'get_price(symbol)' coroutine and formats either a
    human-readable price block or a clear error line.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "TSLA", "SHOP.TO").

    Returns:
        Markdown string with a "📊 Live Price" header, showing last price, OHLC,
        previous close, and absolute/percent change. On failure, returns a line
        like: "❌ get_price error: **...** detail".
    """
    _tool_log("get_price", "start", symbol=symbol)
    res = await _guarded_call_dict("get_price", get_price(symbol))
    if isinstance(res, dict) and "error" in res:
        detail = res.get("detail") or res.get("note") or res.get("hint") or ""
        _tool_log("get_price", "error", symbol=symbol, detail=detail)
        return f"❌ get_price error: **{res.get('error','unknown')}** {detail}"

    if not isinstance(res, dict):
        return "❌ get_price: unexpected response."

    sym = res.get("symbol") or symbol.upper()
    price = res.get("price")
    pc = res.get("prev_close", res.get("prevClose"))
    o = res.get("open")
    h = res.get("high")
    l = res.get("low")
    ch = res.get("change")
    chp = res.get("change_pct", res.get("changePercent"))
    try:
        if (
            (ch is None or chp is None)
            and isinstance(price, (int, float))
            and isinstance(pc, (int, float))
            and pc
        ):
            ch = price - pc
            chp = (ch / pc) * 100.0 if pc else None
    except Exception:
        pass

    def fmt(x, p=2):
        try:
            return f"${float(x):,.{p}f}"
        except Exception:
            return "n/a"

    lines = ["## 📊 Live Price", f"**{sym}**: {fmt(price)}"]
    if any(v is not None for v in (o, h, l)):
        lines.append(f"- Open: {fmt(o)}  High: {fmt(h)}  Low: {fmt(l)}")
    if pc is not None:
        lines.append(f"- Prev Close: {fmt(pc)}")
    if ch is not None or chp is not None:
        ch_str = f"{float(ch):+,.2f}" if isinstance(ch, (int, float)) else ""
        chp_str = f"{float(chp):+,.2f}%" if isinstance(chp, (int, float)) else ""
        both = " ".join([s for s in [ch_str, f"({chp_str})" if chp_str else ""] if s])
        lines.append(f"- Change: {both}" if both else "- Change: n/a")
    _tool_log("get_price", "end", status="ok", price=price)
    return "\n".join(lines)


async def _search_symbol_tool(query: str) -> str:
    """
    Search for symbols by free-text query and render the top matches.

    Args:
        query: User input such as company name, partial ticker, or keyword.

    Returns:
        Markdown list beginning with "🔎 Symbol Search". If no matches are found,
        returns "No symbols found.".
    """
    _tool_log("search_symbol", "start", query=query)
    _items_res = await _guarded_call_dict("search_symbol", search_symbol(query))
    if isinstance(_items_res, dict) and "error" in _items_res:
        return f"❌ search_symbol error: **{_items_res['error']}** {_items_res.get('detail', '')}"

    items = []
    if isinstance(_items_res, list):
        items = _items_res
    elif isinstance(_items_res, dict) and isinstance(_items_res.get("results"), list):
        items = _items_res["results"]

    if not items:
        return "No symbols found."

    lines = ["## 🔎 Symbol Search"]
    for r in items[:10]:
        sym = r.get("symbol") or r.get("ticker") or ""
        desc = r.get("description") or r.get("name") or ""
        mic = r.get("mic") or r.get("exchange") or ""
        mic = f" ({mic})" if mic else ""
        lines.append(f"- **{sym}** — {desc}{mic}")
    _tool_log("search_symbol", "end", status="ok", results=len(items))
    return "\n".join(lines)


async def _get_profile_tool(symbol: str) -> str:
    """
    Fetch company profile metadata and render key attributes.

    Args:
        symbol: Ticker symbol.

    Returns:
        Markdown block titled "🪪 Company Profile" with name, exchange, currency,
        market cap, and IPO date; or an error line if the API call fails.
    """
    _tool_log("get_profile", "start", symbol=symbol)
    res = await _guarded_call_dict("get_profile", get_profile(symbol))
    if isinstance(res, dict) and "error" in res:
        _tool_log("get_profile", "error", symbol=symbol, detail=res.get("detail", ""))
        return f"❌ get_profile error: **{res['error']}** {res.get('detail','')}"
    if not isinstance(res, dict):
        return "❌ get_profile: unexpected response."
    mc = res.get("market_cap")
    mc_str = f"${mc:,.0f}" if mc is not None else "n/a"
    _tool_log("get_profile", "end", status="ok", market_cap=mc)
    return (
        "## 🪪 Company Profile\n"
        f"**{res.get('name') or res['symbol']}** ({res['symbol']})\n"
        f"- Exchange: {res.get('exchange') or 'n/a'}  Currency: {res.get('currency') or 'n/a'}\n"
        f"- Market Cap: {mc_str}\n"
        f"- IPO: {res.get('ipo') or 'n/a'}"
    )


async def _get_candles_tool(
    symbol: str, resolution: str, from_ts: int, to_ts: int
) -> str:
    """
    Fetch historical candles and summarize the response.

    Args:
        symbol: Ticker (e.g., "AAPL").
        resolution: One of {"1","5","15","30","60","D","W","M"}.
        from_ts: Start time (UNIX epoch seconds).
        to_ts: End time (UNIX epoch seconds).

    Returns:
        Markdown one-liner with a "🕯️ Candles" header indicating the number of
        bars returned; or an error line if the API call fails.
    """
    _tool_log(
        "get_candles",
        "start",
        symbol=symbol,
        resolution=resolution,
        from_ts=from_ts,
        to_ts=to_ts,
    )
    res = await _guarded_call_dict(
        "get_candles", get_candles(symbol, resolution, from_ts, to_ts)
    )
    if isinstance(res, dict) and "error" in res:
        _tool_log(
            "get_candles",
            "error",
            symbol=symbol,
            resolution=resolution,
            from_ts=from_ts,
            to_ts=to_ts,
            detail=res.get("detail", ""),
        )
        return f"❌ get_candles error: **{res['error']}** {res.get('note', res.get('detail',''))}"
    t = (res or {}).get("t") if isinstance(res, dict) else None
    n = len(t or [])
    _tool_log(
        "get_candles",
        "end",
        symbol=symbol,
        resolution=resolution,
        from_ts=from_ts,
        to_ts=to_ts,
        bars=n,
    )
    return f"## 🕯️ Candles\n**{symbol.upper()}** {resolution} — returned **{n}** bars."


async def _get_reco_trends_tool(symbol: str) -> str:
    """
    Fetch analyst recommendation trends and render the latest split.

    Args:
        symbol: Ticker symbol.

    Returns:
        Markdown block titled "🧭 Analyst Recommendation Trends" with counts for
        Strong Buy / Buy / Hold / Sell / Strong Sell; or an error line on failure.
    """
    _tool_log("get_recommendation_trends", "start", symbol=symbol)
    res = await _guarded_call_dict(
        "get_recommendation_trends", get_recommendation_trends(symbol)
    )
    if isinstance(res, dict) and "error" in res:
        _tool_log(
            "get_recommendation_trends",
            "error",
            symbol=symbol,
            detail=res.get("detail", ""),
        )
        return f"❌ recommendation error: **{res['error']}**"
    snap = None
    if isinstance(res, list) and res:
        snap = res[0]
    elif isinstance(res, dict):
        snap = res

    if not isinstance(snap, dict):
        return "No recommendation data."

    sym = snap.get("symbol", symbol.upper())
    period = snap.get("period", "latest")
    sb = snap.get("strongBuy", 0)
    b = snap.get("buy", 0)
    h = snap.get("hold", 0)
    s = snap.get("sell", 0)
    ss = snap.get("strongSell", 0)
    _tool_log(
        "get_recommendation_trends",
        "end",
        symbol=sym,
        period=period,
        strong_buy=sb,
        buy=b,
        hold=h,
        sell=s,
        strong_sell=ss,
    )
    return (
        "## 🧭 Analyst Recommendation Trends\n"
        f"**{sym}** for period **{period}**\n"
        f"- Strong Buy: {sb}, Buy: {b}, Hold: {h}, Sell: {s}, Strong Sell: {ss}"
    )


async def _get_company_news_tool(
    symbol: str, days: int = 7, limit: int = 10
) -> Dict[str, Any]:
    """
    Fetch recent company news and return a JSON-serializable dict:
      {"markdown": <str>, "news": [items...]}

    Notes:
      - We resolve Finnhub redirect URLs to publisher URLs (stored in data),
        but we do not render clickable links in the markdown (chat-safe).
    """
    _tool_log("get_company_news", "start", symbol=symbol, days=days, limit=limit)
    res = await _guarded_call_dict(
        "get_company_news", get_company_news(symbol, days, limit)
    )
    if isinstance(res, dict) and "error" in res:
        return {
            "markdown": f"❌ news error: **{res['error']}** {res.get('detail','')}",
            "news": [],
        }
    if not isinstance(res, (dict, list)):
        return {"markdown": "❌ news: unexpected response.", "news": []}

    items = []
    if isinstance(res, dict):
        items = res.get("news") or res.get("items") or []
    if not items:
        return {"markdown": f"No recent news for {symbol.upper()}.", "news": []}

    async def enrich(it: dict) -> dict:
        title = (it.get("title") or it.get("headline") or it.get("name") or "").strip()
        src = (it.get("source") or it.get("provider") or "").strip()
        raw_url = (
            it.get("url") or it.get("link") or it.get("source_url") or ""
        ).strip()
        fid = (it.get("id") or it.get("news_id")) or (
            m.group(1) if (m := _FINNHUB_NEWS_ID.match(raw_url)) else ""
        )
        pub_url = await _resolve_publisher_url(raw_url, item_id=fid)

        it["id"] = fid
        it["publisher_url"] = _clean_url(pub_url)
        it["url"] = _clean_url(raw_url)
        it["href"] = _clean_url(pub_url or raw_url)
        it["title"] = title
        it["source"] = src
        return it

    items = await asyncio.gather(*(enrich(it) for it in items[:limit]))

    lines = [f"## 🗞️ News — {symbol.upper()} (last {days}d)"]
    for it in items:
        title = (it.get("title") or it.get("headline") or "(no headline)").strip()
        src = (it.get("source") or it.get("provider") or "").strip()
        when_ = _fmt_date(it.get("datetime"))
        meta = (f" — *{src}*" if src else "") + (f" · {when_}" if when_ else "")
        lines.append(f"- {title}{meta}")

    md = "\n".join(lines)
    _tool_log("get_company_news", "end", symbol=symbol, articles=len(items))
    return {"markdown": md, "news": items}


# ---- calculator wrappers ---------------------


def simple_interest_tool(
    principal: float,
    rate_percent: float | str,
    years: float,
    inflation_rate_percent: float | str = 0.0,
) -> str:
    """Markdown summary for simple interest."""
    r = calculate_simple_interest(
        float(principal),
        float(_parse_percent_like(rate_percent)),
        float(years),
        float(_parse_percent_like(inflation_rate_percent)),
    )
    out = [
        "## 💰 Simple Interest",
        f"**Principal**: ${r['principal']:,.2f}",
        f"**Rate**: {r['rate']}% ",
        f"**Years**: {r['years']}",
        "### Results",
        f"- **Interest**: ${r['interest']:,.2f}",
        f"- **Total**: ${r['total_amount']:,.2f}",
    ]
    if float(_parse_percent_like(inflation_rate_percent)) > 0:
        out.append(f"- **Inflation-Adjusted (real)**: ${r['real_value']:,.2f}")
        out.append(f"- **Purchasing Power Loss**: ${r['purchasing_power_loss']:,.2f}")
    return "\n".join(out)


def compound_interest_tool(
    principal: float,
    rate_percent: float | str,
    years: float,
    compounding_per_year: int = 1,
    inflation_rate_percent: float | str = 0.0,
) -> str:
    """Markdown summary for compound interest."""
    r = calculate_compound_interest(
        float(principal),
        float(_parse_percent_like(rate_percent)),
        float(years),
        int(compounding_per_year),
        float(_parse_percent_like(inflation_rate_percent)),
    )
    out = [
        "## Compound Interest",
        f"**Principal**: ${r['principal']:,.2f}",
        f"**Rate**: {r['rate']}% ",
        f"**Years**: {r['years']}",
        "### Results",
        f"- **Interest**: ${r['interest']:,.2f}",
        f"- **Total**: ${r['total_amount']:,.2f}",
    ]
    if float(_parse_percent_like(inflation_rate_percent)) > 0:
        out.append(f"- **Inflation-Adjusted (real)**: ${r['real_value']:,.2f}")
        out.append(f"- **Purchasing Power Loss**: ${r['purchasing_power_loss']:,.2f}")
    return "\n".join(out)


def investment_return_tool(
    principal: float,
    rate_percent: float | str,
    years: float,
    compounds_per_year: int = 12,
    contribution_per_period: float = 0.0,
    contribution_frequency_per_year: int = 12,
    contribution_timing: str = "end",
    inflation_rate_percent: float | str = 0.0,
    _question: str | None = None,
) -> str:
    """
    Markdown summary for investment FV with separate compounding & contributions.
    Uses inference helpers to choose compounding, contribution frequency,
    and timing from the original user question. Defaults to ANNUAL compounding.
    """
    rate_pct = float(_parse_percent_like(rate_percent))
    infl_pct = float(_parse_percent_like(inflation_rate_percent))
    r = calculate_investment_return(
        principal=float(principal),
        rate_percent=rate_pct,
        years=float(years),
        compounds_per_year=int(compounds_per_year),
        contribution_per_period=float(contribution_per_period),
        contribution_frequency_per_year=int(contribution_frequency_per_year),
        contribution_timing=contribution_timing,
        inflation_rate_percent=infl_pct,
    )
    out = [
        "## 📈 Investment Projection",
        f"**Principal**: ${r['principal']:,.2f}",
        f"**Rate**: {r['rate']:.4g}%  ·  **Compounding**: {r['compounds_per_year']}×/yr",
        f"**Years**: {r['years']}",
        f"**Contribution**: ${r['contribution_per_period']:,.2f} × {r['contribution_frequency_per_year']}/yr ({r['contribution_timing']})",
        f"**# Contributions**: {r['number_of_contributions']}",
        "### Results",
        f"- **Future Value**: ${r['future_value']:,.2f}",
        f"- **Total Invested**: ${r['total_invested']:,.2f}",
        f"- **Total Return**: ${r['total_return']:,.2f} ({r['return_percentage']:.2f}%)",
    ]
    if infl_pct > 0:
        out.append(f"- **Inflation-Adjusted (real)**: ${r['real_value']:,.2f}")
        out.append(f"- **Purchasing Power Loss**: ${r['purchasing_power_loss']:,.2f}")

    return "\n".join(out)


def investment_return_strings_tool(
    principal: float,
    rate_percent: float | str,
    years: float,
    compound: str = "monthly",
    regular_addition: float = 0.0,
    regular_addition_every: str = "monthly",
    addition_timing: str = "end",
    inflation_rate_percent: float | str = 0.0,
) -> str:
    """Markdown summary using calculator-style strings (e.g., 'monthly', 'biweekly')."""
    r = investment_return_from_strings(
        principal=float(principal),
        rate_percent=float(_parse_percent_like(rate_percent)),
        years=float(years),
        compound=compound,
        regular_addition=float(regular_addition),
        regular_addition_every=regular_addition_every,
        addition_timing=addition_timing,
        inflation_rate_percent=float(_parse_percent_like(inflation_rate_percent)),
    )
    out = [
        "## Investment Return (string inputs)",
        f"**Initial**: ${r['principal']:,.2f}",
        f"**Rate**: {r['rate']}% ",
        f"**Years**: {r['years']}",
        f"**Compounds/yr**: {r['compounds_per_year']}",
        f"**Contrib/period**: ${r['contribution_per_period']:,.2f}",
        f"**Contrib freq/yr**: {r['contribution_frequency_per_year']} ({r['contribution_timing']})",
        f"**# Contributions**: {r['number_of_contributions']}",
        "### Results",
        f"- **Future Value**: ${r['future_value']:,.2f}",
        f"- **Total Invested**: ${r['total_invested']:,.2f}",
        f"- **Total Return**: ${r['total_return']:,.2f} ({r['return_percentage']:.2f}%)",
    ]
    if float(_parse_percent_like(inflation_rate_percent)) > 0:
        out.append(f"- **Inflation-Adjusted (real)**: ${r['real_value']:,.2f}")
        out.append(f"- **Purchasing Power Loss**: ${r['purchasing_power_loss']:,.2f}")
    return "\n".join(out)


def loan_amortization_tool(
    principal: float,
    annual_rate_percent: float | str,
    years: int,
    payments_per_year: int = 12,
) -> str:
    """Markdown summary for level-payment loan amortization."""
    res = calculate_loan_amortization(
        Decimal(str(principal)),
        float(_parse_percent_like(annual_rate_percent)),
        int(years),
        int(payments_per_year),
    )
    payment = res["payment"]
    total_interest = res["total_interest"]
    total_paid = payment * (years * payments_per_year)
    out = [
        "## Loan Amortization",
        f"**Amount**: ${float(principal):,.2f}",
        f"**Rate**: {float(_parse_percent_like(annual_rate_percent))}% ",
        f"**Years**: {years}",
        f"**Payments/yr**: {payments_per_year}",
        "### Results",
        f"- **Payment/period**: ${payment:,.2f}",
        f"- **Total Interest**: ${total_interest:,.2f}",
        f"- **Total Paid**: ${total_paid:,.2f}",
    ]
    return "\n".join(out)


def npv_tool(rate_percent_per_period: float | str, cashflows: List[float]) -> str:
    """Markdown summary for NPV."""
    cf_dec = [Decimal(str(x)) for x in cashflows]
    pv = npv(float(_parse_percent_like(rate_percent_per_period)), cf_dec)
    return (
        "## 📉 Net Present Value (NPV)\n"
        f"- **Rate/period**: {float(_parse_percent_like(rate_percent_per_period))}%\n"
        f"- **Cashflows**: {', '.join(f'${x:,.2f}' for x in cashflows)}\n"
        f"- **NPV**: ${pv:,.2f}"
    )


def cagr_tool(start_value: float, end_value: float, years: float) -> Dict[str, Any]:
    """
    Compute CAGR; return a dict containing:
      - cagr_decimal (fraction), cagr_percent (0..100),
      - inputs (start, end, years), and a markdown summary block.
    """
    res = cagr(float(start_value), float(end_value), float(years))

    if isinstance(res, dict):
        rate = res.get("cagr")
        if rate is None:
            rate = res.get("cagr_percent")
            if rate is not None:
                return {
                    "tool": "calculate_cagr",
                    "start": float(start_value),
                    "end": float(end_value),
                    "years": float(years),
                    "cagr_decimal": float(rate) / 100.0,
                    "cagr_percent": float(rate),
                    "markdown": (
                        "## 📈 CAGR\n"
                        f"- **Start**: ${float(start_value):,.2f} → **End**: ${float(end_value):,.2f}\n"
                        f"- **Years**: {float(years):g}\n"
                        f"- **CAGR**: {float(rate):.2f}%"
                    ),
                }
        else:
            cagr_percent = float(rate)
            cagr_decimal = cagr_percent / 100.0
            return {
                "tool": "calculate_cagr",
                "start": float(start_value),
                "end": float(end_value),
                "years": float(years),
                "cagr_decimal": cagr_decimal,
                "cagr_percent": cagr_percent,
                "markdown": (
                    "## 📈 CAGR\n"
                    f"- **Start**: ${float(start_value):,.2f} → **End**: ${float(end_value):,.2f}\n"
                    f"- **Years**: {float(years):g}\n"
                    f"- **CAGR**: {cagr_percent:.2f}%"
                ),
            }

    cagr_percent = float(res)
    cagr_decimal = cagr_percent / 100.0
    return {
        "tool": "calculate_cagr",
        "start": float(start_value),
        "end": float(end_value),
        "years": float(years),
        "cagr_decimal": cagr_decimal,
        "cagr_percent": cagr_percent,
        "markdown": (
            "## 📈 CAGR\n"
            f"- **Start**: ${float(start_value):,.2f} → **End**: ${float(end_value):,.2f}\n"
            f"- **Years**: {float(years):g}\n"
            f"- **CAGR**: {cagr_percent:.2f}%"
        ),
    }


# ------------- Register tools -----------------

FINNHUB_TOOLS = [
    StructuredTool.from_function(
        func=_sync_guard,
        coroutine=_get_price_tool,
        name="get_live_price",
        description="Live price snapshot for a ticker, e.g., 'AAPL'.",
        args_schema=PriceInput,
    ),
    StructuredTool.from_function(
        func=_sync_guard,
        coroutine=_search_symbol_tool,
        name="search_symbol",
        description="Search for tickers by company name or keyword.",
        args_schema=SearchSymbolInput,
    ),
    StructuredTool.from_function(
        func=_sync_guard,
        coroutine=_get_profile_tool,
        name="get_company_profile",
        description="Basic company profile: name, exchange, currency, market cap, IPO.",
        args_schema=ProfileInput,
    ),
    StructuredTool.from_function(
        func=_sync_guard,
        coroutine=_get_candles_tool,
        name="get_candles",
        description="Fetch OHLCV candles for a symbol between UNIX timestamps; resolution in {1,5,15,30,60,D,W,M}.",
        args_schema=CandlesInput,
    ),
    StructuredTool.from_function(
        func=_sync_guard,
        coroutine=_get_reco_trends_tool,
        name="get_recommendation_trends",
        description="Latest analyst recommendation snapshot for a symbol.",
        args_schema=RecoTrendsInput,
    ),
    StructuredTool.from_function(
        func=_sync_guard,
        coroutine=_get_company_news_tool,
        name="get_company_news",
        description="Recent company news within the past N days (default 7).",
        args_schema=CompanyNewsInput,
    ),
]

TOOLS = [
    StructuredTool.from_function(
        func=simple_interest_tool,
        name="calculate_simple_interest",
        description="Calculate simple interest and totals (optionally inflation-adjusted).",
        args_schema=SimpleInterestInput,
    ),
    StructuredTool.from_function(
        func=compound_interest_tool,
        name="calculate_compound_interest",
        description="Compound interest with configurable compounding frequency and optional inflation.",
        args_schema=CompoundInterestInput,
    ),
    StructuredTool.from_function(
        func=investment_return_tool,
        name="calculate_investment_return",
        description="Investment future value with separate compounding and contribution frequencies.",
        args_schema=InvestmentReturnInput,
    ),
    StructuredTool.from_function(
        func=investment_return_strings_tool,
        name="calculate_investment_return_from_strings",
        description="Investment FV using calculator-style strings like 'monthly', 'biweekly', 'begin/end'.",
        args_schema=InvestmentReturnStringsInput,
    ),
    StructuredTool.from_function(
        func=loan_amortization_tool,
        name="calculate_loan_amortization",
        description="Level payment, total paid, and total interest for a fixed-rate loan.",
        args_schema=LoanAmortizationInput,
    ),
    StructuredTool.from_function(
        func=npv_tool,
        name="npv",
        description="Net Present Value given a per-period discount rate (percent) and cashflows t=0..N.",
        args_schema=NPVInput,
    ),
    StructuredTool.from_function(
        func=cagr_tool,
        name="cagr",
        description="Compound Annual Growth Rate: returns a JSON object with percent and a markdown summary.",
        args_schema=CAGRInput,
    ),
    StructuredTool.from_function(
        func=document_search_tool,
        name="document_search",
        description="Search the local finance knowledge base and answer concisely with citations.",
        args_schema=DocumentSearchInput,
    ),
    StructuredTool.from_function(
        func=get_indicator,
        name="get_macro_indicator",
        description=(
            "Fetch a World Bank indicator for a country (ISO2/ISO3). "
            "Args: country (e.g., 'CA' or 'CAN'), indicator (e.g., 'FP.CPI.TOTL.ZG'), "
            "latest_only (bool, default True). Returns rows with year/value."
        ),
    ),
    *FINNHUB_TOOLS,
]
