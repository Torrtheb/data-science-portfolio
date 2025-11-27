from __future__ import annotations

import os
import re
import time
import asyncio
from typing import Any, Dict, Optional, Tuple, List
import httpx
from fastapi import APIRouter, HTTPException, Query

# --------------------------- Project imports ---------------------------------
try:
    from ..llm.agent_tools import (
        get_profile,
        get_recommendation_trends,
        get_company_news,
        _guarded_call_dict,
    )
except Exception:

    async def get_profile(symbol: str):
        return {"error": "tools_unavailable"}

    async def get_recommendation_trends(symbol: str):
        return {"error": "tools_unavailable"}

    async def get_company_news(symbol: str, days: int, limit: int):
        return {"error": "tools_unavailable"}

    async def _guarded_call_dict(_name: str, awaitable):
        try:
            return await awaitable
        except Exception as e:
            return {"error": "tool_call_failed", "detail": str(e)}


try:
    from ..utils.symbols import NON_US_SUFFIXES, canonize_symbol
except Exception:
    NON_US_SUFFIXES = {
        "TO",
        "V",
        "L",
        "AX",
        "HK",
        "SZ",
        "SS",
        "KS",
        "TW",
        "PA",
        "MI",
        "F",
        "SW",
        "BMV",
        "T",
        "NZ",
    }

    def canonize_symbol(x: str) -> str:
        return x


try:
    from ..core.settings import settings
except Exception:
    settings = None

price_router = APIRouter()

# --------------------------- Constants & Regexes ------------------------------

STOPWORDS = {
    "and",
    "&",
    "the",
    "a", "b", "c",
    "co",
    "company",
    "inc",
    "corp",
    "plc",
    "sa",
    "ltd",
    "nv",
    "ag",
    "holdings",
    "group",
    "class",
    "stock",
    "shares",
    "inc.",
    "classa",
    "classb",
    "classc",
}

_WORD = re.compile(r"[a-z0-9]+", re.I)
_NAMEY = re.compile(r"\s")
_CLASS_SHARE_ROOTS = {"BRK", "BF", "HEI"}

_ALIASES: Dict[str, str] = {
    "procter gamble": "PG",
    "procter and gamble": "PG",
    "procter & gamble": "PG",
    "alphabet": "GOOGL",
    "alphabet inc": "GOOGL",
    "alphabet class a": "GOOGL",
    "alphabet class c": "GOOG",
    "berkshire hathaway": "BRK.B",
    "berkshire hathaway inc": "BRK.B",
    "berkshire hathaway class a": "BRK.A",
    "berkshire hathaway class b": "BRK.B",
    "google": "GOOGL",
    "google inc": "GOOGL",
    "google class a": "GOOGL",
    "google class c": "GOOG",
    "facebook": "META",
    "fb": "META",
    "meta": "META",
    "meta platforms": "META",
    "meta platforms inc": "META",
    "square": "SQ",
    "block": "SQ",
    "block inc": "SQ",
    "proctor and gamble": "PG",
    "proctor & gamble": "PG",
}

_TRANSPORT = httpx.AsyncHTTPTransport(retries=2)
_NAME_CACHE: Dict[Tuple[str, str], Tuple[Optional[str], List[Dict]]] = {}


def _client(timeout: float = 10.0) -> httpx.AsyncClient:
    """
    Create a shared 'httpx.AsyncClient' with a stable User-Agent and
    a process-level transport (with retries).

    Args:
        timeout: Per-request timeout in seconds.

    Returns:
        Configured 'AsyncClient' ready for use in 'async with' blocks.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        transport=_TRANSPORT,
        headers={"User-Agent": "finance-chatbot/price/1.0"},
    )


# ------------------------- Finnhub token retrieval ---------------------------


def _get_finnhub_token() -> str:
    """
    Retrieve the Finnhub API token from configuration.

    Precedence:
      1) 'settings.finnhub_api_key' (Pydantic 'SecretStr' or str)
      2) 'FINNHUB_API_KEY' or 'FINNHUB_TOKEN' environment vars
      3) Empty string if nothing configured

    Returns:
        API token string (possibly empty).
    """
    if settings and getattr(settings, "finnhub_api_key", None):
        try:
            return settings.finnhub_api_key.get_secret_value()
        except Exception:
            return str(settings.finnhub_api_key)
    return os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_TOKEN") or ""


# ----------------------------- Text helpers ----------------------------------


def _norm_name(s: str) -> str:
    """
    Normalize a company name into a stopword-filtered token string.

    - Lowercases and extracts alphanumerics.
    - Removes generic company stopwords (e.g., 'inc', 'ltd', 'the').

    Args:
        s: Raw company/name string.

    Returns:
        Space-joined normalized tokens.
    """
    toks = [t for t in _WORD.findall((s or "").lower()) if t not in STOPWORDS]
    return " ".join(toks)


def _norm(s: str) -> List[str]:
    """
    Tokenize a string to lowercase alphanumerics (no stopword filtering).

    Args:
        s: Input text.

    Returns:
        List of tokens.
    """
    return _WORD.findall((s or "").lower())


def _looks_like_name_query(q: str) -> bool:
    """
    Heuristic to decide if a query is a name (vs. a ticker).

    Rules:
      - Contains whitespace, OR
      - Contains any lowercase character.

    Args:
        q: User query string.

    Returns:
        True if likely a company/name search.
    """
    s = (q or "").strip()
    return bool(s) and (bool(_NAMEY.search(s)) or (s != s.upper()))


def _root(sym: str) -> str:
    """
    Extract the class/root portion of a ticker.

    Examples:
        "BRK.B" → "BRK"
        "AAPL"  → "AAPL"

    Args:
        sym: Symbol string.

    Returns:
        Uppercased root symbol.
    """
    return (sym or "").split(".", 1)[0].upper()


def _canon_local(s: str) -> str:
    """
    Local defensive canonicalizer for tickers.

    Normalizes common variants:
      - Class shares (BRK B → BRK.B, BF/A → BF.A, etc.)
      - Canadian suffixes (.TSX, :CA, -T → .TO; .TSXV → .V)
      - UK/AU/HK suffix hints (:GB→.L, :AU→.AX, :HK→.HK)
      - Removes '.US' / ':US' artifacts
      - Collapses repeated dots

    Args:
        s: Raw symbol string.

    Returns:
        Canonicalized uppercase symbol string.
    """
    if not s:
        return ""
    u = (s or "").strip().upper()
    u = re.sub(r"\bBRK(?:[ \-/])?B\b", "BRK.B", u)
    u = re.sub(r"\bBRK(?:[ \-/])?A\b", "BRK.A", u)
    u = re.sub(r"\bBF(?:[ \-/])?B\b", "BF.B", u)
    u = re.sub(r"\bBF(?:[ \-/])?A\b", "BF.A", u)
    u = re.sub(r"\bHEI(?:[ \-/])?A\b", "HEI.A", u)
    u = re.sub(r"\.TSX$", ".TO", u)
    u = re.sub(r":CA$", ".TO", u)
    u = re.sub(r"-T$", ".TO", u)
    u = re.sub(r"\.TSXV$", ".V", u)
    u = re.sub(r"\.TOR$", ".TO", u)
    u = re.sub(r":GB$", ".L", u)
    u = re.sub(r":AU$", ".AX", u)
    u = re.sub(r":HK$", ".HK", u)
    u = re.sub(r"\.US\b", "", u)
    u = re.sub(r":US\b", "", u)
    u = re.sub(r"\.+", ".", u)
    m = re.fullmatch(r"([A-Z]{1,5})[-/ ]([A-Z])", u)
    if m and m.group(1) in _CLASS_SHARE_ROOTS:
        u = f"{m.group(1)}.{m.group(2)}"
    return u


def CANON(x: str) -> str:
    """
    Canonicalize a symbol using the shared 'canonize_symbol' when available;
    otherwise fall back to '_canon_local'.

    Args:
        x: Raw symbol string.

    Returns:
        Canonicalized symbol.
    """
    try:
        return canonize_symbol(x) if callable(canonize_symbol) else _canon_local(x)
    except Exception:
        return _canon_local(x)


def _is_supported_equity_symbol(sym: str) -> bool:
    """
    Validate that a symbol is supported under a US-only plan.

    Rules:
      - Allow bare US tickers (A–Z, length 1–5).
      - Allow US classes/units/warrants (e.g., BRK.B, XYZ.U, ABC.WS).
      - Reject known non-US exchange suffixes (e.g., .TO, .L, .AX, .HK).

    Args:
        sym: Symbol to check.

    Returns:
        True if acceptable, False otherwise.
    """
    u = (sym or "").upper()
    if "." not in u:
        return u.isalpha() and 1 <= len(u) <= 5

    tail = u.split(".", 1)[1]

    if tail in NON_US_SUFFIXES:
        return False

    if len(tail) == 1 and tail.isalpha():
        return True
    if tail in {"U", "WS"}:
        return True

    return False


# -------------------------- Finnhub search helpers ---------------------------


def _hint_prefs(q: str) -> Dict[str, int]:
    """
    Infer regional exchange hints from user text.

    Seeds a small preference map (US-biased by default) and bumps regions
    the user mentions (e.g., 'TSX', 'London', 'HK').

    Args:
        q: User query text.

    Returns:
        Dict of hint weights keyed by region/exchange tokens.
    """
    s = (q or "").lower()
    prefs = {
        "us": 0,
        "nasdaq": 0,
        "nyse": 0,
        "amex": 0,
        "tsx": 0,
        "toronto": 0,
        "canada": 0,
        "to": 0,
        "tsxv": 0,
        "london": 0,
        "lse": 0,
        "uk": 0,
        "asx": 0,
        "australia": 0,
        "ax": 0,
        "hk": 0,
        "hong kong": 0,
        "sehk": 0,
        "hkex": 0,
    }
    for k in list(prefs.keys()):
        if k in s:
            prefs[k] += 1
    return prefs


def _exchange_hint_weight(sym: str, exch: str, prefs: Dict[str, int]) -> int:
    """
    Score a candidate by exchange desirability and user hints.

    Preference tiers:
      - Strong preference for US listings (NASDAQ/NYSE/ARCA).
      - Modest base weight for TSX/LSE/ASX/SEHK, boosted when hinted in query.

    Args:
        sym: Candidate symbol (unused for now, reserved for future rules).
        exch: Exchange name from profile.
        prefs: Output of '_hint_prefs'.

    Returns:
        Integer weight to add to the candidate score.
    """
    e = (exch or "").upper()
    w = 0
    US_MATCHES = ("NASDAQ", "NEW YORK", "NYSE", "XNYS", "XNAS", "XNGS", "NASDAQ/NMS")
    if any(k in e for k in US_MATCHES):
        w += 9
    if "NYSE ARCA" in e or "ARCX" in e:
        w += 4
    CA_MATCHES = ("TSX", "TORONTO", "XTSE", "XTSX", "TSXV")
    if any(k in e for k in CA_MATCHES):
        w += 2
        if any(prefs.get(k) for k in ("tsx", "toronto", "canada", "to", "tsxv")):
            w += 6
    UK_MATCHES = ("LSE", "LONDON", "XLON")
    if any(k in e for k in UK_MATCHES):
        w += 2
        if any(prefs.get(k) for k in ("london", "lse", "uk")):
            w += 4
    AU_MATCHES = ("ASX", "XASX", "AUSTRALIAN")
    if any(k in e for k in AU_MATCHES):
        w += 2
        if any(prefs.get(k) for k in ("asx", "australia", "ax")):
            w += 4
    HK_MATCHES = ("HONG KONG", "SEHK", "XHKG", "HKEX")
    if any(k in e for k in HK_MATCHES):
        w += 2
        if any(prefs.get(k) for k in ("hk", "hong kong", "sehk", "hkex")):
            w += 4
    return w


async def _search_symbol_finnhub(q: str, token: str) -> List[Dict]:
    """
    Query Finnhub '/search' and normalize the response.

    Args:
        q: Search string (name or ticker).
        token: Finnhub API token.

    Returns:
        List of dicts with fields:
          - symbol
          - displaySymbol
          - description
          - type

    Notes:
        Retries once after a short delay on HTTP 429. Returns [] on 429/503.
    """
    if not token:
        return []
    url = "https://finnhub.io/api/v1/search"
    params = {"q": q, "token": token}
    async with _client(10) as h:
        try:
            r = await h.get(url, params=params)
            if r.status_code == 429:
                await asyncio.sleep(0.25)
                r = await h.get(url, params=params)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if getattr(e.response, "status_code", None) in (429, 503):
                return []
            raise
        js = r.json() or {}
        items = js.get("result") or []
        return [
            {
                "symbol": it.get("symbol"),
                "displaySymbol": it.get("displaySymbol") or it.get("symbol"),
                "description": it.get("description") or "",
                "type": it.get("type") or "",
            }
            for it in items
        ]


async def _profile_bulk_cap_currency(symbols: List[str], token: str) -> Dict[str, Dict]:
    """
    Fetch a small slice of '/stock/profile2' for multiple symbols in parallel.

    Pulled fields:
      - name
      - exchange
      - currency
      - market_cap (marketCapitalization)

    Args:
        symbols: List of symbols to fetch.
        token: Finnhub API token.

    Returns:
        Mapping: symbol → profile dict (missing keys omitted on errors).
    """
    out: Dict[str, Dict] = {}
    if not token or not symbols:
        return out

    async def fetch_one(h: httpx.AsyncClient, s: str) -> None:
        try:
            r = await h.get(
                "https://finnhub.io/api/v1/stock/profile2",
                params={"symbol": s, "token": token},
                timeout=10,
            )
            r.raise_for_status()
            jd = r.json() or {}
            out[s] = {
                "name": jd.get("name"),
                "exchange": jd.get("exchange"),
                "currency": jd.get("currency"),
                "market_cap": jd.get("marketCapitalization") or 0.0,
            }
        except Exception:
            out[s] = {}

    async with _client(10) as h:
        await asyncio.gather(
            *(fetch_one(h, s) for s in symbols), return_exceptions=True
        )
    return out


def _score_candidate(
    q_tokens: List[str],
    cand: Dict,
    prof: Dict,
    prefs: Dict[str, int],
    bare_roots: Optional[set[str]] = None,
) -> float:
    """
    Score a '/search' candidate using multiple weak signals.

    Signals:
      - Token overlap with description (common, starts-with, exact).
      - User region hints + exchange desirability.
      - Market cap (log-scaled).
      - Simplicity (bare tickers > suffixed tickers).
      - Root consistency (prefer BRK over BRK.A if user typed 'BRK').

    Args:
        q_tokens: Tokenized user query.
        cand: Candidate item from '/search'.
        prof: Profile info (exchange, market_cap).
        prefs: Regional hint weights.
        bare_roots: Set of bare roots seen across candidates.

    Returns:
        Floating point score; higher is better.
    """
    sym = (cand.get("symbol") or "").upper()
    disp = cand.get("description") or ""
    typ = (cand.get("type") or "").lower()
    pt = prof or {}
    exch = pt.get("exchange") or ""
    cap = float(pt.get("market_cap") or 0.0)

    toks = _norm(disp)
    common = len(set(q_tokens) & set(toks))
    starts = 1 if toks and toks[0] in q_tokens else 0
    exact = 1 if set(q_tokens) == set(toks) and len(toks) > 0 else 0

    w = 0.0
    w += common * 2.0
    w += starts * 1.5
    w += exact * 3.0

    q_core = [t for t in q_tokens if t not in STOPWORDS]
    if q_core and all(t in toks for t in q_core):
        w += 2.0

    if "stock" in typ:
        w += 1.0
    if "warrant" in typ:
        w -= 1.0

    w += _exchange_hint_weight(sym, exch, prefs)

    if cap > 0:
        from math import log10

        w += min(6.0, max(0.0, log10(cap + 1.0)))

    if "." not in sym:
        w += 1.5

    if q_core and toks[: len(q_core)] == q_core:
        w += 1.25

    if bare_roots and _root(sym) in bare_roots:
        if "." not in sym:
            w += 4.0
        else:
            w -= 2.5

    return w


async def _resolve_name_cached(q: str, token: str) -> Tuple[Optional[str], List[Dict]]:
    """
    Cached wrapper around '_resolve_name_to_symbol'.

    Args:
        q: User query string.
        token: Finnhub API token.

    Returns:
        (best_symbol|None, aliases_list). Uses an LRU-ish dict capped at ~5k.
    """
    k = ((q or "").strip().lower(), token or "")
    if k in _NAME_CACHE:
        return _NAME_CACHE[k]
    res = await _resolve_name_to_symbol(q, token)
    _NAME_CACHE[k] = res
    if len(_NAME_CACHE) > 5000:
        for _ in range(1000):
            try:
                _NAME_CACHE.pop(next(iter(_NAME_CACHE)))
            except StopIteration:
                break
    return res


async def _resolve_name_to_symbol(
    q: str, token: str
) -> Tuple[Optional[str], List[Dict]]:
    """
    Resolve a human name (e.g., 'apple') to a US-supported ticker.

    Steps:
      1) Call Finnhub '/search' with the original query; if empty, retry with
         stopword-stripped query.
      2) Filter to supported symbols via '_is_supported_equity_symbol'.
      3) Drop OTC unless the user explicitly asked for OTC.
      4) Rank head candidates using '_score_candidate'.
      5) Return (best_symbol, up to 5 alias suggestions).

    Args:
        q: Raw user query.
        token: Finnhub API token.

    Returns:
        Tuple of (best symbol or None, list of alias dicts).
    """
    items = await _search_symbol_finnhub(q, token)
    if not items:
        q2 = " ".join(t for t in _norm(q) if t not in STOPWORDS)
        if q2 and q2 != (q or "").lower():
            items = await _search_symbol_finnhub(q2, token)
        if not items:
            return (None, [])

    user_wants_otc = (
        bool(re.search(r"\bOTC\b", (q or ""), re.I)) or ".OTC" in (q or "").upper()
    )

    def _is_otc_symbol_desc(desc: str) -> bool:
        d = (desc or "").upper()
        return any(k in d for k in ("OTC", "OTC MARKETS", "PINK"))

    filtered: List[Dict] = []
    for it in items:
        s = (it.get("symbol") or "").upper()
        if not _is_supported_equity_symbol(s):
            continue
        if (not user_wants_otc) and _is_otc_symbol_desc(it.get("description") or ""):
            continue
        filtered.append(it)

    if not filtered:
        return (None, [])

    q_tokens = _norm(q)

    def coarse_score(it: Dict) -> int:
        desc = (it.get("description") or "").lower()
        hits = sum(1 for t in q_tokens if t in desc)
        starts = 1 if desc.startswith(" ".join(q_tokens[:2]).strip()) else 0
        return hits * 2 + starts

    filtered.sort(key=coarse_score, reverse=True)
    head = filtered[:8]

    prof_map = await _profile_bulk_cap_currency(
        [it["symbol"] for it in head if it.get("symbol")], token
    )

    bare_roots = {
        _root(it.get("symbol") or "")
        for it in filtered
        if it.get("symbol") and "." not in (it.get("symbol") or "")
    }

    prefs = _hint_prefs(q)
    best: Optional[Dict] = None
    best_score = float("-inf")
    for it in head:
        s = (it.get("symbol") or "").upper()
        prof = prof_map.get(s, {})
        sc = _score_candidate(q_tokens, it, prof, prefs, bare_roots=bare_roots)
        if sc > best_score:
            best_score = sc
            best = it

    if not best:
        return (None, [])

    aliases: List[Dict] = []
    seen = set()
    for it in filtered[:12]:
        s = (it.get("symbol") or "").upper()
        nm = (it.get("description") or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        aliases.append({"symbol": s, "name": nm})
        if len(aliases) >= 5:
            break

    return (best.get("symbol"), aliases)


# ------------------------------- Price core ----------------------------------


async def _price_core(q: str, debug: bool = False) -> Dict[str, Any]:
    """
    Resolve a query to a US-supported symbol and fetch current price data.

    Behavior:
      - Applies alias rules for common companies (e.g., Alphabet, Berkshire).
      - Attempts name→ticker resolution when the input looks like a name.
      - Enforces US-only support; rejects non-US suffixes and OTC unless asked.
      - Primary data path: '/quote'; falls back to recent '/stock/candle'
        on 401/403 to surface a last price.
      - Enriches with 'profile2' (name, currency); double-checks OTC.
      - Returns alias suggestions when a search occurred.

    Args:
        q: User query (ticker or company name).
        debug: If True, include '_tools_used' trace in the output.

    Returns:
        Structured dict with either price fields or an error payload:
          Success fields:
            symbol, asset, provider_symbol, price, prevClose, open, high, low,
            change, changePercent, source ('finnhub:quote'|'finnhub:candles'), ts,
            optional: name, currency, aliases[]
          Error fields:
            error (str), status (int), optional hint/detail/path, symbol/provider_symbol.

    Notes:
        This function never raises 'HTTPException'; HTTP layer maps errors.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Missing 'q'")

    FINNHUB_TOKEN = _get_finnhub_token()
    raw_q = (q or "").strip()
    low = raw_q.lower()
    norm = _norm_name(raw_q)
    if raw_q.strip().upper() == "BRK":
        prov_sym = user_sym = "BRK.B"
    elif raw_q.strip().upper() == "BF":
        prov_sym = user_sym = "BF.B"

    if re.search(r"\.(%s)\b" % "|".join(NON_US_SUFFIXES), raw_q, re.I):
        return {
            "query": raw_q,
            "error": "unsupported_market",
            "status": 400,
            "hint": (
                "Your Finnhub plan supports US symbols only. Try the US primary ticker "
                "(e.g., 'SHOP' on NYSE/NASDAQ if available, not 'SHOP.TO')."
            ),
        }

    tools_used: List[Dict] = []
    prov_sym: Optional[str] = None
    user_sym: Optional[str] = None
    aliases_resolved: List[Dict] = []

    if "berkshire" in low and "hathaway" in low:
        if re.search(r"\bclass\s*a\b", low):
            prov_sym = user_sym = "BRK.A"
        elif re.search(r"\bclass\s*b\b", low):
            prov_sym = user_sym = "BRK.B"
        else:
            prov_sym = user_sym = "BRK.B"
    elif "alphabet" in low:
        prov_sym = user_sym = "GOOG" if re.search(r"\bclass\s*c\b", low) else "GOOGL"
    elif norm in _ALIASES:
        prov_sym = user_sym = _ALIASES[norm]

    if not prov_sym and _looks_like_name_query(raw_q):
        try:
            best, aliases = await _resolve_name_cached(raw_q, FINNHUB_TOKEN)
            aliases_resolved = aliases or []
            tools_used.append(
                {"tool": "finnhub.search", "q": raw_q, "aliases": aliases_resolved}
            )
            if best:
                prov_sym = user_sym = CANON(best)
        except Exception as e:
            tools_used.append({"tool": "finnhub.search", "q": raw_q, "error": str(e)})

    if not prov_sym:
        raw_token = re.sub(r"^\$", "", raw_q).strip()
        canon_token = CANON(raw_token).upper()
        user_bare_ticker = bool(re.fullmatch(r"[A-Z]{1,5}", canon_token))
        canon_has_dot = "." in canon_token

        user_sym = prov_sym = CANON(raw_q.upper())

        if user_bare_ticker and canon_token:
            user_sym = prov_sym = canon_token
        elif canon_has_dot:
            if _is_supported_equity_symbol(canon_token):
                user_sym = prov_sym = canon_token
        else:
            try:
                best, aliases = await _resolve_name_cached(raw_q, FINNHUB_TOKEN)
                aliases_resolved = aliases or []
                tools_used.append(
                    {"tool": "finnhub.search", "q": q, "aliases": aliases_resolved}
                )
                if best:
                    bs = CANON(best)
                    user_sym = prov_sym = bs
                else:
                    out = {
                        "error": "symbol_not_found",
                        "status": 404,
                        "detail": "No symbol match",
                        "query": q,
                    }
                    if debug:
                        out["_tools_used"] = tools_used
                    return out
            except Exception as e:
                tools_used.append({"tool": "finnhub.search", "q": q, "error": str(e)})

    if not _is_supported_equity_symbol(prov_sym or ""):
        out = {
            "symbol": user_sym,
            "asset": "equity",
            "provider_symbol": prov_sym,
            "error": "unsupported_market",
            "status": 400,
            "hint": "This market/symbol is not supported by the current Finnhub plan.",
        }
        if debug:
            out["_tools_used"] = tools_used
        return out

    # ----------------------------------------------------------------------------
    # Fetch price: /quote, fallback to /stock/candle on 401/403
    # ----------------------------------------------------------------------------
    try:
        qurl = "https://finnhub.io/api/v1/quote"
        async with _client(10) as h:
            resp = await h.get(
                qurl, params={"symbol": prov_sym, "token": FINNHUB_TOKEN}
            )

        if resp.status_code in (401, 403):
            now = int(time.time())
            frm = now - 3 * 24 * 3600
            curl = "https://finnhub.io/api/v1/stock/candle"
            async with _client(10) as h:
                r2 = await h.get(
                    curl,
                    params={
                        "symbol": prov_sym,
                        "resolution": "D",
                        "from": frm,
                        "to": now,
                        "token": FINNHUB_TOKEN,
                    },
                )

            if r2.status_code in (401, 403):
                out = {
                    "symbol": user_sym,
                    "asset": "equity",
                    "provider_symbol": prov_sym,
                    "error": "forbidden",
                    "status": r2.status_code,
                    "path": "/stock/candle",
                    "hint": "This symbol may be outside your Finnhub plan (e.g., TSX on Free).",
                }
                if debug:
                    out["_tools_used"] = tools_used
                return out

            r2.raise_for_status()
            dd = r2.json() or {}
            if dd.get("s") != "ok" or not dd.get("c"):
                out = {
                    "symbol": user_sym,
                    "asset": "equity",
                    "provider_symbol": prov_sym,
                    "error": "upstream_error",
                    "status": r2.status_code,
                    "path": "/stock/candle",
                    "hint": "No candle data returned.",
                }
                if debug:
                    out["_tools_used"] = tools_used
                return out

            last = float(dd["c"][-1])
            prev = float(dd["c"][-2]) if len(dd["c"]) > 1 else last
            chg = last - prev
            chg_pct = (chg / prev) * 100.0 if prev else None
            ts = int(dd["t"][-1]) if dd.get("t") else int(time.time())
            out = {
                "symbol": user_sym,
                "asset": "equity",
                "provider_symbol": prov_sym,
                "price": last,
                "prevClose": prev,
                "change": chg,
                "changePercent": chg_pct,
                "source": "finnhub:candles",
                "ts": ts,
            }

            try:
                if FINNHUB_TOKEN and isinstance(prov_sym, str) and prov_sym:
                    async with _client(8) as h:
                        pr = await h.get(
                            "https://finnhub.io/api/v1/stock/profile2",
                            params={"symbol": prov_sym, "token": FINNHUB_TOKEN},
                        )
                    pj = pr.json() if pr.status_code == 200 else {}
                    if pj.get("name") and not out.get("name"):
                        out["name"] = pj.get("name")
                    if pj.get("currency") and not out.get("currency"):
                        out["currency"] = pj.get("currency")

                    exch = (pj.get("exchange") or "").upper()
                    is_otc = any(k in exch for k in ("OTC", "OTC MARKETS", "OTCM"))
                    user_asked_otc = bool(
                        re.search(r"\bOTC\b", (raw_q or ""), re.I)
                    ) or (".OTC" in (prov_sym or ""))
                    if is_otc and not user_asked_otc:
                        return {
                            "symbol": user_sym,
                            "asset": "equity",
                            "provider_symbol": prov_sym,
                            "error": "unsupported_market",
                            "status": 400,
                            "hint": "This looks like an OTC ticker; try a primary US-listed symbol or add 'OTC' explicitly.",
                        }
            except Exception:
                pass

            if "aliases" not in out and aliases_resolved:
                out["aliases"] = [
                    {"symbol": a.get("symbol"), "name": a.get("name")}
                    for a in aliases_resolved
                    if a.get("symbol")
                ]
            if debug:
                out["_tools_used"] = tools_used
            return out

        resp.raise_for_status()
        qd = resp.json() or {}

        if (("error" in qd) or (qd.get("c") in (None, 0))) and "." in (prov_sym or ""):
            alt = (prov_sym or "").split(".", 1)[0].upper()
            if alt and alt != prov_sym and _is_supported_equity_symbol(alt):
                async with _client(10) as h:
                    resp2 = await h.get(
                        qurl, params={"symbol": alt, "token": FINNHUB_TOKEN}
                    )
                if resp2.status_code not in (401, 403):
                    resp2.raise_for_status()
                    qd2 = resp2.json() or {}
                    if ("error" not in qd2) and (qd2.get("c") not in (None, 0)):
                        qd = qd2
                        prov_sym = user_sym = alt

        if ("error" in qd) or (qd.get("c") in (None, 0)):
            if not re.fullmatch(r"[A-Z]{1,5}", (raw_q or "").strip().upper()):
                try:
                    best, aliases = await _resolve_name_to_symbol(q, FINNHUB_TOKEN)
                    if best:
                        aliases_resolved = aliases or aliases_resolved
                        bs = CANON(best)
                        async with _client(10) as h:
                            resp3 = await h.get(
                                qurl, params={"symbol": bs, "token": FINNHUB_TOKEN}
                            )
                        if resp3.status_code not in (401, 403):
                            resp3.raise_for_status()
                            qd3 = resp3.json() or {}
                            if ("error" not in qd3) and (qd3.get("c") not in (None, 0)):
                                qd = qd3
                                prov_sym = user_sym = bs
                except Exception:
                    pass

        if ("error" in qd) or (qd.get("c") in (None, 0)):
            return {
                "symbol": user_sym,
                "asset": "equity",
                "provider_symbol": prov_sym,
                "error": qd.get("error") or "symbol_not_supported",
                "status": 200,
                "path": "/quote",
            }

        price = float(qd.get("c"))
        pc = float(qd["pc"]) if qd.get("pc") is not None else None
        o = float(qd["o"]) if qd.get("o") is not None else None
        h = float(qd["h"]) if qd.get("h") is not None else None
        l = float(qd["l"]) if qd.get("l") is not None else None
        ts = int(qd["t"]) if qd.get("t") else int(time.time())

        chg = chg_pct = None
        if pc is not None:
            try:
                chg = price - pc
                chg_pct = (chg / pc * 100.0) if pc else None
            except Exception:
                chg = chg_pct = None

        out: Dict[str, Any] = {
            "symbol": user_sym,
            "asset": "equity",
            "provider_symbol": prov_sym,
            "price": price,
            "prevClose": pc,
            "open": o,
            "high": h,
            "low": l,
            "change": chg,
            "changePercent": chg_pct,
            "source": "finnhub:quote",
            "ts": ts,
        }
        pj = {}
        try:
            if FINNHUB_TOKEN and isinstance(prov_sym, str) and prov_sym:
                async with _client(8) as h:
                    pr = await h.get(
                        "https://finnhub.io/api/v1/stock/profile2",
                        params={"symbol": prov_sym, "token": FINNHUB_TOKEN},
                    )
                if pr.status_code == 200:
                    pj = pr.json() or {}
                    if pj.get("name") and not out.get("name"):
                        out["name"] = pj.get("name")
                    if pj.get("currency") and not out.get("currency"):
                        out["currency"] = pj.get("currency")

                exch = (pj.get("exchange") or "").upper()
                is_otc = any(k in exch for k in ("OTC", "OTC MARKETS", "OTCM"))
                user_asked_otc = bool(re.search(r"\bOTC\b", (raw_q or ""), re.I)) or (
                    ".OTC" in (prov_sym or "")
                )
                if is_otc and not user_asked_otc:
                    return {
                        "symbol": user_sym,
                        "asset": "equity",
                        "provider_symbol": prov_sym,
                        "error": "unsupported_market",
                        "status": 400,
                        "hint": "This looks like an OTC ticker; try a primary US-listed symbol or add 'OTC' explicitly.",
                    }
        except Exception:
            pass

        try:
            if FINNHUB_TOKEN and isinstance(prov_sym, str) and prov_sym:
                async with _client(8) as h:
                    pr = await h.get(
                        "https://finnhub.io/api/v1/stock/profile2",
                        params={"symbol": prov_sym, "token": FINNHUB_TOKEN},
                    )
                if pr.status_code == 200:
                    pj = pr.json() or {}
                    if pj.get("name") and not out.get("name"):
                        out["name"] = pj.get("name")
                    if pj.get("currency") and not out.get("currency"):
                        out["currency"] = pj.get("currency")

                exch = (pj.get("exchange") or "").upper() if "pj" in locals() else ""
                is_otc = any(k in exch for k in ("OTC", "OTC MARKETS", "OTCM"))
                user_asked_otc = bool(re.search(r"\bOTC\b", (raw_q or ""), re.I)) or (
                    ".OTC" in (prov_sym or "")
                )
                if is_otc and not user_asked_otc:
                    return {
                        "symbol": user_sym,
                        "asset": "equity",
                        "provider_symbol": prov_sym,
                        "error": "unsupported_market",
                        "status": 400,
                        "hint": "This looks like an OTC ticker; try a primary US-listed symbol or add 'OTC' explicitly.",
                    }
        except Exception:
            pass

        if "aliases" not in out and aliases_resolved:
            out["aliases"] = [
                {"symbol": a.get("symbol"), "name": a.get("name")}
                for a in aliases_resolved
                if a.get("symbol")
            ]
        return out

    except httpx.HTTPError as e:
        return {
            "symbol": user_sym,
            "asset": "equity",
            "provider_symbol": prov_sym,
            "error": "upstream_error",
            "status": getattr(getattr(e, "response", None), "status_code", None),
            "detail": str(e),
        }


# --------------------------- HTTP endpoints ----------------------------------


@price_router.get("/api/profile")
async def api_profile(symbol: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    Proxy endpoint for the profile tool.

    Args:
        symbol: Ticker symbol.

    Returns:
        Profile dict from tool.

    Raises:
        HTTPException 502: When the underlying tool returns an error payload.
    """
    res = await _guarded_call_dict("get_profile", get_profile(symbol))
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(502, detail=res.get("detail") or res.get("error"))
    return res


@price_router.get("/api/reco-trends")
async def api_reco_trends(symbol: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    Proxy endpoint for analyst recommendation trends.

    Args:
        symbol: Ticker symbol.

    Returns:
        Recommendation trends dict.

    Raises:
        HTTPException 502: When the underlying tool returns an error payload.
    """
    res = await _guarded_call_dict(
        "get_recommendation_trends", get_recommendation_trends(symbol)
    )
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(502, detail=res.get("detail") or res.get("error"))
    return res


@price_router.get("/api/news")
async def api_news(
    symbol: str = Query(..., min_length=1),
    days: int = Query(7, gt=0, le=30),
    limit: int = Query(10, gt=0, le=50),
) -> Any:
    """
    Proxy endpoint for company news.

    Args:
        symbol: Ticker symbol.
        days: Lookback window (1–30).
        limit: Max items to return (1–50).

    Returns:
        News list or dict as returned by the tool.

    Raises:
        HTTPException 502: When the underlying tool returns an error payload.
    """
    res = await _guarded_call_dict(
        "get_company_news", get_company_news(symbol, days, limit)
    )
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(502, detail=res.get("detail") or res.get("error"))
    return res


@price_router.get("/api/price")
async def http_price(
    q: str = Query(..., description="Ticker or company query"), debug: bool = False
) -> Dict[str, Any]:
    """
    Resolve the user query to a supported equity symbol and return price data.

    - US-only enforcement and OTC restrictions (unless explicitly requested).
    - Chooses '/quote' or falls back to '/stock/candle' when needed.
    - Includes alias suggestions when name search occurs.

    Args:
        q: Ticker or company name query.
        debug: Include internal tool trace when True.

    Returns:
        Price payload or structured error from '_price_core'.
    """
    return await _price_core(q, debug)
