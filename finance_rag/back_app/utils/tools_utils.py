from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, List, Optional
from .text import _clean_title, _prettify_provider, _to_iso_day, _pretty_date


# ----------------------------
# Tool schema (Pydantic v1/v2)
# ----------------------------


def _tool_schema(t: Any) -> Dict[str, Any]:
    """
    Return a JSON schema for a LangChain Tool's args_schema, supporting
    both Pydantic v2 ('model_json_schema') and v1 ('schema').

    Args:
        t: Tool-like object (e.g., langchain_core.tools.Tool / StructuredTool).

    Returns:
        Dict JSON schema for the tool's input parameters (or empty dict).
    """
    schema: Dict[str, Any] = {}
    try:
        if getattr(t, "args_schema", None):
            return t.args_schema.model_json_schema()
    except Exception:
        pass

    try:
        if getattr(t, "args_schema", None):
            return t.args_schema.schema()
    except Exception:
        pass

    return schema


# ----------------------------
# Tool invocation wrapper
# ----------------------------


async def _call_tool(t: Any, args: Dict[str, Any]) -> Any:
    """
    Invoke a tool with best-effort compatibility across calling styles.

    Preference order:
      1) t.coroutine(**args)  — if defined and is a coroutine function
      2) t.func(**args)       — if defined (sync)
      3) t.ainvoke(args)      — LangChain async invoke
      4) t.invoke(args)       — LangChain sync invoke

    Args:
        t: Tool-like object.
        args: Dict of arguments to pass.

    Returns:
        Tool result (type depends on the tool).

    Raises:
        RuntimeError: if no callable entrypoint is found.
    """
    coro = getattr(t, "coroutine", None)
    func = getattr(t, "func", None)

    if coro and inspect.iscoroutinefunction(coro):
        return await coro(**(args or {}))
    if func:
        return func(**(args or {}))

    ainvoke = getattr(t, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(args or {})

    invoke = getattr(t, "invoke", None)
    if callable(invoke):
        return invoke(args or {})

    raise RuntimeError("Tool has no callable entry")


# ----------------------------
# Render tool observations → Markdown
# ----------------------------


def _collect_tool_markdown(steps: List[Any] | None) -> str:
    """
    Convert agent 'intermediate_steps' into a compact Markdown section.

    Expected observation shapes inside steps (best-effort detection):
        - dict with one of: {"news": [...]} | {"articles": [...]} | {"links": [...]} | {"items": [...]}
          where each item is a dict with keys such as:
              title/headline/name, source/provider, datetime

    The output looks like:

        ## 🗞️ News Headlines:

        - Title one · *Source* · Jan 2, 2025
        - Title two · *Source* · Jan 1, 2025

    Notes:
        - Titles are de-duplicated by a normalized token key.
        - Articles are sorted newest → oldest using YYYY-MM-DD.
        - If no qualifying items found, returns "".

    Args:
        steps: LangChain agent intermediate steps (list of (action, observation) or dicts).

    Returns:
        Markdown string (possibly empty).
    """
    sections: List[str] = []

    for step in steps or []:
        action = observation = None
        if isinstance(step, (list, tuple)) and len(step) == 2:
            action, observation = step
        elif isinstance(step, dict):
            action = step.get("action")
            observation = step.get("observation")

        obj: Optional[Dict[str, Any]] = None
        if isinstance(observation, str):
            try:
                maybe = json.loads(observation)
                if isinstance(maybe, dict):
                    obj = maybe
            except Exception:
                obj = None
        elif isinstance(observation, dict):
            obj = observation

        if not isinstance(obj, dict):
            continue

        items: Optional[List[Dict[str, Any]]] = None
        for k in ("news", "articles", "links", "items"):
            v = obj.get(k)
            if isinstance(v, (list, tuple)):
                items = [x for x in v if isinstance(x, dict)]
                break
        if not items:
            continue

        articles: List[tuple[str, str, str, str]] = []
        seen_titles: set[str] = set()

        for a in items:
            raw_title = str(
                (a.get("title") or a.get("headline") or a.get("name") or "Article")
            ).strip()
            title = _clean_title(raw_title)
            key = re.sub(r"\W+", "", title).lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)

            src = (a.get("source") or a.get("provider") or "").strip()
            src = _prettify_provider(src)

            iso_day = _to_iso_day(a.get("datetime"))
            pretty_day = _pretty_date(iso_day) if iso_day else ""

            sort_key = iso_day or "1900-01-01"
            articles.append((sort_key, title, src, pretty_day))

        if not articles:
            continue

        articles.sort(key=lambda x: x[0], reverse=True)

        header = "## 🗞️ News Headlines:"
        lines = [header, ""]

        for _, title, src, pretty_day in articles:
            meta = ""
            if src:
                meta += f" · *{src}*"
            if pretty_day:
                meta += f" · {pretty_day}"
            lines.append(f"- {title}{meta}")
            lines.append("")

        sections.append("\n".join(lines).rstrip())

    md = "\n\n".join(s for s in sections if s).strip()
    return md


# ----------------------------
# Toolability heuristic
# ----------------------------

TICKER_RX = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b")

_FINANCE_KEYWORDS_RX = re.compile(
    r"("
    r"\bstock(s)?\b|"
    r"\b(etf|fund|mutual fund|index|indices)\b|"
    r"\b(portfolio|asset allocation|rebalance|diversif(y|ication))\b|"
    r"\binvest(ing|ment|or|ors)?\b|"
    r"\bdividend(s)?\b|"
    r"\bbond(s)?\b|"
    r"\bloan(s)?\b|"
    r"\bmortgage(s)?\b|"
    r"\bcredit card(s)?\b|"
    r"\bapr\b|"
    r"\bapy\b|"
    r"\byield(s)?\b|"
    r"\bretirement\b|"
    r"\b401k\b|"
    r"\broth\b|"
    r"\bira\b|"
    r"\btfsa\b|"
    r"\brrsp\b|"
    r"\bpension\b|"
    r"\bbudget(ing)?\b|"
    r"\bsaving(s)?\b|"
    r"\bcash flow\b|"
    r"\bexpense ratio\b|"
    r"\bfee(s)?\b|"
    r"\binflation\b|"
    r"\bunemployment\b|"
    r"\bgdp\b|"
    r"\bnpv\b|"
    r"\birr\b|"
    r"\bcagr\b|"
    r"\bvaluation\b|"
    r"\binterest rate(s)?\b|"
    r"\bcompound interest\b|"
    r"\bcompounding\b|"
    r"\binterest\b|"
    r"\bdiscount rate\b"
    r")",
    re.I,
)

_CURRENCY_RX = re.compile(
    r"([$€£¥]|" r"\b(usd|eur|gbp|cad|aud|chf|jpy)\b)",
    re.I,
)


def _is_finance_query(q: str) -> bool:
    """
    Heuristic: is this query clearly finance-related?

    Signals:
        - Ticker-like tokens (AAPL, RY.TO).
        - Common finance/macro keywords (stocks, ETFs, loans, retirement, GDP, etc.).
        - Currency symbols or codes ($, EUR, CAD, ...).

    Notes:
        - Favor precision over recall: some borderline finance questions may slip
          through and be handled by the model rails, but clearly non-finance
          questions (jokes, general trivia) are rejected early.
    """
    if not q or not q.strip():
        return False
    s = q.strip()
    if TICKER_RX.search(s):
        return True
    if _CURRENCY_RX.search(s):
        return True
    if _FINANCE_KEYWORDS_RX.search(s):
        return True
    return False


def _looks_toolable(q: str) -> bool:
    """
    Heuristic: should we try tool calls for this query?

    Rules:
        - If there's a ticker-like token (AAPL, BNS.TO) → True.
        - If market-data verbs are present (price/quote/news/chart/profile),
          require either a ticker or a finance keyword to avoid false positives.
        - Otherwise, finance-calculation intents (pe/p-e/dividend/cagr/npv/etc.)
          can be toolable *without* a ticker.

    Args:
        q: Raw user query.

    Returns:
        True if the query suggests tool usage.
    """
    if not q:
        return False
    s = q.strip()
    if TICKER_RX.search(s):
        return True
    price_like = re.search(
        r"\b(price|quote|candles?|chart|news|profile|market price)\b", s, re.I
    )
    if price_like:
        # price/quote-like queries should be toolable even if the finance classifier misses
        return True
    return (
        re.search(
            r"\b(pe|p\/e|dividend|cagr|npv|loan|amort|compound(?:ing)?|interest|inflation|contribution)\b",
            s,
            re.I,
        )
        is not None
    )


# ----------------------------
# Analytics: extract tool events
# ----------------------------


def _extract_tool_events(steps: List[Any] | None, session_id: str) -> list[dict]:
    """
    Convert intermediate steps into analytics-friendly tool events.

    Produces a list of dicts like:
        {
          "type": "tool",
          "session_id": "...",
          "tool_name": "get_company_news",
          "ok": True/False,
          "error": str|None,
          "latency_ms": int|None,
        }

    Observation inference:
        - If observation is a JSON dict with "ok"/"error"/"elapsed_ms", we surface those.

    Args:
        steps: Agent intermediate steps (list of (action, observation) or dicts).
        session_id: Current chat session ID.

    Returns:
        List of analytics events.
    """
    events: list[dict] = []
    for st in steps or []:
        action = observation = None
        if isinstance(st, (list, tuple)) and len(st) == 2:
            action, observation = st
        elif isinstance(st, dict):
            action = st.get("action")
            observation = st.get("observation")
        name = (
            getattr(action, "tool", None)
            or (isinstance(action, dict) and action.get("tool"))
            or None
        )
        if not name:
            continue

        ok = True
        err: Optional[str] = None
        latency: Optional[int] = None
        obj: Optional[Dict[str, Any]] = None
        if isinstance(observation, str):
            try:
                maybe = json.loads(observation)
                if isinstance(maybe, dict):
                    obj = maybe
            except Exception:
                obj = None
        elif isinstance(observation, dict):
            obj = observation

        if isinstance(obj, dict):
            if "ok" in obj:
                try:
                    ok = bool(obj["ok"])
                except Exception:
                    pass
            if obj.get("error"):
                ok = False
                err = str(obj["error"])
            if isinstance(obj.get("elapsed_ms"), (int, float)):
                latency = int(obj["elapsed_ms"])

        events.append(
            {
                "type": "tool",
                "session_id": session_id,
                "tool_name": str(name),
                "ok": ok,
                "error": err,
                "latency_ms": latency,
            }
        )
    return events
