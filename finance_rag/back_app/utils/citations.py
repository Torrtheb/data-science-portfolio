from __future__ import annotations
import os
import re
import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

# ----------------------------
# Globals
# ----------------------------

SID_RE = re.compile(r"\[(S\d+|N\d+|T\d+)\]")

BOOKS: dict[str, str] = {
    "A_Random_Walk_Down_Wall_Street_B_Malkiel.pdf": "Malkiel, Burton G. *A Random Walk Down Wall Street.* Norton, 2019.",
    "Broke millennial takes on investing a beginners guide to leveling up your money by Lowry, Erin (z-lib.org).epub_text.pdf": "Lowry, Erin. *Broke Millennial Takes on Investing: A Beginner's Guide to Leveling Up Your Money.* TarcherPerigee, 2019.",
    "Broke_Millenial_Erin_Lowry.pdf": "Lowry, Erin. *Broke Millennial: Stop Scraping By and Get Your Financial Life Together.* TarcherPerigee, 2017.",
    "Broke_Millennial_Takes_on_Investing_Erin_Lowry.pdf": "Lowry, Erin. *Broke Millennial Takes on Investing: A Beginner's Guide to Leveling Up Your Money.* TarcherPerigee, 2019.",
    "Common_Sense_on_Mutual_Funds_N_Hoboken.pdf": "Bogle, John C. *Common Sense on Mutual Funds.* Wiley, 2010.",
    "Common_Stocks_and_Uncommon_Profits_and_Other_Writings_Philip_Fisher.pdf": "Fisher, Philip A. *Common Stocks and Uncommon Profits and Other Writings.* Wiley, 2003.",
    "Electronic_Commerce_The_Strategic_Perspective_Richard_Watson.pdf": "Watson, Richard T. *Electronic Commerce: The Strategic Perspective.* St. Lucie Press, 1997.",
    "Financial-Empowerment-Personal-Finance-for-Indigenous-and-Non-Indigenous-People-1730307500.pdf": "Anderson, Monique, et al. *Financial Empowerment: Personal Finance for Indigenous and Non-Indigenous People.* Indigenous Learning Co., 2020.",
    "Financial_Mathematics_ee397c84fa.pdf": "Unknown author. *Financial Mathematics.* n.p., n.d.",
    "Fooled_by_Randomness _Nassim_Taleb.pdf": "Taleb, Nassim Nicholas. *Fooled by Randomness: The Hidden Role of Chance in Life and in the Markets.* Random House, 2001.",
    "Get_Good_with_Money_Tiffany_Aliche.pdf": "Aliche, Tiffany. *Get Good with Money: Ten Simple Steps to Becoming Financially Whole.* Rodale, 2021.",
    "I_Will_Teach_You_to_be_Rich_Ramit_Sethi.pdf": "Sethi, Ramit. *I Will Teach You to Be Rich.* Workman Pub., 2009.",
    "Intermediate_Financial_Accounting_Vol1_G_Arnold_and_S_Kyle.pdf": "Arnold, Glenn, and Suzanne Kyle. *Intermediate Financial Accounting, Vol. 1.* n.p., 2018.",
    "Intermediate_Financial_Accounting_Vol2_G_Arnold_and_S_Kyle.pdf": "Arnold, Glenn, and Suzanne Kyle. *Intermediate Financial Accounting, Vol. 2.* n.p., 2018.",
    "Irrational_Exuberance_Robert_Shiller.pdf": "Shiller, Robert J. *Irrational Exuberance.* Princeton University Press, 2015.",
    "One_up_on_Wall_Street_Peter_Lynch.pdf": "Lynch, Peter. *One Up on Wall Street.* Simon & Schuster, 2000.",
    "Poor_Charlie’s_Almanack_Charles_Munger.pdf": "Munger, Charles T. *Poor Charlie's Almanack.* Donning Company, 2006.",
    "Security_Analysis_Benjamin_Graham_David_Dodd.pdf": "Graham, Benjamin, and David Dodd. *Security Analysis.* McGraw-Hill, 2008.",
    "Stocks for the Long Run 5_E_ The Definitive Guide to Financial Market Returns & Long-Term Investment Strategies - PDF Room.pdf": "Siegel, Jeremy J. *Stocks for the Long Run.* McGraw-Hill, 2014.",
    "The_Bogleheads'_Guide_to_Investing_Taylor_Larimore.pdf": "Larimore, Taylor, et al. *The Bogleheads' Guide to Investing.* Wiley, 2014.",
    "The_bogleheads_Guide_to_Retirement_Planning_Taylor_Larimore.pdf": "Larimore, Taylor, et al. *The Bogleheads' Guide to Retirement Planning.* Wiley, 2009.",
    "The_Dhandho_Investor_Mohnish_Pabraj.pdf": "Pabrai, Mohnish. *The Dhandho Investor: The Low-Risk Value Method to High Returns.* Wiley, 2007.",
    "The_Intelligent_Investor_Benjamin_Graham.pdf": "Graham, Benjamin. *The Intelligent Investor.* HarperBusiness, 2006.",
    "The_Misbehavior_of_Markets_Benoit_Mandelbrot_and_Richard_Hudson.pdf": "Mandelbrot, Benoit, and Richard L. Hudson. *The Misbehavior of Markets: A Fractal View of Financial Turbulence.* Basic Books, 2004.",
    "The_Psychology_of_Money_Marcus_Lancaster.pdf": "Housel, Morgan. *The Psychology of Money.* Harriman House, 2020.",
    "The_Total_Money_Makeover_Dave_Ramsey.pdf": "Ramsey, Dave. *The Total Money Makeover.* Thomas Nelson, 2013.",
    "Think_and_Grow_Rich_Napoleon_Hill.pdf": "Hill, Napoleon. *Think and Grow Rich.* The Ralston Society, 1937.",
    "security-analysis-benjamin-graham-6th-edition-pdf-february-24-2010-12-08-am-3-0-meg.pdf": "Graham, Benjamin, and David Dodd. *Security Analysis.* 6th ed., McGraw-Hill, 2008.",
}


@dataclass
class SourceEntry:
    """Canonical representation of a source entry."""

    id: str
    display: str
    href: Optional[str] = None
    meta: Optional[dict] = None
    snippet: Optional[str] = None


# ----------------------------
# Normalization helpers
# ----------------------------


def _norm(s: str) -> str:
    """
    Normalize a string for use as a stable key.

    Operations:
      - NFKD unicode normalization
      - Lowercasing
      - Replace curly apostrophes and '%27' with "'"
      - Strip trailing '.pdf'
      - Replace non-alphanumerics with underscores and trim

    Args:
        s: Raw string (filename/title/identifier).

    Returns:
        Normalized underscore-separated key.
    """

    s = unicodedata.normalize("NFKD", s or "")
    s = s.lower().replace("’", "'").replace("%27", "'")
    s = re.sub(r"\.(pdf)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _citation_title(cit: str) -> str | None:
    """
    Extract the title from a citation formatted with asterisks.

    Example:
        "Sethi, Ramit. *I Will Teach You to Be Rich.* ..." → "I Will Teach You to Be Rich"

    Args:
        cit: Citation text.

    Returns:
        Title between asterisks or None if not present.
    """
    m = re.search(r"\*([^*]+)\*", cit or "")
    return m.group(1).strip() if m else None


def _basename(p: str) -> str:
    """
    Safe wrapper around 'os.path.basename'.

    Args:
        p: Path or URL-like string.

    Returns:
        Basename portion of the path, or the input if extraction fails.
    """
    try:
        return os.path.basename(p)
    except Exception:
        return p


def _strip_url_tail(s: str) -> str:
    """
    Remove query string and fragment from a URL or path.

    Args:
        s: URL or path string.

    Returns:
        String without '?query' or '#fragment' suffixes.
    """
    return re.split(r"[?#]", s or "", maxsplit=1)[0]


def _looks_like_pdf(s: str) -> bool:
    """
    Determine whether a string points to a PDF resource.

    Behavior:
      - First strips query/fragment via '_strip_url_tail'
      - Checks for '.pdf' suffix case-insensitively

    Args:
        s: URL or path.

    Returns:
        True if it appears to reference a PDF, else False.
    """
    s2 = _strip_url_tail(s or "")
    return s2.lower().endswith(".pdf")


# ----------------------------
# Book citation indices & regex
# ----------------------------

BOOKS_INDEX: dict[str, str] = {_norm(k): v for k, v in BOOKS.items()}

BOOKS_TITLE_INDEX: dict[str, str] = {}
for fn, cit in (BOOKS or {}).items():
    title = _citation_title(cit)
    if not title:
        base = os.path.splitext(fn)[0].replace("_", " ").replace("-", " ")
        title = base
    BOOKS_TITLE_INDEX[_norm(title)] = cit

BOOKS_TITLE_PATTERNS: list[tuple[re.Pattern, str]] = []
for key_norm, cit in BOOKS_TITLE_INDEX.items():
    toks = [
        re.escape(tok)
        for tok in re.findall(r"[A-Za-z0-9']+", key_norm.replace("_", " "))
    ]
    if toks:
        pat = re.compile(r"\b" + r"\s+".join(toks) + r"\b", re.I)
        BOOKS_TITLE_PATTERNS.append((pat, cit))

BOOKS_BY_TITLE: dict[str, str] = {}
BOOK_TITLE_PATTERNS: list[tuple[re.Pattern, str]] = []
for fn, cit in (BOOKS or {}).items():
    title = _citation_title(cit)
    if not title:
        continue
    BOOKS_BY_TITLE[_norm(title)] = cit
    toks = re.findall(r"[A-Za-z0-9']+", title)
    if toks:
        pat = re.compile(r"\b" + r"\s+".join(map(re.escape, toks)) + r"\b", re.I)
        BOOK_TITLE_PATTERNS.append((pat, cit))


# ----------------------------
# Book source builders
# ----------------------------


def book_source_from_filename(path_or_name: str) -> dict[str, Any]:
    """
    Build a canonical "book" source dict from a filename or URL path.

    Resolution:
      - Normalizes the basename and looks up a known citation in BOOKS_INDEX.
      - If known, returns the curated citation string.
      - Otherwise, uses a prettified filename (sans path) as a display/title fallback.

    Args:
        path_or_name: Full path/URL or bare filename.

    Returns:
        Dict with keys: {type: "book", display, title, href: None}.
    """
    base = _basename(_strip_url_tail(path_or_name or ""))
    key = _norm(base)
    citation = BOOKS_INDEX.get(key)
    if citation:
        return {"type": "book", "display": citation, "title": citation, "href": None}
    title = base or "Document"
    title = title.replace("_", " ").replace("-", " ").replace("’", "'")
    title = re.sub(r"\s+", " ", title).strip()
    return {"type": "book", "display": title, "title": title, "href": None}


# ----------------------------
# Generic source normalization
# ----------------------------


def normalize_source(item: str | dict | Any) -> dict[str, Any]:
    """
    Normalize arbitrary inputs into a standard source dict for UI display.

    Input cases:
      - str:
          * If a PDF → book source (via 'book_source_from_filename')
          * If http(s) URL → {type:"web", display:url, href:url}
          * Else → {type:"web", display:text, href:None}
      - dict:
          * If already shaped and typed (book/web/tool/doc) → normalize 'href' for web
          * Else infer from 'source'/'file'/'path'/'url'/'title' fields
      - anything else:
          * Stringify as a web-like display with no href

    Args:
        item: String/dict/other describing a source.

    Returns:
        Canonical source dict with at least {type, display, href?}.
    """
    if isinstance(item, str):
        s = item.strip()
        if _looks_like_pdf(s):
            return book_source_from_filename(s)
        if s.startswith("http://") or s.startswith("https://"):
            return {"type": "web", "display": s, "href": s}
        return {"type": "web", "display": s, "href": None}

    if isinstance(item, dict):
        if item.get("type") in {"book", "web", "tool", "doc"} and (
            "display" in item or "title" in item or "url" in item
        ):
            if item.get("type") == "web":
                href = item.get("href")
                url = item.get("url")
                item = {**item, "href": href if href is not None else (url or None)}
            # If a page number is present and not already reflected in display,
            # append a simple " | Page X" suffix for human-friendly citations.
            page = item.get("page")
            if page not in (None, "", "?"):
                try:
                    p = int(page)
                except Exception:
                    p = page
                label = item.get("display") or item.get("title")
                if isinstance(label, str) and f"Page {p}" not in label:
                    item = {**item, "display": f"{label} | Page {p}"}
            return item

        src = item.get("source") or item.get("file") or item.get("path") or ""
        url = item.get("url")
        title = item.get("title")
        page = item.get("page")
        if isinstance(src, str) and _looks_like_pdf(src):
            return book_source_from_filename(src)
        if url:
            display = title or url
            if page not in (None, "", "?"):
                try:
                    p = int(page)
                except Exception:
                    p = page
                display = f"{display} | Page {p}"
            return {
                "type": "web",
                "display": display,
                "href": url,
                "title": title or None,
                "page": page,
            }
        if title:
            display = title
            if page not in (None, "", "?"):
                try:
                    p = int(page)
                except Exception:
                    p = page
                display = f"{display} | Page {p}"
            return {
                "type": "web",
                "display": display,
                "href": None,
                "title": title,
                "page": page,
            }
        return {"type": "web", "display": str(item), "href": None}

    return {"type": "web", "display": str(item), "href": None}


def dedupe_sources(items: list[Any]) -> list[dict[str, Any]]:
    """
    Deduplicate sources by 'href' (case-insensitive), falling back to 'display'.

    Side effects:
      - Ensures each returned item has 'id' and 'n' (1-based index).

    Args:
        items: Iterable of raw or normalized source items.

    Returns:
        New list of unique, normalized source dicts.
    """
    unique: list[dict] = []
    seen = set()
    for x in items or []:
        sx = normalize_source(x)
        key = (sx.get("href") or "").lower() or (sx.get("display") or "").lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(sx)
    for i, s in enumerate(unique, 1):
        s.setdefault("id", i)
        s.setdefault("n", i)
    return unique


# ----------------------------
# Tool trace → source mapping
# ----------------------------


def _preview160(x: Any) -> str:
    """
    Produce a safe, human-readable 160-character preview for any object.

    Formatting rules:
      - str/bytes: decode/trim
      - list/tuple/set/dict: JSON-encode then trim
      - fallback: 'str(x)' then trim

    Args:
        x: Arbitrary object.

    Returns:
        Short preview string (<=160 chars). Empty on failure.
    """
    try:
        if isinstance(x, str):
            return x[:160]
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")[:160]
        if isinstance(x, (list, tuple, set, dict)):
            return json.dumps(x, ensure_ascii=False)[:160]
        return str(x)[:160]
    except Exception:
        return ""


def _summarize_args(args: Any) -> str:
    """
    Compactly summarize dict-like arguments as 'k=v' pairs.

    Constraints:
      - Only includes simple scalars (str/int/float/bool)
      - Values longer than 32 chars are skipped
      - Non-dict inputs are stringified and truncated (<=64 chars)

    Args:
        args: Dict or arbitrary object.

    Returns:
        Comma-separated summary string.
    """
    try:
        if isinstance(args, dict):
            parts = []
            for k, v in args.items():
                if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 32:
                    parts.append(f"{k}={v}")
            return ", ".join(parts)
        return str(args)[:64]
    except Exception:
        return ""


def tools_trace_to_sources(tools: list[dict]) -> list[dict[str, Any]]:
    """
    Convert agent/ledger tool traces into displayable "source" objects.

    Output shape for each item:
        { "type": "tool", "display": str, "href": None, "meta": {...} }

    Rules:
      - Agent-style: uses 'tool', includes summarized args and observation preview.
      - Ledger-style: uses 'name', includes ok/duration/meta.
      - MCP tools are labeled prominently as 'MCP:<server>/<tool>'.

    Args:
        tools: List of trace dicts from agents or execution ledgers.

    Returns:
        List of source dicts suitable for UI "Sources" panel.
    """

    def _mcp_label(name: str) -> str | None:
        if not isinstance(name, str) or not name.startswith("mcp:"):
            return None
        parts = name.split(":")
        if len(parts) >= 3:
            return f"MCP:{parts[1]}/{parts[2]}"
        return f"MCP:{name}"

    out: list[dict] = []
    for t in tools or []:
        if "tool" in t:
            tool_name = t.get("tool") or ""
            mcp = _mcp_label(tool_name)
            if mcp:
                out.append(
                    {
                        "type": "tool",
                        "display": mcp,
                        "href": None,
                        "meta": {
                            "args": t.get("args"),
                            "observation_preview": _preview160(t.get("observation")),
                        },
                    }
                )
                continue

            sig = _summarize_args(t.get("args", {}))
            label = f"🔧 {tool_name}({sig})" if sig else f"🔧 {tool_name}"
            out.append(
                {
                    "type": "tool",
                    "display": label,
                    "href": None,
                    "meta": {
                        "args": t.get("args"),
                        "observation_preview": _preview160(t.get("observation")),
                    },
                }
            )
            continue
        if "name" in t:
            name = t.get("name") or ""
            mcp = _mcp_label(name)
            if mcp:
                out.append(
                    {
                        "type": "tool",
                        "display": mcp,
                        "href": None,
                        "meta": {
                            "meta": t.get("meta") or {},
                            "duration_ms": t.get("duration_ms"),
                            "ok": t.get("ok"),
                        },
                    }
                )
                continue

            status = "ok" if t.get("ok") else "error"
            dur = t.get("duration_ms")
            suffix = f" — {status}" + (f", {dur}ms" if isinstance(dur, int) else "")
            meta = t.get("meta") or {}
            sig = _summarize_args(meta)
            label = f"🧰 {name}({sig}){suffix}" if sig else f"🧰 {name}{suffix}"
            out.append(
                {
                    "type": "tool",
                    "display": label,
                    "href": None,
                    "meta": {"meta": meta, "duration_ms": dur, "ok": t.get("ok")},
                }
            )
    return out


# ----------------------------
# Source filtering
# ----------------------------


def _filter_sources_to_text(
    text: str,
    sources: list[dict],
    keep_docs_if_present: bool = True,
) -> list[dict[str, Any]]:
    """
    Filter a candidate source list to only those referenced by the response text.

    Matching strategy:
      - Always keep tool sources.
      - For web/news: keep if URL appears in text (linked or bare), or host matches,
        or enough display-title words appear in the body.
      - For docs/books:
          * If 'keep_docs_if_present' and any doc exists: keep all docs/books.
          * Else, apply the same matching rules as web/news.

    Args:
        text: Assistant reply markdown/text.
        sources: Candidate sources (already normalized or raw).
        keep_docs_if_present: If True, keep docs/books wholesale when any exist.

    Returns:
        Deduplicated list of sources likely referenced by 'text'.
    """
    from contextlib import suppress

    if not sources or not isinstance(sources, list):
        return []

    sources = [s for s in sources if (s.get("type") or "").lower() != "model"]
    if not sources:
        return []

    body = text or ""
    body_l = body.lower()

    md_link_re = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
    bare_url_re = re.compile(r"(?<!\]\()(?P<url>https?://[^\s)]+)")

    urls_in_text: set[str] = set()
    for m in md_link_re.finditer(body):
        urls_in_text.add(m.group(2).strip())
    for m in bare_url_re.finditer(body):
        urls_in_text.add(m.group("url").strip())

    hosts_in_text: set[str] = set()
    for u in urls_in_text:
        with suppress(Exception):
            p = urlparse(u)
            if p.hostname:
                hosts_in_text.add(p.hostname.lower())

    def host_of(href: str | None) -> str:
        if not href:
            return ""
        with suppress(Exception):
            p = urlparse(href)
            return (p.hostname or "").lower()
        return ""

    def appears_in_text(item: dict) -> bool:
        href = (item.get("href") or item.get("url") or "") or ""
        h = host_of(href)
        if href and any(href.strip().lower() == u.lower() for u in urls_in_text):
            return True
        if h and h in hosts_in_text:
            return True

        disp = (item.get("display") or item.get("title") or "") or ""
        disp_l = disp.lower().strip()
        if not disp_l:
            return False
        words = [w for w in re.findall(r"[a-z]{3,}", disp_l)]
        if not words:
            return False
        uniq = list(dict.fromkeys(words))
        hits = sum(1 for w in uniq if w in body_l)
        need = 1 if len(uniq) <= 3 else max(2, int(round(0.4 * len(uniq))))
        return hits >= need

    docs = [s for s in sources if (s.get("type") in {"doc", "book"})]
    rest = [s for s in sources if s.get("type") not in {"doc", "book"}]

    kept_tools = [s for s in rest if (s.get("type") == "tool")]
    kept_web = [s for s in rest if (s.get("type") != "tool") and appears_in_text(s)]

    kept_docs = (
        docs
        if (keep_docs_if_present and docs)
        else [s for s in docs if appears_in_text(s)]
    )

    out = kept_docs + kept_tools + kept_web

    try:
        out = dedupe_sources(out) or out
    except Exception:
        pass
    return out


def _had_rag_or_tools(sources: list[dict]) -> bool:
    """
    Determine whether any real (non-placeholder) RAG/tool sources are present.

    Considers types: {"tool", "web", "doc", "book"} and ignores
    the placeholder "Model answer (no tools used)".

    Args:
        sources: List of source dicts.

    Returns:
        True if at least one qualifying source exists, else False.
    """
    if not sources:
        return False
    for s in sources:
        t = (s.get("type") or s.get("kind") or "").lower()
        if t in {"tool", "web", "doc", "book"}:
            if s.get("display") != "Model answer (no tools used)":
                return True
    return False
