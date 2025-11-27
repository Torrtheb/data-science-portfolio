from __future__ import annotations
import os
import re
from datetime import datetime, timezone
from typing import Any, List, Optional
import unicodedata


# ---- Regexes for URL & LaTeX forms ------------------------------------------
HREF_RE = re.compile(r"\\href\{([^}]+)\}\{([^}]+)\}")
URL_CMD_RE = re.compile(r"\\url\{([^}]+)\}")
TEXTTT_URL_RE = re.compile(r"\\texttt\{(https?://[^}]+)\}")
INLINE_MATH_URL_RE = re.compile(r"\\\((https?://[^)]+)\\\)")
DISPLAY_MATH_URL_RE = re.compile(r"\\\[(https?://[^\]]+)\\\]")
DOLLAR_URL_RE = re.compile(r"\$(https?://[^$]+)\$")
BARE_URL_RE = re.compile(r"(?<!\]\()(?P<u>https?://[^\s)]+)")
BARE_URL2_RE = re.compile(r"(?<!\]\()(?P<url>https?://[^\s)]+)")
_ESCAPED_BRACKETS = re.compile(r"\\\[\s*([\s\S]*?)\s*\\\]")
_ESCAPED_PARENS = re.compile(r"\\\(\s*([\s\S]*?)\s*\\\)")


# ---- Unicode spaces & invisibles -------------------------------------------

_UWS_CHARS = (
    "\u00a0"  # NBSP
    "\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a"  # EN/EM/THIN/HAIR/etc
    "\u202f"  # NARROW NBSP
    "\u205f"  # MMSP
    "\u3000"  # IDEOGRAPHIC SPACE
    "\u200b\u200c\u200d\u2060"  # ZWSP/ZWNJ/ZWJ/WORD JOINER
)
_UWS_CLASS = f"[{_UWS_CHARS}]"
_WS_OR_UWS = rf"(?:\s|{_UWS_CLASS})"
SUMMARY_BLOCK_RE = re.compile(r"(?ims)^\s*summary:\s*\n(?:(?!^\s*$).*\n)*")

# -- Math detection & currency heuristic --------------------------------------
_INLINE_MATH_NO_CURRENCY = rf"(?<!\\)\$(?!{_WS_OR_UWS}*[+\-]?\d)[\s\S]+?(?<!\\)\$"
_MATH_ANY_RE = re.compile(
    rf"(?:\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|{_INLINE_MATH_NO_CURRENCY}|\\\([\s\S]*?\\\))"
)
INLINE_OR_DISPLAY_MATH_RE = re.compile(
    r"(?:\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))"
)
_CURRENCY_RE = re.compile(rf"(?<!\\)\$({_WS_OR_UWS}*[+\-]?\d[\d,]*(?:\.\d+)?)")

# -- Healing for zero-width characters after backslashes ----------------------
_ZW_AFTER_BACKSLASH_RE = re.compile(r"\\[\u200B\u200C\u200D\u2060]+([()\[\]])")

# -- URL sanitation ------------------------------------------------------------
_BAD_URL_CHARS_RE = re.compile(
    r"[\s\u00A0\u202F\u00AD\u200B-\u200F\u2060-\u206F\uFEFF\u2028\u2029]"
)
_WHITELIST_URL_RE = re.compile(r"[^A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]")
NEWS_LINKS = os.getenv("NEWS_LINKS", "1") != "0"
_LINK_URL_RE = re.compile(r"\]\(\s*([^)]+?)\s*\)")


# ---------------------------------
# URL sanitation & normalization
# ---------------------------------


def _url_sanitize_strict(u: Optional[str]) -> str:
    """
    Aggressive URL sanitizer.

    Steps:
      1) Unicode-normalize (NFKC) and trim.
      2) Remove whitespace/invisible characters.
      3) Strip any non-RFC3986-safe characters (via whitelist).
      4) If the result starts with 'www.', prefix 'https://'.

    Args:
        u: Raw URL string (possibly messy or pasted from rich text).

    Returns:
        Cleaned URL string, or empty string if 'u' is falsy.
    """
    if not u:
        return ""
    s = unicodedata.normalize("NFKC", str(u).strip())
    s = _BAD_URL_CHARS_RE.sub("", s)
    s = _WHITELIST_URL_RE.sub("", s)
    if s.startswith("www."):
        s = "https://" + s
    return s


def _unsplit_link_urls(md: str) -> str:
    """
    Sanitize the **URL portion** inside Markdown links while preserving link text.

    Args:
        md: Markdown text.

    Returns:
        Markdown with normalized link targets using '_url_sanitize_strict'.
    """

    if not md:
        return md

    def _fix(m: re.Match) -> str:
        return f"](<{_url_sanitize_strict(m.group(1))}>)"

    return _LINK_URL_RE.sub(_fix, md)


def _wrap_sanitized(m: re.Match) -> str:
    """
    Wrap a **bare URL** match as a Markdown link with itself as the label, after
    running it through '_url_sanitize_strict'.

    This is used when auto-linking outside math/code:
        'https://example.com' → '[https://example.com](https://example.com)'

    Args:
        m: Regex match object with a named group '"url"'.

    Returns:
        Markdown link string using the sanitized URL for both label and target.
    """
    u = _url_sanitize_strict(m.group("url"))
    return f"[{u}]({u})"


def _normalize_links(text: str) -> str:
    """
    Normalize and harden links across mixed content (Markdown + LaTeX).

    Pipeline:
      1) If a '\\[...\\]' or '\\(...\\)' segment looks like *plain text*, unescape
         to literal '[...]' / '(...)' (prevents LaTeX interpretation).
      2) Convert LaTeX link forms to Markdown/plain:
           - '\\href{URL}{TEXT}' → '[TEXT](URL)'
           - '\\url{URL}' → 'URL'
           - '\\texttt{http...}' → 'http...'
           - URLs inside inline/display math forms → the raw URL
      3) Auto-link **bare URLs** outside math.
      4) Inside fenced code blocks, prevent accidental Markdown linking by
         inserting a space: '"]( → "] ('.

    Args:
        text: Input string that may contain Markdown/LaTeX.

    Returns:
        String with links normalized to robust Markdown.
    """
    if not text:
        return text
    s = text

    def _unescape_if_non_math(body: str, brackets: bool) -> str:
        return f"[{body}]" if brackets else f"({body})"

    s = _ESCAPED_BRACKETS.sub(
        lambda m: (
            _unescape_if_non_math(m.group(1), brackets=True)
            if _looks_non_math_text(m.group(1))
            else m.group(0)
        ),
        s,
    )
    s = _ESCAPED_PARENS.sub(
        lambda m: (
            _unescape_if_non_math(m.group(1), brackets=False)
            if _looks_non_math_text(m.group(1))
            else m.group(0)
        ),
        s,
    )

    s = HREF_RE.sub(r"[\2](\1)", s)
    s = URL_CMD_RE.sub(r"\1", s)
    s = TEXTTT_URL_RE.sub(r"\1", s)
    s = INLINE_MATH_URL_RE.sub(r"\1", s)
    s = DISPLAY_MATH_URL_RE.sub(r"\1", s)
    s = DOLLAR_URL_RE.sub(r"\1", s)

    chunks: List[str] = []
    last = 0
    for m in _MATH_ANY_RE.finditer(s):
        before = s[last : m.start()]
        before = BARE_URL2_RE.sub(_wrap_sanitized, before)
        chunks.append(before)
        chunks.append(m.group(0))
        last = m.end()
    tail = s[last:]
    tail = BARE_URL2_RE.sub(_wrap_sanitized, tail)
    chunks.append(tail)
    s = "".join(chunks)

    blocks = s.split("'''")
    for i in range(1, len(blocks), 2):
        blocks[i] = blocks[i].replace("](", "] (")
    s = "'''".join(blocks)

    return s


# ------------------------------------------------------------------------------
# Generic text formatting helpers
# ------------------------------------------------------------------------------


def _coerce_strings(obj: Any) -> Any:
    """
    Recursively coerce an arbitrary JSON-ish structure into a UI/JSON-friendly
    form.

    Rules:
      - Drop Python 'slice(...)' objects (return empty string).
      - Keep 'str' as is.
      - Convert tuples → lists and recurse into sequences.
      - Ensure dict keys are 'str' and recurse into values.
      - Preserve 'None', numbers, and booleans.
      - For unknown objects, return 'str(obj)'.

    Args:
        obj: Any nested structure.

    Returns:
        A JSON-serializable structure composed of primitives, lists, and dicts.
    """
    if isinstance(obj, slice):
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return [_coerce_strings(x) for x in obj]
    if isinstance(obj, tuple):
        return [_coerce_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _coerce_strings(v) for k, v in obj.items()}
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    return str(obj)


def _sanitize(name: str) -> str:
    """
    Sanitize an arbitrary tool/name to a safe identifier.

    Process:
      - Replace disallowed chars with '_' (allowed: '[A-Za-z0-9_-]').
      - Collapse multiple '_' into one.
      - Trim leading/trailing '_'.
      - Fallback to ''tool'' if empty.
      - Truncate to max 64 chars.

    Args:
        name: Raw name.

    Returns:
        A safe, compact identifier.
    """
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name or "")
    s = re.sub(r"_+", "_", s).strip("_") or "tool"
    return s[:64]


def _unique_safe_names(tools: List[Any]) -> List[Any]:
    """
    Ensure every tool in the list has a **unique** and **sanitized** '.name'.

    Mutates the tool objects in place:
      - Runs '_sanitize' on 't.name'.
      - Appends '_2', '_3', ... to resolve collisions.
      - Truncates to 64 chars.

    Args:
        tools: Iterable of objects with a '.name' attribute.

    Returns:
        The same list, after mutation for unique names.
    """
    used: set[str] = set()
    for t in tools:
        base = _sanitize(getattr(t, "name", "") or "tool")
        name = base
        i = 2
        while name in used:
            name = f"{base}_{i}"[:64]
            i += 1
        t.name = name
        used.add(name)
    return tools


def _clean_title(t: str) -> str:
    """
    Lightly tidy a title or headline, **without** changing case or meaning.

    Operations:
      - Collapse all whitespace to single spaces.
      - Strip trailing separators/spaces (e.g., '·', '-', '—', '|').

    Args:
        t: Raw title.

    Returns:
        Cleaned title string.
    """
    t = " ".join((t or "").split())
    t = t.strip(" \t\r\n·|-—–")
    return t


# ------------------------------------------------------------------------------
# News/provider presentation helpers
# ------------------------------------------------------------------------------


def _prettify_provider(src: str) -> str:
    """
    Normalize common news provider names to a short, consistent label.

    Known mappings (extend as needed):
        - "The Wall Street Journal" → "WSJ"
        - "Yahoo Finance"/"Yahoo! Finance" → "Yahoo"
        - "Financial Times" → "FT"
        - "CNBC.com" → "CNBC"
        - "Bloomberg.com" → "Bloomberg"

    Args:
        src: Provider string from an API.

    Returns:
        A normalized provider label, or the input trimmed if no mapping exists.
    """
    if not src:
        return ""
    s = src.strip()
    mapping = {
        "Yahoo Finance": "Yahoo",
        "Yahoo! Finance": "Yahoo",
        "The Wall Street Journal": "WSJ",
        "Wall Street Journal": "WSJ",
        "Financial Times": "FT",
        "CNBC.com": "CNBC",
        "Bloomberg.com": "Bloomberg",
        "Barron's": "Barron's",
        "MarketWatch": "MarketWatch",
        "The Verge": "The Verge",
        "Reuters": "Reuters",
    }
    return mapping.get(s, s)


def _to_iso_day(when_val: Any) -> Optional[str]:
    """
    Normalize various time hints to an ISO date string ('YYYY-MM-DD').

    Accepts:
      - ISO strings like ''2025-08-29T12:34:56Z'' (returns first 10 chars).
      - Integers/floats (epoch seconds or milliseconds).
      - Numeric strings (same as above).

    Args:
        when_val: Any of the supported input types.

    Returns:
        ISO date ('YYYY-MM-DD') or 'None' if parsing fails.
    """
    if when_val is None or when_val == "":
        return None

    if isinstance(when_val, str):
        s = when_val.strip()
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if s.isdigit():
            try:
                iv = int(s)
                if iv > 10_000_000_000:
                    iv = iv / 1000.0
                return datetime.fromtimestamp(iv, tz=timezone.utc).date().isoformat()
            except Exception:
                return None

    if isinstance(when_val, (int, float)):
        iv = int(when_val)
        if iv > 10_000_000_000:
            iv = iv / 1000.0
        try:
            return datetime.fromtimestamp(iv, tz=timezone.utc).date().isoformat()
        except Exception:
            return None

    return None


def _pretty_date(iso_day: Optional[str]) -> str:
    """
    Convert an ISO day ('YYYY-MM-DD') to a human label like ''Aug 29, 2025''.

    Uses a portable fallback for systems that don’t support '%-d'.

    Args:
        iso_day: ISO date string (first 10 chars considered).

    Returns:
        Pretty date string, or empty string on failure.
    """
    if not iso_day or not isinstance(iso_day, str):
        return ""
    try:
        d = datetime.strptime(iso_day[:10], "%Y-%m-%d").date()
        return d.strftime("%b %-d, %Y")
    except Exception:
        try:
            return datetime.strptime(iso_day[:10], "%Y-%m-%d").strftime("%b %d, %Y")
        except Exception:
            return ""


# ------------------------------------------------------------------------------
# Markdown/text normalization for chat display
# ------------------------------------------------------------------------------
def _heal_inline_math_delimiters(md: str) -> str:
    """
    Heal LaTeX inline/display math delimiters that were broken by **zero-width**
    characters inserted after the backslash.

    What it fixes:
        - Turns '\\​(' back into '\\(' and '\\​)' back into '\\)'.
        - Same for display '\\[' / '\\]'.
        - Operates **only outside** fenced code blocks.

    Args:
        md: Markdown text.

    Returns:
        The input with invisible chars removed after LaTeX backslashes in
        non-code regions.
    """
    if not md:
        return md
    parts = md.split("'''")
    for i in range(0, len(parts), 2):
        parts[i] = _ZW_AFTER_BACKSLASH_RE.sub(r"\\\1", parts[i])
    return "'''".join(parts)


def _break_currency_pairs(md: str) -> str:
    """
    Reduce accidental '$...$' inline-math by **disambiguating multiple currency
    amounts** in the same paragraph.

    Strategy:
        - In non-code regions, if a paragraph contains **2+ currency tokens**
          (e.g., '$1,000 ... $1,232.93'), rewrite **all but the last** as
          'USD <amount>'. This ensures at most **one** literal dollar sign per
          paragraph, which blocks '$...$' math spans.

    Args:
        md: Markdown text.

    Returns:
        Markdown with earlier currency amounts rewritten as 'USD <amount>' in
        paragraphs that had multiple '$'.
    """
    if not md:
        return md

    parts = md.split("'''")
    for i in range(0, len(parts), 2):
        prose = parts[i]
        chunks = re.split(r"(\n\s*\n)", prose)

        for j in range(0, len(chunks), 2):
            para = chunks[j]
            if not para.strip():
                continue

            matches = list(_CURRENCY_RE.finditer(para))
            if len(matches) < 2:
                continue

            out = []
            cursor = 0
            for k, m in enumerate(matches):
                out.append(para[cursor : m.start()])
                amt = m.group(1)
                amt = re.sub(rf"^{_WS_OR_UWS}*", "", amt)

                if k < len(matches) - 1:
                    out.append(f"USD {amt}")
                else:
                    out.append(para[m.start() : m.end()])
                cursor = m.end()
            out.append(para[cursor:])
            chunks[j] = "".join(out)

        parts[i] = "".join(chunks)

    return "'''".join(parts)


def _normalize_inline_math_whitespace(block: str) -> str:
    """
    Normalize explicit LaTeX math spans by collapsing internal whitespace and
    escaping '%' (LaTeX requires '\\%' inside math).

    Supported explicit delimiters:
        - Inline: '\\(...\\)'
        - Display: '$$...$$' and '\\[...\\]'

    Args:
        block: A single math block (already identified), or arbitrary text.

    Returns:
        The same block, but with dense spacing and safe '%' escaping. Non-math
        inputs are returned unchanged.
    """
    if not block or len(block) < 2:
        return block

    if (block.startswith("$$") and block.endswith("$$")) or (
        block.startswith("\\[") and block.endswith("\\]")
    ):
        inner = block[2:-2] if block.startswith("\\[") else block[2:-2]
        inner = re.sub(r"\s+", " ", inner).strip()
        inner = re.sub(r"(?<!\\)%", r"\\%", inner)
        return f"\\[{inner}\\]" if block.startswith("\\[") else f"$${inner}$$"

    if block.startswith("\\(") and block.endswith("\\)"):
        inner = re.sub(r"\s+", " ", block[2:-2]).strip()
        inner = re.sub(r"(?<!\\)%", r"\\%", inner)
        return f"\\({inner}\\)"

    return block


def _escape_dollars_outside_math_and_code(md: str) -> str:
    """
    Escape **all** '$' characters in **non-code** regions that are **not inside**
    explicit math blocks, to eliminate '$...$' inline math entirely.

    Keeps:
        - Explicit math blocks '$$...$$', '\\[...\\]', '\\(...\\)' untouched.
        - Code blocks fenced by triple backticks untouched.

    Args:
        md: Markdown text.

    Returns:
        Markdown where currency '$' becomes '\\$' outside code/math.
    """
    if not md:
        return md

    parts = md.split("'''")
    for i in range(0, len(parts), 2):
        block = parts[i]
        holes: list[str] = []

        def mask(m: re.Match) -> str:
            holes.append(m.group(0))
            return f"__MB_{len(holes)-1}__"

        holed = INLINE_OR_DISPLAY_MATH_RE.sub(mask, block)

        holed = re.sub(r"(?<!\\)\$", r"\\$", holed)

        for idx, mb in enumerate(holes):
            holed = holed.replace(f"__MB_{idx}__", mb)

        parts[i] = holed

    return "'''".join(parts)


def _looks_non_math_text(s: str) -> bool:
    """
    Heuristic to decide whether a string looks like **ordinary text** (headline/
    paragraph) rather than LaTeX content.

    Signals for *text*:
      - Contains a URL or Markdown link.
      - Has ≥5 “words” and **lacks** LaTeX commands (e.g., '\\alpha') and
        math operators like '= ^ _ { }'.

    Args:
        s: Candidate string.

    Returns:
        True if it looks like normal text; False otherwise.
    """
    body = (s or "").strip()
    if not body:
        return False
    if re.search(r"(https?://|www\.)", body, re.I):
        return True
    if re.search(r"\[[^\]]+\]\([^)]+\)", body):
        return True
    has_cmd = re.search(r"\\[A-Za-z]+", body) is not None
    has_ops = re.search(r"[=^_{}]", body) is not None
    word_count = len(re.findall(r"\b[\w’']+\b", body))
    return (word_count >= 5) and (not has_cmd) and (not has_ops)


def _postprocess_markdown(s: str) -> str:
    """
    Final, **order-sensitive** Markdown normalization pass for chat display.

    Steps (outside fenced code blocks unless stated otherwise):
      1) Heal zero-width chars after LaTeX backslashes.
      2) Disambiguate multiple currency '$' in a paragraph → 'USD <amount>' except last.
      3) Escape all remaining '$' outside explicit math and code.
      4) Temporarily extract explicit math blocks, then:
          - Remove stray backslashes before digits (common paste artifact).
          - Convert LaTeX link forms and auto-link bare URLs.
          - Normalize ']( )' spacing.
      5) Light structure normalization:
          - Insert newlines before headings/bullets/ordered items.
          - Collapse excessive blank lines.
          - Merge wrapped lines in plain paragraphs.
      6) Remove leading 'summary:' metadata blocks if present.
      7) Restore math blocks after normalizing their whitespace and '%'.
      8) Run link normalization and sanitize link targets.

    Args:
        s: Raw Markdown string from an LLM/tool.

    Returns:
        Clean, stable Markdown suitable for your chat UI.
    """
    if not s:
        return s

    parts = s.split("'''")
    for i in range(0, len(parts), 2):
        parts[i] = _heal_inline_math_delimiters(parts[i])

    for i in range(0, len(parts), 2):
        parts[i] = _break_currency_pairs(parts[i])

    for i in range(0, len(parts), 2):
        parts[i] = _escape_dollars_outside_math_and_code(parts[i])

    math_store: List[str] = []

    def _math_preserve(m: re.Match) -> str:
        idx = len(math_store)
        math_store.append(m.group(0))
        return f"__MATH_BLOCK_{idx}__"

    def _math_restore(text: str) -> str:
        for i, block in enumerate(math_store):
            fixed = _normalize_inline_math_whitespace(block)
            text = text.replace(f"__MATH_BLOCK_{i}__", fixed)
        return text

    for i in range(0, len(parts), 2):
        parts[i] = INLINE_OR_DISPLAY_MATH_RE.sub(_math_preserve, parts[i])

    for i in range(0, len(parts), 2):
        seg = parts[i]

        seg = re.sub(r"\\(?=\d)", "", seg)

        parts[i] = seg

    HREF_RE = re.compile(r"\\href\{([^}]+)\}\{([^}]+)\}")
    URL_CMD_RE = re.compile(r"\\url\{([^}]+)\}")
    TEXTTT_URL_RE = re.compile(r"\\texttt\{(https?://[^}]+)\}")
    INLINE_MATH_URL_RE = re.compile(r"\\\((https?://[^)]+)\\\)")
    DISPLAY_MATH_URL_RE = re.compile(r"\\\[(https?://[^\]]+)\\\]")
    BARE_URL_RE = re.compile(r"(?<!\]\()(?P<u>https?://[^\s)]+)")

    for i in range(0, len(parts), 2):
        seg = parts[i]
        seg = HREF_RE.sub(r"[\2](\1)", seg)
        seg = URL_CMD_RE.sub(r"\1", seg)
        seg = TEXTTT_URL_RE.sub(r"\1", seg)
        seg = INLINE_MATH_URL_RE.sub(r"\1", seg)
        seg = DISPLAY_MATH_URL_RE.sub(r"\1", seg)
        seg = BARE_URL_RE.sub(r"[\g<u>](\g<u>)", seg)
        seg = re.sub(r"\]\s+\(", "](", seg)
        parts[i] = seg

    HEADING_OR_STEP_RX = re.compile(
        r"([^\n])\s+((?:#{1,6}\s+)|(?:\d{1,3}[.)]\s+)|(?:[-*]\s+))"
    )

    STRUCTURE_LINE_RX = re.compile(
        r"^(#{1,6}\s+|[-*+]\s+|\d{1,3}[.)]\s+|>\s+|\|\s)", re.M
    )

    for i in range(0, len(parts), 2):
        seg = parts[i]
        seg = HEADING_OR_STEP_RX.sub(r"\1\n\2", seg)
        seg = re.sub(r"\n{3,}", "\n\n", seg)

        lines = seg.split("\n")
        out: List[str] = []
        for ln in lines:
            if not ln.strip():
                out.append("")
                continue
            if STRUCTURE_LINE_RX.match(ln):
                out.append(ln.rstrip())
                continue
            if (
                out
                and out[-1]
                and not STRUCTURE_LINE_RX.match(out[-1])
                and out[-1].strip()
            ):
                out[-1] = out[-1].rstrip() + " " + ln.strip()
            else:
                out.append(ln.strip())
        seg = "\n".join(out)
        seg = re.sub(r"\n{3,}", "\n\n", seg)
        seg = re.sub(r"\n[ \t]*\n+", "\n\n", seg)
        parts[i] = seg
    for i in range(0, len(parts), 2):
        parts[i] = SUMMARY_BLOCK_RE.sub("", parts[i]).strip()

    for i in range(0, len(parts), 2):
        parts[i] = _math_restore(parts[i])

    s = "'''".join(parts).strip()

    s = _normalize_links(s)
    s = _unsplit_link_urls(s)
    return s
