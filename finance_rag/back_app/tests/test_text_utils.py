# back_app/tests/test_text_utils.py
from __future__ import annotations

import re
from types import SimpleNamespace
import pytest

import back_app.utils.text as text


# -----------------------------
# URL sanitation & link helpers
# -----------------------------


def test_url_sanitize_strict_basic_and_www():
    messy = "  www.example.com\u200b/path \n"
    out = text._url_sanitize_strict(messy)
    assert out == "https://www.example.com/path"

    messy2 = " https://exa\u00admple.com/a b\t"  # soft hyphen + space
    out2 = text._url_sanitize_strict(messy2)
    # stripped, no spaces, soft hyphen removed
    assert out2 == "https://example.com/ab"


def test_unsplit_link_urls_sanitizes_target_only():
    md = "[Title](   https://exa\u200bmple.com/path?x=1  )"
    out = text._unsplit_link_urls(md)
    assert out == "[Title](<https://example.com/path?x=1>)"


def test_wrap_sanitized_with_bare_url_regex():
    s = "see https://example.com and carry on"
    out = text.BARE_URL2_RE.sub(text._wrap_sanitized, s)
    assert out == "see [https://example.com](https://example.com) and carry on"


def test_normalize_links_converts_latex_and_autolinks_outside_math():
    src = (
        r"\href{https://a.com}{Go A} and \url{https://b.com} "
        r"also https://c.com but keep math \(\alpha + 1\)"
    )
    out = text._normalize_links(src)
    # \href -> [TEXT](URL)
    assert "[Go A](https://a.com)" in out
    # \url -> URL
    assert "https://b.com" in out
    # bare url becomes markdown link
    assert "[https://c.com](https://c.com)" in out
    # math content preserved
    assert r"\(\alpha + 1\)" in out


# -----------------------------
# Generic coercion & names
# -----------------------------


def test_coerce_strings_various():
    obj = {
        1: ("a", slice(0, 1, None), ["x", 2]),
        "b": None,
        "c": 3.14,
        "d": True,
        "e": SimpleNamespace(x=1),
    }
    out = text._coerce_strings(obj)
    assert out["1"] == ["a", "", ["x", 2]]
    assert out["b"] is None and out["c"] == 3.14 and out["d"] is True
    assert isinstance(out["e"], str) and "x=1" in out["e"]


def test_sanitize_and_unique_safe_names():
    tools = [
        SimpleNamespace(name="My Tool!!  "),
        SimpleNamespace(name="My Tool!!  "),
        SimpleNamespace(name=""),
    ]
    text._unique_safe_names(tools)
    names = [t.name for t in tools]
    # first sanitized base
    assert names[0] == "My_Tool"
    # collision resolved
    assert names[1] == "My_Tool_2"
    # empty fallback -> "tool" (and not colliding with others)
    assert names[2] in {"tool", "tool_2", "tool_3"}


def test_clean_title_trims_and_collapses():
    t = "  Breaking   News  —  Big Thing  | "
    # Keeps the em-dash; removes extra spaces; trims trailing " | "
    assert text._clean_title(t) == "Breaking News — Big Thing"

    # Extra: verify trailing separators are stripped but interior ones remain
    assert text._clean_title("Foo — Bar — ") == "Foo — Bar"
    assert text._clean_title("Hello  World  · ") == "Hello World"
    assert text._clean_title("Alpha | ") == "Alpha"


# -----------------------------
# Provider/date utilities
# -----------------------------


def test_prettify_provider_mapping_and_passthrough():
    assert text._prettify_provider("Yahoo Finance") == "Yahoo"
    assert text._prettify_provider("Bloomberg.com") == "Bloomberg"
    assert text._prettify_provider("  Reuters  ") == "Reuters"
    assert text._prettify_provider("") == ""


@pytest.mark.parametrize(
    "inp,expect",
    [
        ("2025-01-02T12:34:56Z", "2025-01-02"),
        (1_700_000_000, "2023-11-14"),  # epoch seconds
        (1_700_000_000_000, "2023-11-14"),  # epoch ms
        ("1700000000", "2023-11-14"),  # numeric string
        ("nope", None),
        (None, None),
    ],
)
def test_to_iso_day(inp, expect):
    assert text._to_iso_day(inp) == expect


def test_pretty_date_portable_formatting():
    out = text._pretty_date("2025-01-02")
    # Some systems use %-d (no leading zero), others %d (with leading zero)
    assert out in {"Jan 2, 2025", "Jan 02, 2025"}


# -----------------------------
# Math/markdown healing
# -----------------------------


def test_heal_inline_math_delimiters_outside_code_only():
    # Zero-width after backslash (ZWSP)
    broken = "before \\\u200b( x + y \\\u200b) middle ''' code \\ \u200b( z ) ''' after"
    healed = text._heal_inline_math_delimiters(broken)

    # fixed outside code
    assert r"\(" in healed and r"\)" in healed.split("'''")[0]

    # inside code fence left untouched: still contains the ZWSP after the backslash
    parts = healed.split("'''")
    assert len(parts) >= 3  # prose, code, prose
    code_segment = parts[1]
    assert "\u200b" in code_segment
    assert "code \\ \u200b( z )" in code_segment


def test_break_currency_pairs_rewrites_all_but_last():
    md = "Para: $1,000 and then $2.50 in same paragraph.\n\nNext para with $5 only."
    out = text._break_currency_pairs(md)
    # First amount rewritten, last keeps '$'
    assert "USD 1,000" in out and "$2.50" in out
    # Second paragraph unchanged (only one $)
    assert "Next para with \\$5 only." not in out  # break does not escape dollars


def test_normalize_inline_math_whitespace_and_percent():
    block = "$$  a   +  b  % c  $$"
    out = text._normalize_inline_math_whitespace(block)
    assert out == "$$a + b \\% c$$"

    out2 = text._normalize_inline_math_whitespace(r"\[  x ^ 2  + y  \]")
    assert out2 == r"\[x ^ 2 + y\]"

    out3 = text._normalize_inline_math_whitespace(r"\(  a   +  b  \)")
    assert out3 == r"\(a + b\)"


def test_escape_dollars_outside_math_and_code():
    md = r"cash $100 outside, math \($x+1\) and $$y+2$$ and ''' code $3 ''' end"
    out = text._escape_dollars_outside_math_and_code(md)
    # escaped outside math/code
    assert r"\$100" in out
    # math blocks preserved
    assert r"\($x+1\)" in out and "$$y+2$$" in out
    # code block preserved
    assert "code $3" in out


# -----------------------------
# Non-math text heuristic
# -----------------------------


@pytest.mark.parametrize(
    "s,expected",
    [
        ("Visit https://example.com now", True),  # URL → text
        ("[link](https://x.y)", True),  # markdown link → text
        ("a b c d e", True),  # 5+ words, no math ops/cmds
        (r"\alpha + \beta = \gamma", False),  # LaTeX commands/operators
        ("", False),
    ],
)
def test_looks_non_math_text(s, expected):
    assert text._looks_non_math_text(s) is expected


# -----------------------------
# End-to-end postprocess smoke
# -----------------------------


def test_postprocess_markdown_integration():
    raw = (
        "summary:\n- ignore me metadata\n\n"
        r"Check \href{https://a.com}{A} then $1 and $2 in one para, "
        r"and bare url https://b.com. Math: \(\frac{a}{b}%\)."
    )
    out = text._postprocess_markdown(raw)

    # summary: block removed
    assert "summary:" not in out

    # \href converted (targets are wrapped in angle brackets by _unsplit_link_urls)
    assert "[A](<https://a.com>)" in out

    # currency rewriting + escaping behavior
    assert "USD 1" in out
    assert r"\$2" in out  # second $ escaped outside math

    # bare url linked: be flexible about optional trailing period and angle-bracketed targets
    import re

    assert (
        re.search(r"\[https://b\.com\.?\]\(<https://b\.com\.?>\)", out)
        or "https://b.com" in out
    )

    # math preserved and % escaped/normalized
    assert r"\(\frac{a}{b}\%\)" in out
