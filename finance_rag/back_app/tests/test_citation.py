# tests/test_citation.py
from __future__ import annotations
import importlib.util
import pathlib
import pytest

# --- Load the non-package module by file path ---
import back_app.utils.citations as citation
# ----------------------------
# Normalization helpers
# ----------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("I_Will_Teach_You_to_be_Rich_Ramit_Sethi.pdf", "i_will_teach_you_to_be_rich_ramit_sethi"),
        ("The Bogleheads’ Guide to Investing", "the_bogleheads_guide_to_investing"),
        ("Weird%27Title.PDF", "weird_title"),
        ("  Mixed—Punctuations!? ", "mixed_punctuations"),
        ("", ""),
    ],
)
def test__norm(raw, expected):
    assert citation._norm(raw) == expected


@pytest.mark.parametrize(
    "cit, title",
    [
        ("Author. *Some Book Title.* Pub, 2020.", "Some Book Title."),
        ("No stars here", None),
        ("*Wrapped* at start", "Wrapped"),
        ("prefix *Mid Title* suffix", "Mid Title"),
    ],
)
def test__citation_title(cit, title):
    assert citation._citation_title(cit) == title


@pytest.mark.parametrize(
    "s, looks",
    [
        ("file.pdf", True),
        ("FILE.PDF", True),
        ("https://x.com/a/b.pdf", True),
        ("https://x.com/a/b.pdf?token=123", True),
        ("https://x.com/a/b.pdf#p=3", True),
        ("https://x.com/a/b", False),
        ("notpdf.txt", False),
        ("", False),
        (None, False),
    ],
)
def test__looks_like_pdf(s, looks):
    assert citation._looks_like_pdf(s) is looks


# ----------------------------
# Books indices and builders
# ----------------------------

def test_books_indices_have_expected_keys():
    # The module defines BOOKS with three items; indices should include normalized keys.
    assert isinstance(citation.BOOKS, dict) and len(citation.BOOKS) >= 3
    # Sanity on derived indices:
    assert set(citation.BOOKS_INDEX.keys())  # not empty
    assert set(citation.BOOKS_TITLE_INDEX.keys())
    assert citation.BOOKS_TITLE_PATTERNS  # compiled regex list
    assert citation.BOOKS_BY_TITLE
    assert citation.BOOK_TITLE_PATTERNS


def test_book_source_from_filename_known_and_unknown():
    # Known PDF filename from BOOKS keys should return canonical citation
    known = next(iter(citation.BOOKS.keys()))
    src = citation.book_source_from_filename(known)
    assert src["type"] == "book"
    assert src["display"] == citation.BOOKS[known]
    assert src["href"] is None

    # Unknown filename becomes a titled book with display=basename (sans query/fragments normalized)
    unk = "My_Custom_Doc.pdf?x=1#frag"
    src2 = citation.book_source_from_filename(unk)
    assert src2["type"] == "book"
    assert src2["display"] == "My Custom Doc.pdf"
    assert src2["title"] == "My Custom Doc.pdf"


# ----------------------------
# normalize_source
# ----------------------------

@pytest.mark.parametrize(
    "item, expect",
    [
        ("https://example.com/a.pdf?dl=1", ("book", None)),       # PDF url → book
        ("https://example.com/post", ("web", "https://example.com/post")),  # plain URL → web
        ("Some free text", ("web", None)),                        # not URL/PDF → web, href None
    ],
)
def test_normalize_source_str(item, expect):
    out = citation.normalize_source(item)
    assert out["type"] == expect[0]
    if expect[1] is None:
        assert out.get("href") is None
    else:
        assert out.get("href") == expect[1]


def test_normalize_source_dict_pre_norm_web_and_pdf_and_title():
    # Already-normalized web with url (href should mirror url if href missing)
    d1 = {"type": "web", "display": "X", "url": "https://x.test"}
    out1 = citation.normalize_source(d1)
    assert out1["type"] == "web" and out1["href"] == "https://x.test"

    # Dict with PDF path → book
    d2 = {"source": "/docs/The_Dhandho_Investor_Mohnish_Pabrai.pdf"}
    out2 = citation.normalize_source(d2)
    assert out2["type"] == "book" and "Dhandho" in out2["display"]

    # Dict with url+title
    d3 = {"url": "https://news.site/a", "title": "Breaking News"}
    out3 = citation.normalize_source(d3)
    assert out3["type"] == "web" and out3["href"] == "https://news.site/a" and out3["display"] == "Breaking News"

    # Dict with only title → web with href None
    d4 = {"title": "Only A Title"}
    out4 = citation.normalize_source(d4)
    assert out4["type"] == "web" and out4["href"] is None and out4["display"] == "Only A Title"

    # Arbitrary dict → web display=str(dict)
    d5 = {"foo": 1}
    out5 = citation.normalize_source(d5)
    assert out5["type"] == "web" and out5["href"] is None and "foo" in out5["display"]


# ----------------------------
# dedupe_sources
# ----------------------------

def test_dedupe_sources_by_href_then_display_and_assigns_ids():
    items = [
        "https://site/a",                  # web by URL
        {"url": "https://site/a"},         # duplicate by same href
        "A free text",                     # web by display (no href)
        {"display": "A free text", "type": "web"},  # duplicate by display
        {"url": "https://site/b", "title": "B"},    # unique
    ]
    out = citation.dedupe_sources(items)
    # Expect 3 uniques: site/a, "A free text", site/b
    assert len(out) == 3
    # Ensure ids/n are 1-based and sequential
    assert [s["id"] for s in out] == [1, 2, 3]
    assert [s["n"] for s in out] == [1, 2, 3]


# ----------------------------
# tools_trace_to_sources
# ----------------------------

def test_tools_trace_to_sources_agent_and_mcp_and_ledger():
    traces = [
        # Agent-style normal tool
        {"tool": "web_search", "args": {"q": "tesla"}, "observation": "Found 10 results..."},
        # Agent-style MCP tool
        {"tool": "mcp:world_bank:get_indicator_for_country", "args": {"country_id": "CAN"}, "observation": {"value": 42}},
        # Ledger-style normal item
        {"name": "http_fetch", "ok": True, "duration_ms": 120, "meta": {"status": 200, "path": "/news"}},
        # Ledger-style MCP item
        {"name": "mcp:world_bank:get_countries", "ok": False, "duration_ms": 15, "meta": {"error": "boom"}},
    ]
    out = citation.tools_trace_to_sources(traces)
    # Four entries back
    assert len(out) == 4
    disp = [x["display"] for x in out]
    # Normal tool has the "🔧 name(" shape or "🔧 name"
    assert any(d.startswith("🔧 web_search") for d in disp)
    # MCP entries labeled MCP:world_bank/...
    assert any(d.startswith("MCP:world_bank/get_indicator_for_country") for d in disp)
    assert any(d.startswith("MCP:world_bank/get_countries") for d in disp)
    # Ledger normal has "🧰" and status suffix
    assert any(
        d.startswith("🧰 http_fetch")
        and " — ok" in d
        and "120ms" in d
        for d in disp
    )

# ----------------------------
# _filter_sources_to_text
# ----------------------------

def test_filter_sources_to_text_rules_and_dedupe():
    text = (
        "See the analysis at https://alpha.example.com/post and also check Beta News. "
        "Here is a markdown link to [Gamma](https://gamma.example.com/path)."
    )

    sources = [
        # Tool source — always kept
        {"type": "tool", "display": "🔧 web_search", "href": None},
        # Web source that appears as a bare URL
        {"type": "web", "display": "Alpha Example", "href": "https://alpha.example.com/post"},
        # Web source by host match (different path)
        {"type": "web", "display": "Gamma Site", "href": "https://gamma.example.com/another"},
        # Web source NOT in text at all
        {"type": "web", "display": "Delta Site", "href": "https://delta.example.com/x"},
        # Doc/book source kept (keep_docs_if_present=True)
        {"type": "book", "display": "Some Book"},
        {"type": "doc", "display": "Internal Doc"},
        # Duplicate (same href as Alpha)
        {"type": "web", "display": "Alpha Dup", "href": "https://alpha.example.com/post"},
    ]

    kept = citation._filter_sources_to_text(text, sources, keep_docs_if_present=True)
    # Expect: tools (1), alpha (by direct url), gamma (by host), docs (2), deduped alpha
    kinds = [s["type"] for s in kept]
    assert kinds.count("tool") == 1
    assert any(s.get("href") == "https://alpha.example.com/post" for s in kept)
    assert any(s.get("href") == "https://gamma.example.com/another" for s in kept)
    assert sum(1 for s in kept if s["type"] in {"book", "doc"}) == 2
    # Delta not present
    assert not any(s.get("href") == "https://delta.example.com/x" for s in kept)

    # If we set keep_docs_if_present=False, docs should only be kept if referenced by title/url
    kept2 = citation._filter_sources_to_text(text, sources, keep_docs_if_present=False)
    assert all(s["type"] != "book" for s in kept2)
    assert all(s["type"] != "doc" for s in kept2)


def test_filter_sources_to_text_title_keyword_match():
    # When there is no URL/host in text, title keywords can trigger a match.
    text = "An insightful breakdown of value investing methods for beginners."
    srcs = [
        {"type": "web", "display": "Value Investing Methods for Beginners", "href": None},
        {"type": "web", "display": "Unrelated Title", "href": None},
    ]
    kept = citation._filter_sources_to_text(text, srcs, keep_docs_if_present=False)
    assert len(kept) == 1
    assert kept[0]["display"].startswith("Value Investing")


# ----------------------------
# _had_rag_or_tools
# ----------------------------

def test_had_rag_or_tools_true_and_false():
    assert citation._had_rag_or_tools([]) is False
    # Placeholder model source should NOT count
    assert citation._had_rag_or_tools(
        [{"type": "model", "display": "Model answer (no tools used)"}]
    ) is False
    # Real web/tool/doc/book should count
    assert citation._had_rag_or_tools(
        [{"type": "web", "display": "https://example.com"}]
    ) is True
    assert citation._had_rag_or_tools(
        [{"type": "tool", "display": "🔧 web_search"}]
    ) is True
    assert citation._had_rag_or_tools(
        [{"type": "doc", "display": "Internal Doc"}]
    ) is True
    assert citation._had_rag_or_tools(
        [{"type": "book", "display": "Some Book"}]
    ) is True
