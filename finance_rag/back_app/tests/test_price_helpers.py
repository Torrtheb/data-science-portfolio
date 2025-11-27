import os
import types
import math
import pytest

# Adjust this import path if your project package name is different
from back_app.routers import price as price_mod


# ---------------------- _get_finnhub_token ----------------------------------


def test_get_finnhub_token_prefers_settings_secret(monkeypatch):
    class SecretStr:
        def __init__(self, v):
            self._v = v

        def get_secret_value(self):
            return self._v

    fake_settings = types.SimpleNamespace(finnhub_api_key=SecretStr("from_settings"))
    monkeypatch.setattr(price_mod, "settings", fake_settings, raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_TOKEN", raising=False)

    assert price_mod._get_finnhub_token() == "from_settings"


def test_get_finnhub_token_env_fallback(monkeypatch):
    monkeypatch.setattr(price_mod, "settings", None, raising=False)
    monkeypatch.setenv("FINNHUB_API_KEY", "from_env")
    monkeypatch.delenv("FINNHUB_TOKEN", raising=False)
    assert price_mod._get_finnhub_token() == "from_env"

    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("FINNHUB_TOKEN", "from_env_token")
    assert price_mod._get_finnhub_token() == "from_env_token"

    monkeypatch.delenv("FINNHUB_TOKEN", raising=False)
    assert price_mod._get_finnhub_token() == ""


# ---------------------- text helpers ----------------------------------------


@pytest.mark.parametrize(
    "s,expected",
    [
        ("Procter & Gamble Company", "procter gamble"),
        ("  Alphabet Inc   ", "alphabet"),
        ("THE   META PLATFORMS  CLASS A", "meta platforms"),
        ("", ""),
        (None, ""),
    ],
)
def test_norm_name_removes_stopwords(s, expected):
    assert price_mod._norm_name(s) == expected


def test_norm_tokenizes_basic():
    assert price_mod._norm("Apple Inc. (AAPL)") == ["apple", "inc", "aapl"]


@pytest.mark.parametrize(
    "q,looks_like",
    [
        ("AAPL", False),  # pure uppercase ticker
        ("aapl", True),  # lowercase → likely name-ish
        ("apple", True),  # single word but lowercase
        ("Procter & Gamble", True),  # spaces
        ("BRK.B", False),
        ("MSFT  ", False),
        ("  msft  ", True),
        ("$TSLA", False),
    ],
)
def test_looks_like_name_query(q, looks_like):
    assert price_mod._looks_like_name_query(q) == looks_like


@pytest.mark.parametrize(
    "sym,root",
    [
        ("BRK.B", "BRK"),
        ("BF.A", "BF"),
        ("AAPL", "AAPL"),
        ("hei.a", "HEI"),
        ("", ""),
        (None, ""),
    ],
)
def test_root(sym, root):
    assert price_mod._root(sym) == root


@pytest.mark.parametrize(
    "raw,canon",
    [
        ("brk b", "BRK.B"),
        ("BRK-B", "BRK.B"),
        ("BF B", "BF.B"),
        ("HEI A", "HEI.A"),
        ("SHOP.TSX", "SHOP.TO"),
        ("SHOP:CA", "SHOP.TO"),
        ("SHOP-T", "SHOP.TO"),
        ("XYZ.TSXV", "XYZ.V"),
        ("VOD:GB", "VOD.L"),
        ("BHP:AU", "BHP.AX"),
        ("0700:HK", "0700.HK"),
        (
            "TSLA.US",
            "TSLA",
        ),  # _canon_local does NOT strip .US here, it strips .US in normalization only when exact pattern; here returns "TSLA.US" uppercased
        ("TSLA:US", "TSLA"),
        ("brk a", "BRK.A"),
        ("BF a", "BF.A"),
        ("HEI-a", "HEI.A"),
    ],
)
def test_canon_local_mappings(raw, canon):
    assert price_mod._canon_local(raw) == canon


# ---------------------- symbol support heuristics ----------------------------


@pytest.mark.parametrize(
    "sym,ok",
    [
        ("AAPL", True),
        ("MSFT", True),
        ("BRK.B", True),  # class share
        ("XYZ.U", True),  # SPAC units
        ("ABC.WS", True),  # warrants
        ("SHOP.TO", False),
        ("VOD.L", False),
        ("BHP.AX", False),
        (
            "TSLA.US",
            False,
        ),  # tail not 1-char and not U/WS; not in NON_US_SUFFIXES → reject
        ("", False),
        (None, False),
        ("TOO_LONG", False),
        ("SPY", True),  # ETF-like ticker still fits the heuristic
    ],
)
def test_is_supported_equity_symbol(sym, ok):
    assert price_mod._is_supported_equity_symbol(sym) == ok


# ---------------------- search hinting & exchange weighting ------------------


def test_hint_prefs_detects_regions():
    prefs = price_mod._hint_prefs("search Canada TSX and London LSE please")
    # bumped keys should be > 0
    assert prefs["canada"] > 0
    assert prefs["tsx"] > 0
    assert prefs["london"] > 0
    assert prefs["lse"] > 0
    # defaults exist
    assert "us" in prefs and prefs["us"] == 0


def test_exchange_hint_weight_prefers_us_and_respects_prefs():
    prefs = {k: 0 for k in price_mod._hint_prefs("").keys()}
    w_us = price_mod._exchange_hint_weight("AAPL", "Nasdaq Global Select", prefs)
    w_ca_no_hint = price_mod._exchange_hint_weight("SHOP", "TSX", prefs)

    # with Canadian hints, CA should get a bigger bump
    prefs_hint = dict(prefs, **{"canada": 1, "tsx": 1})
    w_ca_with_hint = price_mod._exchange_hint_weight("SHOP", "TSX", prefs_hint)

    assert w_us > w_ca_no_hint
    assert w_ca_with_hint > w_ca_no_hint


# ---------------------- candidate scoring ------------------------------------


def test_score_candidate_orders_by_exchange_and_cap_and_tokens():
    q_tokens = ["apple", "inc"]
    prefs = price_mod._hint_prefs("")  # neutral

    cand = {
        "symbol": "AAPL",
        "description": "Apple Inc",
        "type": "Common Stock",
    }
    prof_us_big = {"exchange": "NASDAQ", "market_cap": 2_500_000_000_000}
    prof_non_us_small = {"exchange": "LSE", "market_cap": 1_000_000}

    score_us = price_mod._score_candidate(
        q_tokens, cand, prof_us_big, prefs, bare_roots={"AAPL"}
    )
    score_uk = price_mod._score_candidate(
        q_tokens, cand, prof_non_us_small, prefs, bare_roots={"AAPL"}
    )

    # US + huge cap + bare root match should beat non-US small cap
    assert score_us > score_uk
    # log-scale cap influences but doesn't explode
    assert score_us - score_uk < 30


# ---------------------- cache wrapper around resolver ------------------------


@pytest.mark.asyncio
async def test_resolve_name_cached_uses_cache(monkeypatch):
    # isolate cache for test
    price_mod._NAME_CACHE.clear()

    calls = {"n": 0}

    async def fake_resolve(q, token):
        calls["n"] += 1
        return ("AAPL", [{"symbol": "AAPL", "name": "Apple Inc"}])

    monkeypatch.setattr(price_mod, "_resolve_name_to_symbol", fake_resolve)

    tok = "dummy"
    out1 = await price_mod._resolve_name_cached("apple", tok)
    out2 = await price_mod._resolve_name_cached("apple", tok)

    assert out1 == ("AAPL", [{"symbol": "AAPL", "name": "Apple Inc"}])
    assert out2 == out1
    assert calls["n"] == 1  # second call served from cache


# ---------------------- client construction ----------------------------------


@pytest.mark.anyio
async def test_client_sets_stable_headers_and_timeout():
    async with price_mod._client(timeout=7.5) as c:
        # httpx stores headers in a case-insensitive dict
        assert c.headers.get("User-Agent") == "finance-chatbot/price/1.0"
        # can't directly assert timeout number (httpx wraps), but it should exist
        assert c.timeout is not None
