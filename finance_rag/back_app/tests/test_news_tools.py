# tests/test_news_tools.py
# Tests for news helpers using monkeypatch + deterministic data (no network).

import types
import datetime as dt
import pytest
import time as real_time
import back_app.llm.tools as tools


class FixedDate(dt.date):
    @classmethod
    def today(cls) -> "FixedDate":
        # Return a real date instance; subtraction with timedelta will work.
        return cls(2025, 9, 5)


@pytest.mark.asyncio
async def test_get_company_news_basic(monkeypatch):
    """
    Ensures normalization & de-duplication and that limit is respected.
    """

    calls = {"count": 0}

    async def fake_fh_get(path, params):
        calls["count"] += 1
        assert path == "/company-news"
        # Include a duplicate URL and a finnhub redirect URL
        return [
            {
                "headline": "Apple launches new product",
                "url": "https://example.com/aapl1",
                "source": "ExampleNews",
                "datetime": 1700000000,
            },
            {
                "headline": "Apple launches new product",  # duplicate headline
                "url": "https://example.com/aapl1",  # duplicate url
                "source": "ExampleNews",
                "datetime": 1700000001,
            },
            {
                "headline": "Analyst upgrades AAPL",
                "url": "https://finnhub.io/api/news?id=ABC123",
                "source": "Finnhub",
                "datetime": 1700000002,
            },
        ]

    tools._NEWS_CACHE.clear()
    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    # Patch the module’s `date` symbol to our subclass
    monkeypatch.setattr(tools, "date", FixedDate, raising=False)

    out = await tools.get_company_news("AAPL", days=7, limit=5)
    assert out["symbol"] == "AAPL"
    items = out["items"]
    # Deduped: first unique; second duplicate removed; third (redirect) kept
    assert len(items) == 2
    for it in items:
        assert {"title", "url", "source", "datetime"} <= it.keys()


@pytest.mark.asyncio
async def test_get_company_news_cache_ttl(monkeypatch):
    """
    Confirms cached value is returned within TTL without calling upstream again,
    then simulates TTL expiry and ensures we call upstream a second time.
    """
    # Keep TTL small for the test
    monkeypatch.setenv("NEWS_TTL_SEC", "180")
    tools._NEWS_TTL_SEC = 180

    clock = {"now": 1_700_000_000.0}

    def fake_time():
        return clock["now"]

    calls = {"count": 0}

    async def fake_fh_get(path, params):
        calls["count"] += 1
        return [
            {
                "headline": "Item 1",
                "url": "https://example.com/1",
                "source": "X",
                "datetime": 1,
            },
            {
                "headline": "Item 2",
                "url": "https://example.com/2",
                "source": "Y",
                "datetime": 2,
            },
        ]

    tools._NEWS_CACHE.clear()
    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    # Patch the module’s `date` symbol to our subclass (correct way)
    monkeypatch.setattr(tools, "date", FixedDate, raising=False)
    # Patch the module’s `time` reference (get_company_news reads `time.time()` from module namespace)
    monkeypatch.setattr(real_time, "time", fake_time, raising=True)

    # First call — hits upstream
    out1 = await tools.get_company_news("MSFT", days=7, limit=10)
    assert calls["count"] == 1
    assert len(out1["items"]) == 2

    # Second call within TTL — should use cache (no new upstream call)
    clock["now"] += 60  # +1 minute
    out2 = await tools.get_company_news("MSFT", days=7, limit=10)
    assert calls["count"] == 1
    assert out2 is out1  # same cached object

    # After TTL — should re-fetch
    clock["now"] += 181  # move beyond 180s TTL
    out3 = await tools.get_company_news("MSFT", days=7, limit=10)
    assert calls["count"] == 2
    assert len(out3["items"]) == 2
    assert out3 is not out1
