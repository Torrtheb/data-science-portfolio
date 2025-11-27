# tests/test_market_tools.py
# Network-free tests for Finnhub-based helpers using monkeypatch.
import types
import pytest

import back_app.llm.tools as tools


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    # Ensure FINNHUB_API_KEY is present so _fh_get doesn't return config error
    monkeypatch.setenv("FINNHUB_API_KEY", "sandbox_TEST")
    yield


# -------------------- get_price --------------------


@pytest.mark.asyncio
async def test_get_price_equity_success(monkeypatch):
    async def fake_fh_get(path, params):
        assert path == "/quote"
        assert params["symbol"] == "AAPL"
        return {
            "c": 195.5,
            "o": 194.0,
            "h": 196.0,
            "l": 193.8,
            "pc": 193.0,
            "d": 2.5,
            "dp": 1.30,
            "t": 1700000000,
        }

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    out = await tools.get_price("aapl")
    assert out["symbol"] == "AAPL"
    assert out["price"] == 195.5
    assert out["change_pct"] == 1.30
    assert out["changePercent"] == 1.30


@pytest.mark.asyncio
async def test_get_price_crypto_fallback(monkeypatch):
    # First call (/quote) returns an error; because symbol contains ":", it should try /crypto/candle
    clock = {"now": 1_700_000_000}

    def fake_time():
        return clock["now"]

    async def fake_fh_get(path, params):
        if path == "/quote":
            return {"error": "api_error", "detail": "not an equity"}
        if path == "/crypto/candle":
            # note: get_price asks for last hour at resolution=1
            assert params["symbol"] == "BINANCE:BTCUSDT"
            return {
                "s": "ok",
                "c": [100, 101, 102],
                "t": [clock["now"] - 10, clock["now"] - 5, clock["now"]],
            }
        raise AssertionError(f"Unexpected path {path}")

    # Patch global time.time because get_price uses time.time() directly
    import time as real_time

    monkeypatch.setattr(real_time, "time", fake_time, raising=True)
    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)

    out = await tools.get_price("BINANCE:btcusdt")
    assert out["symbol"] == "BINANCE:BTCUSDT"
    assert out["price"] == 102
    assert out["open"] is None  # crypto fallback sets many fields to None
    assert out["ts"] == clock["now"]


@pytest.mark.asyncio
async def test_get_price_validation_empty():
    out = await tools.get_price("")
    assert out["error"] == "validation"


# -------------------- screen_equities --------------------


@pytest.mark.asyncio
async def test_screen_equities_filters(monkeypatch):
    # Return varying metrics; only MSFT should pass thresholds below
    async def fake_fh_get(path, params):
        assert path == "/company-basic-financials"
        if params["symbol"] == "AAPL":
            return {
                "metric": {
                    "marketCapitalization": 900_000_000_000,
                    "peBasicExclExtraTTM": 45,
                    "dividendYieldTTM": 0.005,
                }
            }
        if params["symbol"] == "MSFT":
            return {
                "metric": {
                    "marketCapitalization": 2_400_000_000_000,
                    "peTTM": 32,
                    "dividendYieldTTM": 0.009,
                }
            }
        if params["symbol"] == "BROKEN":
            return {"error": "api_error", "detail": "bad"}
        return {"metric": {}}

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    rows = await tools.screen_equities(
        ["AAPL", "MSFT", "BROKEN", "UNKNOWN"],
        min_market_cap=1e12,
        min_dividend_yield=0.008,
        max_pe=40.0,
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "MSFT"
    assert rows[0]["pe"] == 32
    assert rows[0]["div_yield"] == 0.009


# -------------------- search_symbol --------------------


@pytest.mark.asyncio
async def test_search_symbol_normalizes(monkeypatch):
    async def fake_fh_get(path, params):
        assert path == "/search"
        assert params["q"] == "apple"
        return {
            "result": [
                {
                    "symbol": "AAPL",
                    "description": "Apple Inc",
                    "type": "Common Stock",
                    "mic": "XNAS",
                },
                {
                    "symbol": "APPL34",
                    "description": "Apple Inc DRN",
                    "type": "DR",
                    "mic": "BVMF",
                },
            ]
        }

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    out = await tools.search_symbol("apple")
    assert out and out[0]["symbol"] == "AAPL"
    assert {"symbol", "description", "type", "mic"} <= out[0].keys()


@pytest.mark.asyncio
async def test_search_symbol_empty_query_returns_empty():
    out = await tools.search_symbol("   ")
    assert out == []


# -------------------- get_profile --------------------


@pytest.mark.asyncio
async def test_get_profile_happy(monkeypatch):
    async def fake_fh_get(path, params):
        assert path == "/stock/profile2"
        return {
            "name": "Apple Inc",
            "exchange": "NASDAQ",
            "currency": "USD",
            "ticker": "AAPL",
            "marketCapitalization": 3_000_000_000_000,
            "ipo": "1980-12-12",
            "logo": "https://logo",
        }

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    out = await tools.get_profile("aapl")
    assert out["symbol"] == "AAPL"
    assert out["name"] == "Apple Inc"
    assert out["market_cap"] == 3_000_000_000_000


@pytest.mark.asyncio
async def test_get_profile_error_passthrough(monkeypatch):
    async def fake_fh_get(path, params):
        return {"error": "forbidden", "status": 403, "detail": "Nope"}

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    out = await tools.get_profile("AAPL")
    assert out["error"] == "forbidden"
    assert out["status"] == 403


# -------------------- get_recommendation_trends --------------------


@pytest.mark.asyncio
async def test_get_recommendation_trends_happy(monkeypatch):
    async def fake_fh_get(path, params):
        assert path == "/stock/recommendation"
        # API returns an array; function selects first element and adds 'symbol'
        return [
            {
                "period": "2025-08-01",
                "strongBuy": 10,
                "buy": 20,
                "hold": 5,
                "sell": 1,
                "strongSell": 0,
            },
            {
                "period": "2025-07-01",
                "strongBuy": 8,
                "buy": 18,
                "hold": 6,
                "sell": 2,
                "strongSell": 0,
            },
        ]

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    out = await tools.get_recommendation_trends("msft")
    assert out["symbol"] == "MSFT"
    assert out["period"] == "2025-08-01"
    assert out["strongBuy"] == 10


@pytest.mark.asyncio
async def test_get_recommendation_trends_no_data(monkeypatch):
    async def fake_fh_get(path, params):
        return []

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    out = await tools.get_recommendation_trends("MSFT")
    assert out["error"] == "no_data"


# -------------------- get_candles --------------------


@pytest.mark.asyncio
async def test_get_candles_stock_endpoint(monkeypatch):
    async def fake_fh_get(path, params):
        assert path == "/stock/candle"
        assert params["symbol"] == "AAPL"
        assert params["resolution"] == "D"
        return {"s": "ok", "c": [1, 2], "t": [1, 2]}

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    out = await tools.get_candles("AAPL", "D", 1000, 2000)
    assert out["s"] == "ok"


@pytest.mark.asyncio
async def test_get_candles_crypto_endpoint(monkeypatch):
    async def fake_fh_get(path, params):
        assert path == "/crypto/candle"
        assert params["symbol"] == "BINANCE:BTCUSDT"
        assert params["resolution"] == "60"
        return {"s": "ok", "c": [100], "t": [1]}

    monkeypatch.setattr(tools, "_fh_get", fake_fh_get, raising=True)
    out = await tools.get_candles("BINANCE:BTCUSDT", "60", 1000, 2000)
    assert out["s"] == "ok"


@pytest.mark.asyncio
async def test_get_candles_validation_errors():
    out = await tools.get_candles("", "D", 1, 2)
    assert out["error"] == "validation"
    out = await tools.get_candles("AAPL", "2H", 1, 2)
    assert out["error"] == "validation"
    out = await tools.get_candles("AAPL", "D", 10, 5)
    assert out["error"] == "validation"
