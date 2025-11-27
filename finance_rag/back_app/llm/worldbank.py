from __future__ import annotations
import os
from typing import List, Dict
import httpx

DEFAULT_INDICATOR = "FP.CPI.TOTL.ZG"  # Inflation, consumer prices (annual %)
WORLD_BANK_BASE = os.getenv("WORLD_BANK_API", "https://api.worldbank.org/v2")


async def _fetch_indicator(
    country: str, indicator: str, latest_only: bool = True
) -> List[Dict]:
    params = {
        "format": "json",
        "per_page": 500,
    }
    url = f"{WORLD_BANK_BASE}/country/{country.lower()}/indicator/{indicator}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            return []
        series = data[1] or []
        rows = []
        for row in series:
            yr = row.get("date")
            val = row.get("value")
            if val is None:
                continue
            rows.append(
                {
                    "year": yr,
                    "value": float(val),
                    "indicator": indicator,
                    "country": country.upper(),
                }
            )
        rows.sort(key=lambda r: r["year"], reverse=True)
        if latest_only and rows:
            return rows[:1]
        return rows


async def get_indicator(
    country: str, indicator: str = DEFAULT_INDICATOR, latest_only: bool = True
) -> Dict:
    """
    Fetch a World Bank indicator (default: CPI inflation) for a country code.

    Args:
        country: ISO2 or ISO3 country code, e.g., "CA" or "CAN".
        indicator: World Bank indicator code, default FP.CPI.TOTL.ZG (inflation).
        latest_only: If True, return only the latest non-null observation.

    Returns:
        dict with 'rows' list; includes error on failure.
    """
    if not country or not country.strip():
        return {
            "error": "country_missing",
            "detail": "Provide ISO2/ISO3 country code (e.g., CA or CAN).",
        }
    indicator = indicator.strip() or DEFAULT_INDICATOR
    try:
        rows = await _fetch_indicator(
            country.strip(), indicator, latest_only=latest_only
        )
        return {"ok": True, "rows": rows}
    except Exception as e:
        return {"error": "world_bank_error", "detail": str(e)}
