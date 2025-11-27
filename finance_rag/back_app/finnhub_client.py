from __future__ import annotations
import os
from typing import Any, Optional
from pydantic import SecretStr
import httpx
from fastapi import HTTPException


class _FinnhubClient:
    """
    Async wrapper for the Finnhub REST API with sensible timeouts / retries.
    Injects the 'token' query param and raises HTTP 400 if the API key is missing.
    """

    def __init__(self, api_key: Optional[str | SecretStr]):
        # Accept SecretStr or plain string and normalize to raw value.
        try:
            self.api_key = api_key.get_secret_value() if api_key is not None else None  # type: ignore[attr-defined]
        except Exception:
            self.api_key = str(api_key) if api_key else None
        base_timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
        timeout = httpx.Timeout(
            connect=10.0, read=base_timeout, write=10.0, pool=base_timeout
        )
        transport = httpx.AsyncHTTPTransport(retries=2)
        self._client = httpx.AsyncClient(
            base_url="https://finnhub.io/api/v1",
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            headers={"User-Agent": "finance-chatbot/1.0"},
        )

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise HTTPException(
                status_code=400,
                detail="FINNHUB_API_KEY not configured for market-data tools",
            )
        p = dict(params or {})
        p["token"] = self.api_key
        r = await self._client.get(path, params=p)
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()


FINNHUB = _FinnhubClient(os.getenv("FINNHUB_API_KEY"))
