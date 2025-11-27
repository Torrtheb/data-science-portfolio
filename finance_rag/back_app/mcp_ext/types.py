from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class WorldBankValueArgs(BaseModel):
    """
    Request payload for a single-year World Bank indicator lookup.
    Accepts a country name or code; the router will normalize to ISO-3.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    country_id: str = Field(
        ..., description="Country name or code, e.g. 'Albania' or 'ALB'"
    )
    indicator_id: str = Field(
        ..., description="World Bank indicator code, e.g. 'NY.GDP.PCAP.PP.KD'"
    )
    year: int = Field(..., ge=1900, le=2100, description="Year, 4 digits")
    prefer_constant: Optional[bool] = Field(
        True,
        description="If PPP GDP per capita, prefer constant 2021 series (KD) when True.",
    )


class WorldBankValueResult(BaseModel):
    """
    Response payload from the WB value endpoint.
    The router will ensure country fields and clamped year are present.
    """

    model_config = ConfigDict(extra="ignore")

    ok: bool = True
    country: Optional[str] = None
    country_iso3: Optional[str] = None
    indicator_id: Optional[str] = None
    indicator_name: Optional[str] = None
    year: Optional[int] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    source_url: Optional[str] = None
    message: Optional[str] = None
