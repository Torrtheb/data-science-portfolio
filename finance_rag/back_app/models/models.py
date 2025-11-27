from __future__ import annotations
from typing import Optional, List, Dict, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationInfo
from ..utils.utils import parse_percent

# =============================================================================
# Chat (RAG + Agent) payloads
# =============================================================================


class Message(BaseModel):
    """Single chat message."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


ChatMessage = Message


class ToolPriceIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["price"]
    symbol: str


class ToolNewsIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["news"]
    symbol: str
    days: Optional[int] = None
    limit: Optional[int] = None


ToolIntent = Union[ToolPriceIntent, ToolNewsIntent]


class ChatInput(BaseModel):
    """
    Flexible chat request that accepts either:
      * 'messages' (full conversation), or
      * a single 'question' string (coerced to one user message).
    After validation, 'messages' will always be present.
    """

    model_config = ConfigDict(extra="forbid")

    question: Optional[str] = None
    messages: Optional[List[Message]] = None

    @field_validator("messages", mode="before")
    @classmethod
    def normalize_messages(cls, v, info: ValidationInfo):
        if v:
            return v
        q = (info.data or {}).get("question")
        if q:
            return [Message(role="user", content=q)]
        raise ValueError("Provide either 'messages' or 'question'")


class ChatRequest(BaseModel):
    """
    Internal chat call shape. Always messages; may include explicit tool intents.
    """

    model_config = ConfigDict(extra="forbid")

    messages: List[Message]
    intents: Optional[List[ToolIntent]] = None


class ChatResponse(BaseModel):
    """Assistant answer + optional sources, tool outputs, and token usage."""

    model_config = ConfigDict(extra="ignore")

    answer: str
    sources: List[Dict] = Field(default_factory=list)
    tools: List[Dict] = Field(default_factory=list)
    usage: Dict = Field(default_factory=dict)


class ChatStreamRequest(BaseModel):
    """Request payload for streaming chat endpoints."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: Optional[str] = None
    messages: List[ChatMessage] = Field(
        min_length=1, description="Conversation history"
    )


# =============================================================================
# Finnhub / market data
# =============================================================================


def _norm_symbol(s: str) -> str:
    return (s or "").strip().upper()


class PriceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1, description="Ticker symbol, e.g., AAPL")

    @field_validator("symbol")
    @classmethod
    def _v_sym(cls, v: str) -> str:
        return _norm_symbol(v)


class SymbolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1, description="Ticker symbol, e.g., AAPL")

    @field_validator("symbol")
    @classmethod
    def _v_sym(cls, v: str) -> str:
        return _norm_symbol(v)


class SymbolQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(
        min_length=1, max_length=60, description="Free-text search query"
    )


class CompanyNewsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1)
    days: int = Field(default=7, gt=0, le=30, description="Lookback window in days")
    limit: int = Field(default=10, gt=0, le=50, description="Max news items to return")

    @field_validator("symbol")
    @classmethod
    def _v_sym(cls, v: str) -> str:
        return _norm_symbol(v)


class RecoBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1)

    @field_validator("symbol")
    @classmethod
    def _v_sym(cls, v: str) -> str:
        return _norm_symbol(v)


Resolution = Literal["1", "5", "15", "30", "60", "D", "W", "M"]


class CandleInput(BaseModel):
    """
    OHLCV candles request.
    Uses alias 'from' in JSON as 'from_' in Python.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    symbol: str
    resolution: Resolution
    from_: int = Field(alias="from", description="UNIX seconds (start)")
    to: int = Field(description="UNIX seconds (end)")

    @field_validator("symbol")
    @classmethod
    def _v_sym(cls, v: str) -> str:
        return _norm_symbol(v)


# =============================================================================
# Calculators — strict inputs
# =============================================================================


class InterestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: float = Field(gt=0)
    rate: float = Field(ge=0, description="Annual rate in percent, e.g., 7 for 7%")
    years: float = Field(gt=0)
    compounding_per_year: Optional[int] = Field(default=None, gt=0)
    compounds_per_year: Optional[int] = Field(
        default=None,
        gt=0,
        description="Alias for compounding_per_year; either is accepted.",
    )
    inflation_rate_percent: Optional[float] = Field(default=0.0, ge=0)
    inflation_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Alias for inflation_rate_percent; either is accepted.",
    )

    @field_validator("compounding_per_year", "compounds_per_year", mode="after")
    @classmethod
    def _coalesce_compounds(cls, v, info: ValidationInfo):
        return v


class AmortizationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: float = Field(gt=0)
    annual_rate: float = Field(ge=0, description="APR in percent")
    years: int = Field(gt=0)
    payments_per_year: int = Field(default=12, gt=0)


# =============================================================================
# Calculators
# =============================================================================


class SimpleInterestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: float = Field(gt=0, description="Starting balance (> 0)")
    rate: float | str = Field(..., description="Annual interest (e.g., '6%', 0.06, 6)")
    years: float = Field(gt=0, description="Years (> 0)")
    inflation_rate_percent: Optional[float | str] = Field(
        default=None, description="Optional annual inflation"
    )
    inflation_rate: Optional[float | str] = Field(
        default=None, description="Alias of inflation_rate_percent"
    )

    @field_validator("rate", "inflation_rate_percent", "inflation_rate", mode="before")
    @classmethod
    def _v_pct(cls, v):
        return parse_percent(v)


class CompoundInterestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: float = Field(gt=0, description="Starting balance (> 0)")
    rate: float | str = Field(..., description="Annual interest, percent-like")
    years: float = Field(gt=0, description="Years (> 0)")
    compounds_per_year: int = Field(1, gt=0, description="Compounds per year (>= 1)")
    inflation_rate_percent: Optional[float | str] = Field(
        default=None, description="Optional annual inflation"
    )
    inflation_rate: Optional[float | str] = Field(
        default=None, description="Alias of inflation_rate_percent"
    )

    @field_validator("rate", "inflation_rate_percent", "inflation_rate", mode="before")
    @classmethod
    def _v_pct(cls, v):
        return parse_percent(v)


class AmortizationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: float = Field(gt=0, description="Loan principal (> 0)")
    annual_rate: float | str = Field(
        ..., description="Nominal annual rate, percent-like"
    )
    years: int = Field(gt=0, description="Term in years (> 0)")
    payments_per_year: int = Field(12, gt=0, description="Payments per year (>= 1)")

    @field_validator("annual_rate", mode="before")
    @classmethod
    def _v_pct(cls, v):
        return parse_percent(v)


class InvestmentReturnBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: float = Field(ge=0, description="Starting balance (>= 0)")
    rate: float | str = Field(..., description="Annual nominal rate, percent-like")
    years: float = Field(..., description="Years")
    compounds_per_year: int = Field(12, gt=0, description="Compounds per year (>= 1)")
    contribution_per_period: float = Field(
        0.0, ge=0, description="Contribution per period (>= 0)"
    )
    contribution_frequency_per_year: int = Field(
        12, ge=1, description="Contribution frequency per year (>= 1)"
    )
    contribution_timing: Literal["end", "begin"] = Field(
        "end", description='"end" (ordinary) or "begin" (due)'
    )
    inflation_rate_percent: Optional[float | str] = Field(
        default=None, description="Optional annual inflation"
    )

    @field_validator("rate", "inflation_rate_percent", mode="before")
    @classmethod
    def _v_pct(cls, v):
        return parse_percent(v)


class CagrBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial: float = Field(gt=0)
    final: float = Field(gt=0)
    years: float = Field(gt=0)


class NpvBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate_percent_per_period: float | str
    cashflows: List[float]

    @field_validator("rate_percent_per_period", mode="before")
    @classmethod
    def _v_pct(cls, v):
        out = parse_percent(v)
        if out is None:
            raise ValueError("rate_percent_per_period is required")
        return out
