from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Union, Optional
from pathlib import Path
import json
import os

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    AliasChoices,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------- Helpers ----------


def _parse_str_list(value: Optional[str]) -> Optional[list[str]]:
    """
    Parse an environment string containing either JSON (["a","b"]) or CSV ("a,b")
    into a list[str]. Returns None if input is None.
    """
    if value is None:
        return None
    raw = value.strip()
    if raw == "":
        return []
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [str(x).strip() for x in loaded if str(x).strip()]
        return [str(loaded).strip()]
    except Exception:
        return [s.strip() for s in raw.split(",") if s.strip()]


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    """
    Normalize stringy booleans: 1/0, true/false, yes/no, on/off.
    """
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _mask_secret(s: Optional[SecretStr], visible: int = 4) -> str:
    """
    Mask SecretStr for logs. Keep last 'visible' chars if available.
    """
    if not s:
        return "∅"
    val = s.get_secret_value()
    if len(val) <= visible:
        return "***"
    return f"***{val[-visible:]}"


# ---------- Pricing data classes (unchanged API) ----------


@dataclass(frozen=True)
class ModelPricing:
    """Per-1k token pricing (USD) for chat/completions models."""

    input_per_1k: float
    output_per_1k: float


PRICING = {
    "gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
    "gpt-3.5-turbo": ModelPricing(input_per_1k=0.00050, output_per_1k=0.00150),
}


@dataclass(frozen=True)
class EmbeddingPrice:
    """Per-1k token pricing for embedding models."""

    input_per_1k: float


EMBEDDING_PRICING: dict[str, EmbeddingPrice] = {
    "text-embedding-3-small": EmbeddingPrice(input_per_1k=0.00001),
    "text-embedding-3-large": EmbeddingPrice(input_per_1k=0.000065),
}


# ---------- Settings model ----------


class Settings(BaseSettings):
    """
    Application configuration loaded from environment with sensible defaults.

    Env precedence examples (AliasChoices):
    - OPENAI_API_KEY or openai_api_key
    - FRONTEND_ORIGIN, front_origin, or frontend_origin
    """

    cors_allow_origins_env_raw: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CORS_ALLOW_ORIGINS", "cors_allow_origins"),
        description="Raw env for CORS origins, JSON or CSV.",
    )
    trusted_hosts_env_raw: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TRUSTED_HOSTS", "trusted_hosts"),
        description="Raw env for trusted hosts, JSON or CSV.",
    )
    cors_allow_origins: list[Union[AnyHttpUrl, str]] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="CORS allowed origins.",
    )
    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"],
        description="Proxy/FastAPI trusted hosts.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGSMITH_API_KEY", "langsmith_api_key"),
    )
    twelvedata_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TWELVEDATA_API_KEY", "twelvedata_api_key"),
    )
    openai_model: str = Field(
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        validation_alias=AliasChoices("OPENAI_MODEL", "openai_model"),
        description="Default chat model.",
    )
    openai_embedding_model: str = Field(
        default=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            os.getenv("embedding_model", "text-embedding-3-small"),
        ),
        validation_alias=AliasChoices(
            "OPENAI_EMBEDDING_MODEL", "openai_embedding_model", "embedding_model"
        ),
        description="Default embedding model.",
    )
    docs_dir: str = Field(
        default=os.getenv("DOCS_DIR", str(REPO_ROOT / "docs")),
        validation_alias=AliasChoices("DOCS_DIR", "docs_dir"),
        description="Directory for knowledge base source documents.",
    )
    finnhub_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FINNHUB_API_KEY", "finnhub_api_key"),
    )
    finnhub_rps: int = Field(
        default=int(os.getenv("FINNHUB_RPS", "30")),
        validation_alias=AliasChoices("FINNHUB_RPS", "finnhub_rps"),
        description="Allowed requests per second to Finnhub (client-side throttle).",
    )
    finnhub_period: int = Field(
        default=int(os.getenv("FINNHUB_PERIOD", "1")),
        validation_alias=AliasChoices("FINNHUB_PERIOD", "finnhub_period"),
        description="Throttle window (seconds) for Finnhub RPS.",
    )
    external_timeout: float = Field(
        default=float(os.getenv("EXTERNAL_TIMEOUT", "12.0")),
        validation_alias=AliasChoices("EXTERNAL_TIMEOUT", "external_timeout"),
        description="Default HTTP client timeout (seconds) for external APIs.",
    )
    rate_limit_rps: float = Field(
        default=float(os.getenv("RATE_LIMIT_RPS", "3.0")),
        validation_alias=AliasChoices("RATE_LIMIT_RPS", "rate_limit_rps"),
        description="Application-level average RPS (token bucket).",
    )
    api_rate_max: int = Field(
        default=int(os.getenv("API_RATE_MAX", "60")),
        validation_alias=AliasChoices("API_RATE_MAX", "api_rate_max"),
        description="Max requests per window per client.",
    )
    api_rate_window: int = Field(
        default=int(os.getenv("API_RATE_WINDOW", "60")),
        validation_alias=AliasChoices("API_RATE_WINDOW", "api_rate_window"),
        description="Window seconds for api_rate_max.",
    )
    session_token_secret: str | None = Field(
        default=os.getenv("SESSION_TOKEN_SECRET"),
        validation_alias=AliasChoices("SESSION_TOKEN_SECRET", "session_token_secret"),
        description="Secret for signing per-session bearer tokens. If set, chat/export routes require a valid token.",
    )
    session_token_ttl_seconds: int = Field(
        default=int(os.getenv("SESSION_TOKEN_TTL_SECONDS", str(7 * 24 * 3600))),
        validation_alias=AliasChoices(
            "SESSION_TOKEN_TTL_SECONDS", "session_token_ttl_seconds"
        ),
        description="TTL (seconds) for signed session tokens.",
    )
    session_token_required: bool = Field(
        default=_as_bool(os.getenv("SESSION_TOKEN_REQUIRED", "false")),
        validation_alias=AliasChoices(
            "SESSION_TOKEN_REQUIRED", "session_token_required"
        ),
        description="Force session token requirement even if secret is absent (primarily for tests).",
    )
    openai_moderation_enabled: bool = Field(
        default=_as_bool(os.getenv("OPENAI_MODERATION_ENABLED", "false")),
        validation_alias=AliasChoices(
            "OPENAI_MODERATION_ENABLED", "openai_moderation_enabled"
        ),
        description="If true, run OpenAI moderation before chat/stream.",
    )
    openai_moderation_model: str = Field(
        default=os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest"),
        validation_alias=AliasChoices(
            "OPENAI_MODERATION_MODEL", "openai_moderation_model"
        ),
        description="Moderation model name.",
    )
    max_body_bytes: int = Field(
        default=int(os.getenv("MAX_BODY_BYTES", str(5 * 1024 * 1024))),
        validation_alias=AliasChoices("MAX_BODY_BYTES", "max_body_bytes"),
        description="Maximum request body size in bytes.",
    )
    backend_origin: Union[AnyHttpUrl, str] = Field(
        default=os.getenv("BACKEND_ORIGIN", "http://localhost:8000"),
        validation_alias=AliasChoices("BACKEND_ORIGIN", "backend_origin"),
        description="Backend origin (CORS & FE requests).",
    )
    frontend_origin: Union[AnyHttpUrl, str] = Field(
        default=os.getenv(
            "FRONTEND_ORIGIN", os.getenv("front_origin", "http://localhost:3000")
        ),
        validation_alias=AliasChoices(
            "FRONTEND_ORIGIN", "front_origin", "frontend_origin"
        ),
        description="Frontend origin.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("API_KEY", "api_key"),
        description="Optional shared API key for protected routes.",
    )
    allowed_quote_sources: List[str] = Field(
        default_factory=lambda: ["finnhub"],
        validation_alias=AliasChoices("ALLOWED_QUOTE_SOURCES", "allowed_quote_sources"),
        description="Whitelisted quote providers.",
    )
    max_quote_age_seconds: int = Field(
        default=int(os.getenv("MAX_QUOTE_AGE_SECONDS", str(5 * 60))),
        validation_alias=AliasChoices("MAX_QUOTE_AGE_SECONDS", "max_quote_age_seconds"),
        description="Max staleness for cached quotes.",
    )
    track_tokens: bool = Field(
        default=_as_bool(os.getenv("TRACK_TOKENS"), True),
        validation_alias=AliasChoices("TRACK_TOKENS", "track_tokens"),
        description="Enable token accounting / analytics.",
    )
    rag_chunk_size: int = Field(
        default=int(os.getenv("RAG_CHUNK_SIZE", "800")),
        validation_alias=AliasChoices("RAG_CHUNK_SIZE", "rag_chunk_size"),
        description="RAG document chunk size (characters).",
    )
    rag_chunk_overlap: int = Field(
        default=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        validation_alias=AliasChoices("RAG_CHUNK_OVERLAP", "rag_chunk_overlap"),
        description="RAG document chunk overlap (characters).",
    )
    rag_top_k: int = Field(
        default=int(os.getenv("RAG_TOP_K", "4")),
        validation_alias=AliasChoices("RAG_TOP_K", "rag_top_k"),
        description="Default top-k documents to retrieve for RAG.",
    )
    rag_min_score: float = Field(
        default=float(os.getenv("RAG_MIN_SCORE", "0.25")),
        validation_alias=AliasChoices("RAG_MIN_SCORE", "rag_min_score"),
        description="Minimum relevance score to keep a RAG document.",
    )

    # ----- Back-compat aliases (to avoid AttributeError in older code) -----

    @property
    def embedding_model(self) -> str:
        """Back-compat alias for openai_embedding_model (read-only)."""
        return self.openai_embedding_model

    @property
    def front_origin(self) -> Union[AnyHttpUrl, str]:
        """Back-compat alias for frontend_origin (read-only)."""
        return self.frontend_origin

    @property
    def cors_origins(self) -> list[Union[AnyHttpUrl, str]]:
        """Back-compat alias for cors_allow_origins (read-only)."""
        return self.cors_allow_origins

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        if isinstance(v, str) or v is None:
            return _parse_str_list(v) or ["http://localhost:3000"]
        return v

    @field_validator("trusted_hosts", mode="before")
    @classmethod
    def _parse_trusted_hosts(cls, v):
        if isinstance(v, str) or v is None:
            return _parse_str_list(v) or ["localhost", "127.0.0.1"]
        return v

    @field_validator("allowed_quote_sources", mode="before")
    @classmethod
    def _parse_sources(cls, v):
        if isinstance(v, str) or v is None:
            return _parse_str_list(v) or ["finnhub"]
        return v

    @field_validator(
        "finnhub_rps",
        "finnhub_period",
        "api_rate_max",
        "api_rate_window",
        "max_body_bytes",
        "max_quote_age_seconds",
        mode="after",
    )
    @classmethod
    def _must_be_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Value must be a positive integer.")
        return v

    @field_validator("external_timeout", "rate_limit_rps", mode="after")
    @classmethod
    def _must_be_positive_number(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Value must be a positive number.")
        return v

    def safe_summary(self) -> str:
        """Short masked summary for startup logs."""
        return (
            "Settings("
            f"openai_model={self.openai_model}, "
            f"embed_model={self.openai_embedding_model}, "
            f"openai_api_key={_mask_secret(self.openai_api_key)}, "
            f"finnhub_api_key={_mask_secret(self.finnhub_api_key)}, "
            f"backend_origin={self.backend_origin}, "
            f"frontend_origin={self.frontend_origin}, "
            f"rate_limit={self.api_rate_max}/{self.api_rate_window}s, "
            f"track_tokens={self.track_tokens}"
            ")"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Create and cache a Settings instance.
    Use this from app startup and request handlers.
    """
    return Settings()


settings = get_settings()
TRACK_TOKENS: bool = settings.track_tokens
