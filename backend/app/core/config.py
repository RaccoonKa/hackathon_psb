"""Application configuration loaded from environment variables and `.env`."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

_PLACEHOLDER_API_KEY = "your-api-key-here"
_DEFAULT_LLM_MODEL = "deepseek/deepseek-v4.1-flash"
_DEFAULT_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def _first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty stripped environment variable from `names`."""
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            stripped = raw.strip()
            if stripped:
                return stripped
    return default


def _parse_cors_origins(raw: str | None) -> list[str]:
    """Parse comma-separated CORS origins with sensible local development defaults."""
    if not raw or not raw.strip():
        return list(_DEFAULT_CORS_ORIGINS)
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or list(_DEFAULT_CORS_ORIGINS)


class Settings(BaseModel):
    """Runtime settings for the backend API service."""

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = _DEFAULT_LLM_MODEL
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    llm_history_limit: int = 20
    assistant_backend: str = "direct_llm"
    database_url: str = "sqlite:///./storage/chat.db"
    data_dir: Path = Path("./data")
    cors_origins: list[str] = Field(default_factory=lambda: list(_DEFAULT_CORS_ORIGINS))

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip().rstrip("/")

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _normalize_api_key(cls, value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @property
    def llm_configured(self) -> bool:
        """Return True iff both a non-placeholder API key and base URL are configured."""
        return (
            bool(self.llm_base_url)
            and bool(self.llm_api_key)
            and self.llm_api_key != _PLACEHOLDER_API_KEY
        )

    @classmethod
    def from_env(cls) -> Settings:
        """Build a `Settings` instance from environment variables."""
        return cls(
            llm_base_url=_first_env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL"),
            llm_api_key=_first_env("LLM_API_KEY", "OPENAI_API_KEY"),
            llm_model=_first_env(
                "LLM_MODEL",
                "GEN_MODEL",
                "OPENAI_MODEL",
                default=_DEFAULT_LLM_MODEL,
            ),
            llm_temperature=float(_first_env("LLM_TEMPERATURE", "TEMPERATURE", default="0.0")),
            llm_max_tokens=int(_first_env("LLM_MAX_TOKENS", "MAX_TOKENS", default="1024")),
            llm_timeout_seconds=float(_first_env("LLM_TIMEOUT_SECONDS", default="60.0")),
            llm_history_limit=int(_first_env("LLM_HISTORY_LIMIT", default="20")),
            assistant_backend=_first_env("ASSISTANT_BACKEND", default="direct_llm"),
            database_url=_first_env("DATABASE_URL", default="sqlite:///./storage/chat.db"),
            data_dir=Path(_first_env("DATA_DIR", default="./data")),
            cors_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS")),
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings loaded from environment variables."""
    return Settings.from_env()
