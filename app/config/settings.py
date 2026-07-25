"""Application settings.

This is the ONLY place in the whole app that reads environment variables.
Every other file should import `get_settings` and use the values from here.

Why do it this way?
- If a setting is missing or wrong (e.g. PORT is not a number), the app
  will fail immediately at startup with a clear error — instead of
  crashing randomly later, deep inside some unrelated function.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """The environment the app is currently running in."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(StrEnum):
    """How detailed the logs should be."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """All configuration values the app needs, with types and defaults.

    Values come from (in priority order):
    1. Real environment variables (e.g. set by Docker)
    2. The `.env` file
    3. The defaults written below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of crashing
    )

    # App
    APP_NAME: str = "Knowledge Base RAG"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)

    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_JSON: bool = False

    @property
    def is_production(self) -> bool:
        """True only when running in production."""
        return self.ENVIRONMENT == Environment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    """Return the app settings (loaded only once, then reused).

    Using a function (instead of just a global variable) lets us swap in
    fake settings during tests, using FastAPI's dependency override system.
    """
    return Settings()