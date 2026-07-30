"""LLM-specific errors."""

from app.core.exceptions import AppException


class LLMError(AppException):
    """Raised when an LLM request fails (after retries are exhausted)."""


class AllLLMProvidersFailedError(LLMError):
    """Raised when BOTH the primary and fallback LLM models fail."""