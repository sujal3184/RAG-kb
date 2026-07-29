"""Embedding-specific errors."""

from app.core.exceptions import AppException


class EmbeddingError(AppException):
    """Raised when text cannot be embedded (model failure, invalid input, etc.)."""


class AllEmbeddingProvidersFailedError(EmbeddingError):
    """Raised when BOTH the primary and fallback embedding providers fail."""