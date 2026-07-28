"""Chunking-specific errors."""

from app.core.exceptions import AppException


class ChunkingError(AppException):
    """Raised when text cannot be chunked (e.g. invalid configuration)."""