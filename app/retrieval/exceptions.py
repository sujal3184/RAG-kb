"""Vector store-specific errors."""

from app.core.exceptions import AppException


class VectorStoreError(AppException):
    """Raised when a vector store operation fails (connection, query, etc.)."""