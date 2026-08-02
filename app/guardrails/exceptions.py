"""Guardrail-specific errors."""

from app.core.exceptions import AppException


class GuardrailViolationError(AppException):
    """Raised when input fails a guardrail check and must be blocked."""