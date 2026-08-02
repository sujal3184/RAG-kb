"""Custom error types used across the whole app."""


class AppException(Exception):
    """Base class for every custom error in this app."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class NotFoundError(AppException):
    """Something the user asked for doesn't exist. (-> HTTP 404)"""


class ConflictError(AppException):
    """The action conflicts with existing data, e.g. duplicate entry. (-> HTTP 409)"""


class ValidationError(AppException):
    """The input data is invalid in a way pydantic alone can't catch. (-> HTTP 422)"""


class AuthenticationError(AppException):
    """Login failed, or a token is missing/invalid/expired. (-> HTTP 401)"""


class AuthorizationError(AppException):
    """The user is logged in but not allowed to do this. (-> HTTP 403)"""

class GuardrailViolationError(AppException):
    """ Guardrails exception"""