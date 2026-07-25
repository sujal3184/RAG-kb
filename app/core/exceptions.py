"""Custom error types used across the whole app.

Instead of raising plain `Exception` (too vague) or `HTTPException`
directly from business logic (mixes web-framework concerns into business
rules), every meaningful error in this app is one of these types.

Later, `main.py` catches these and turns them into proper HTTP responses.
"""


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