"""Security response headers.

Adds standard headers that instruct browsers to apply protections against
common attack classes. These matter when a browser-based frontend
consumes this API.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_SECURITY_HEADERS = {
    # Prevents browsers from guessing content types, which can turn an
    # uploaded text file into executable script.
    "X-Content-Type-Options": "nosniff",
    # Disallows this API being embedded in a frame (clickjacking defence).
    "X-Frame-Options": "DENY",
    # Limits how much referrer information leaks to other sites.
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # This is a JSON API — no scripts, styles, or frames should ever load
    # from it, so the strictest possible policy is appropriate.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    # Disables browser features this API never needs.
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attaches security headers to every response."""

    def __init__(self, app, *, enabled: bool) -> None:
        """Configure the middleware.

        Args:
            app: the ASGI app being wrapped.
            enabled: master switch — disabling is useful only when a
                reverse proxy sets these headers instead.
        """
        super().__init__(app)
        self._enabled = enabled

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Add security headers to the response."""
        response = await call_next(request)

        if self._enabled:
            for header, value in _SECURITY_HEADERS.items():
                response.headers.setdefault(header, value)

        return response