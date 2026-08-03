"""In-memory sliding-window rate limiting.

IMPORTANT LIMITATION: counts are tracked PER APP INSTANCE. With multiple
replicas behind a load balancer, each tracks its own window, so the
effective limit is (limit x replica count). For multi-instance
deployment this must move to a shared store (Redis). Correct as-is for
single-instance deployment; see DEPLOYMENT.md.
"""

import logging
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60

# Paths where a stricter limit applies — brute-forcing credentials or
# spamming password-reset emails are the attacks worth constraining hardest.
_STRICT_PATH_PREFIXES = (
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/resend-verification",
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Limits requests per client IP using a sliding time window."""

    def __init__(
        self,
        app,
        *,
        enabled: bool,
        requests_per_minute: int,
        auth_requests_per_minute: int,
    ) -> None:
        """Configure the limiter.

        Args:
            app: the ASGI app being wrapped.
            enabled: master switch.
            requests_per_minute: limit for general endpoints.
            auth_requests_per_minute: stricter limit for auth endpoints.
        """
        super().__init__(app)
        self._enabled = enabled
        self._general_limit = requests_per_minute
        self._auth_limit = auth_requests_per_minute
        # client_key -> deque of request timestamps within the window
        self._request_times: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Reject requests exceeding the per-IP limit for this path."""
        if not self._enabled:
            return await call_next(request)

        client_ip = self._client_ip(request)
        path = request.url.path
        limit = self._limit_for_path(path)
        bucket_key = f"{client_ip}:{'auth' if limit == self._auth_limit else 'general'}"

        if self._is_over_limit(bucket_key, limit):
            logger.warning(
                "Rate limit exceeded",
                extra={"client_ip": client_ip, "path": path, "limit": limit},
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Too many requests. Limit is {limit} per minute.",
                },
                headers={"Retry-After": str(_WINDOW_SECONDS)},
            )

        return await call_next(request)

    def _limit_for_path(self, path: str) -> int:
        """Return the applicable limit — stricter for auth endpoints."""
        if any(path.startswith(prefix) for prefix in _STRICT_PATH_PREFIXES):
            return self._auth_limit
        return self._general_limit

    def _is_over_limit(self, bucket_key: str, limit: int) -> bool:
        """Record this request and report whether the limit is exceeded.

        Uses a sliding window: timestamps older than the window are
        discarded, then the remaining count is compared to the limit.
        """
        now = time.monotonic()
        timestamps = self._request_times[bucket_key]

        cutoff = now - _WINDOW_SECONDS
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= limit:
            return True

        timestamps.append(now)
        return False

    @staticmethod
    def _client_ip(request: Request) -> str:
        """Determine the client IP, honouring X-Forwarded-For when behind
        a proxy.

        NOTE: X-Forwarded-For is client-controllable unless a trusted
        proxy overwrites it. Only rely on this when running behind a
        reverse proxy you control — see DEPLOYMENT.md.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"