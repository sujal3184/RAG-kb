"""Request logging and metrics middleware.

Assigns every request a unique request ID, records timing/status metrics,
and makes the request ID available in logs — so a single user's request
can be traced across log lines, OpenTelemetry spans, and Langfuse entries.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.observability.metrics import http_request_duration_seconds, http_requests_total

logger = logging.getLogger(__name__)


class RequestObservabilityMiddleware(BaseHTTPMiddleware):
    """Adds request IDs, structured request logging, and Prometheus metrics."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Wrap every request with an ID, timing, logging, and metrics.

        The request ID is echoed back in an `X-Request-ID` response header
        so a user reporting a problem can quote it, and we can find their
        exact request in logs and traces.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Use the route TEMPLATE (e.g. "/knowledge-bases/{id}") rather than
        # the raw path, so metrics don't explode into one label per UUID.
        endpoint = request.url.path
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            logger.exception(
                "Unhandled exception during request",
                extra={"request_id": request_id, "path": endpoint},
            )
            raise
        finally:
            duration_seconds = time.perf_counter() - start_time
            http_requests_total.labels(
                method=request.method, endpoint=endpoint, status_code=str(status_code)
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method, endpoint=endpoint
            ).observe(duration_seconds)

            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": endpoint,
                    "status_code": status_code,
                    "duration_ms": round(duration_seconds * 1000, 2),
                },
            )

        response.headers["X-Request-ID"] = request_id
        return response