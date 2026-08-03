"""Application entrypoint."""

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.v1.router import api_router
from app.config.settings import get_settings
from app.core.exceptions import (
    AppException,
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from app.core.logging import configure_logging
from app.guardrails.exceptions import GuardrailViolationError
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.request_logging import RequestObservabilityMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.observability.langfuse_client import get_langfuse_tracer
from app.observability.tracing import configure_tracing

logger = logging.getLogger(__name__)

_EXCEPTION_STATUS_MAP: dict[type[AppException], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
    ValidationError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
    GuardrailViolationError: status.HTTP_400_BAD_REQUEST,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    settings = get_settings()
    configure_logging(settings)
    configure_tracing(settings, app=app)
    logger.info("App starting", extra={"environment": settings.ENVIRONMENT.value})
    yield
    get_langfuse_tracer().flush()
    logger.info("App shutting down")


def create_app() -> FastAPI:
    """Build and return the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        docs_url="/docs" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Middleware order matters: the LAST added runs FIRST on the way in.
    # Security headers should wrap everything (including error responses),
    # so it's added last; rate limiting should reject before any real work.
    app.add_middleware(RequestObservabilityMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        enabled=settings.RATE_LIMIT_ENABLED,
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        auth_requests_per_minute=settings.RATE_LIMIT_AUTH_REQUESTS_PER_MINUTE,
    )

    # Strict CSP would break Swagger UI, which loads external scripts and
    # styles — so it's only applied in production, where /docs is disabled.
    app.add_middleware(
        SecurityHeadersMiddleware,
        enabled=settings.SECURITY_HEADERS_ENABLED and settings.is_production,
    )

    if settings.cors_origins_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    _add_error_handlers(app)

    if settings.METRICS_ENABLED:
        _add_metrics_endpoint(app, settings.METRICS_AUTH_TOKEN)

    return app


def _add_metrics_endpoint(app: FastAPI, auth_token: str) -> None:
    """Expose Prometheus metrics, optionally protected by a bearer token.

    A token is what the application itself can enforce. It is NOT a
    substitute for restricting /metrics at the network layer — see
    DEPLOYMENT.md.
    """

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        if auth_token:
            provided = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            # Constant-time comparison avoids leaking the token through
            # response-timing differences.
            if not secrets.compare_digest(provided, auth_token):
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _add_error_handlers(app: FastAPI) -> None:
    """Translate application exceptions into JSON HTTP responses."""

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        status_code = _EXCEPTION_STATUS_MAP.get(type(exc), status.HTTP_400_BAD_REQUEST)
        logger.warning(
            "App error handled",
            extra={
                "type": type(exc).__name__,
                "path": str(request.url.path),
                "request_id": getattr(request.state, "request_id", None),
            },
        )
        return JSONResponse(
            status_code=status_code,
            content={"error": type(exc).__name__, "message": exc.message},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for unhandled errors.

        Logs the full traceback server-side but returns a generic message
        to the client — internal error details (stack traces, SQL, file
        paths) must never reach users.
        """
        logger.exception(
            "Unhandled exception",
            extra={
                "path": str(request.url.path),
                "request_id": getattr(request.state, "request_id", None),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "message": "An unexpected error occurred."},
        )


app = create_app()