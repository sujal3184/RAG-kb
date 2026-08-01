"""Application entrypoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
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
from app.middleware.request_logging import RequestObservabilityMiddleware
from app.observability.langfuse_client import get_langfuse_tracer
from app.observability.tracing import configure_tracing

logger = logging.getLogger(__name__)

_EXCEPTION_STATUS_MAP: dict[type[AppException], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
    ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    settings = get_settings()
    configure_logging(settings)
    configure_tracing(settings, app=app)
    logger.info("App starting", extra={"environment": settings.ENVIRONMENT.value})
    yield
    # Flush any buffered LLM traces before the process exits.
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

    app.add_middleware(RequestObservabilityMiddleware)
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    _add_error_handlers(app)

    if settings.METRICS_ENABLED:
        _add_metrics_endpoint(app)

    return app


def _add_metrics_endpoint(app: FastAPI) -> None:
    """Expose Prometheus metrics for scraping.

    Deliberately NOT under /api/v1 — metrics are infrastructure, not part
    of the public API surface, and are typically restricted to an internal
    network in production (see Module 22: production hardening).
    """

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _add_error_handlers(app: FastAPI) -> None:
    """Turns our custom AppException errors into clean JSON error responses."""

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


app = create_app()