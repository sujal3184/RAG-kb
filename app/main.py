"""App entrypoint.

`create_app()` builds the FastAPI app. We use a function (instead of just
writing `app = FastAPI()` directly) so that tests can create a fresh copy
of the app whenever they need one.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

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

logger = logging.getLogger(__name__)

# Maps our custom errors to HTTP status codes.
_EXCEPTION_STATUS_MAP: dict[type[AppException], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
    ValidationError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup and once at shutdown."""
    settings = get_settings()
    configure_logging(settings)
    logger.info("App starting", extra={"environment": settings.ENVIRONMENT.value})
    yield
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

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    _add_error_handlers(app)
    return app


def _add_error_handlers(app: FastAPI) -> None:
    """Turns our custom AppException errors into clean JSON error responses."""

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        status_code = _EXCEPTION_STATUS_MAP.get(type(exc), status.HTTP_400_BAD_REQUEST)
        logger.warning(
            "App error handled",
            extra={"type": type(exc).__name__, "path": str(request.url.path)},
        )
        return JSONResponse(
            status_code=status_code,
            content={"error": type(exc).__name__, "message": exc.message},
        )


app = create_app()