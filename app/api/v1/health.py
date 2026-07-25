"""Health check endpoint.

Used to check the app is alive — for example, Docker uses this to know
when the container is ready.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config.settings import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """What the /health endpoint returns."""

    status: str
    app_name: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Check if the app is alive")
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Return a simple 'I'm alive' response."""
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        environment=settings.ENVIRONMENT.value,
    )