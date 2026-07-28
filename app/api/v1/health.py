"""Health check endpoints.

Two kinds of health checks (a standard production practice):
- Liveness (`/health`): "is the process running at all?" No dependencies
  checked. Used by Docker/Kubernetes to decide whether to restart the pod.
- Readiness (`/health/ready`): "is the app ready to serve real traffic?"
  Checks that the database is actually reachable.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.db.dependencies import get_db_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    database: str


@router.get("/health", response_model=HealthResponse, summary="Check if the app is alive")
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Basic liveness check — no external dependencies checked."""
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        environment=settings.ENVIRONMENT.value,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Check if the app is ready to serve traffic (DB reachable)",
)
async def readiness_check(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> ReadinessResponse:
    """Readiness check — actually queries the database.

    Runs `SELECT 1`, the standard, cheapest possible query used purely to
    confirm the database connection is alive.
    """
    await db.execute(text("SELECT 1"))
    return ReadinessResponse(status="ok", database="reachable")