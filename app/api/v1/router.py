"""Combines all v1 API routes into one router.

When we add new features (e.g. Knowledge Base endpoints in Module 5), we
just register their router here — `main.py` never needs to change.
"""

from fastapi import APIRouter

from app.api.v1 import health

api_router = APIRouter()
api_router.include_router(health.router)