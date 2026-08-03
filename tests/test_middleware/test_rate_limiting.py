"""Tests for rate limiting middleware."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.middleware.rate_limiting import RateLimitMiddleware


def _app_with_limit(limit: int, auth_limit: int = 5) -> FastAPI:
    """Build a minimal app with rate limiting at a known low limit."""
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        enabled=True,
        requests_per_minute=limit,
        auth_requests_per_minute=auth_limit,
    )

    @app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    @app.post("/api/v1/auth/login")
    async def login_endpoint():
        return {"ok": True}

    return app


@pytest.mark.asyncio
async def test_allows_requests_under_the_limit() -> None:
    app = _app_with_limit(limit=5)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(5):
            response = await client.get("/test")
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_blocks_requests_over_the_limit() -> None:
    app = _app_with_limit(limit=3)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            await client.get("/test")

        response = await client.get("/test")

        assert response.status_code == 429
        assert "Retry-After" in response.headers


@pytest.mark.asyncio
async def test_auth_endpoints_have_a_stricter_limit() -> None:
    """Auth endpoints should reject sooner than general endpoints."""
    app = _app_with_limit(limit=100, auth_limit=2)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/v1/auth/login")
        await client.post("/api/v1/auth/login")

        response = await client.post("/api/v1/auth/login")

        assert response.status_code == 429


@pytest.mark.asyncio
async def test_disabled_limiter_allows_everything() -> None:
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware, enabled=False, requests_per_minute=1, auth_requests_per_minute=1
    )

    @app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(10):
            response = await client.get("/test")
            assert response.status_code == 200