"""Tests for security headers middleware."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.middleware.security_headers import SecurityHeadersMiddleware


def _app(enabled: bool) -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, enabled=enabled)

    @app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    return app


@pytest.mark.asyncio
async def test_adds_security_headers_when_enabled() -> None:
    transport = ASGITransport(app=_app(enabled=True))

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers


@pytest.mark.asyncio
async def test_omits_headers_when_disabled() -> None:
    transport = ASGITransport(app=_app(enabled=False))

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test")

    assert "X-Content-Type-Options" not in response.headers