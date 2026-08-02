"""Integration tests for admin endpoints, focusing on authorization
boundaries — the most important thing to verify for admin-only routes."""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.user import User, UserRole

pytestmark = pytest.mark.docker


async def _register_and_login(client: AsyncClient, email: str) -> str:
    """Register a regular user and return their access token."""
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correcthorse123"}
    )
    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "correcthorse123"}
    )
    return response.json()["access_token"]


async def _make_admin(client: AsyncClient, email: str) -> str:
    """Register a user, promote them to admin directly in the database,
    then log in and return their token."""
    from app.db.session import async_session_factory

    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correcthorse123"}
    )

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one()
        user.role = UserRole.ADMIN
        await session.commit()

    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "correcthorse123"}
    )
    return response.json()["access_token"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_regular_user_cannot_access_admin_stats(client: AsyncClient) -> None:
    """A logged-in non-admin should get 403, not 200 or 404."""
    token = await _register_and_login(client, f"regular_{uuid.uuid4()}@example.com")

    response = await client.get("/api/v1/admin/stats", headers=_headers(token))

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_request_is_rejected(client: AsyncClient) -> None:
    """No token at all should be 401, not 403."""
    response = await client.get("/api/v1/admin/stats")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_can_view_system_stats(client: AsyncClient) -> None:
    token = await _make_admin(client, f"admin_{uuid.uuid4()}@example.com")

    response = await client.get("/api/v1/admin/stats", headers=_headers(token))

    assert response.status_code == 200
    body = response.json()
    assert body["total_users"] >= 1
    assert "documents_by_status" in body


@pytest.mark.asyncio
async def test_admin_can_list_users(client: AsyncClient) -> None:
    token = await _make_admin(client, f"admin_{uuid.uuid4()}@example.com")

    response = await client.get("/api/v1/admin/users", headers=_headers(token))

    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert all("role" in item for item in body["items"])


@pytest.mark.asyncio
async def test_admin_can_deactivate_another_user(client: AsyncClient) -> None:
    admin_token = await _make_admin(client, f"admin_{uuid.uuid4()}@example.com")
    target_email = f"target_{uuid.uuid4()}@example.com"
    await _register_and_login(client, target_email)

    users_response = await client.get("/api/v1/admin/users", headers=_headers(admin_token))
    target_id = next(
        u["id"] for u in users_response.json()["items"] if u["email"] == target_email
    )

    response = await client.patch(
        f"/api/v1/admin/users/{target_id}/status",
        json={"is_active": False},
        headers=_headers(admin_token),
    )

    assert response.status_code == 200
    assert response.json()["is_active"] is False


@pytest.mark.asyncio
async def test_deactivated_user_cannot_log_in(client: AsyncClient) -> None:
    """Deactivation should actually block access, not just set a flag."""
    admin_token = await _make_admin(client, f"admin_{uuid.uuid4()}@example.com")
    target_email = f"target_{uuid.uuid4()}@example.com"
    await _register_and_login(client, target_email)

    users_response = await client.get("/api/v1/admin/users", headers=_headers(admin_token))
    target_id = next(
        u["id"] for u in users_response.json()["items"] if u["email"] == target_email
    )
    await client.patch(
        f"/api/v1/admin/users/{target_id}/status",
        json={"is_active": False},
        headers=_headers(admin_token),
    )

    login_response = await client.post(
        "/api/v1/auth/login", json={"email": target_email, "password": "correcthorse123"}
    )

    assert login_response.status_code == 401


@pytest.mark.asyncio
async def test_admin_cannot_deactivate_themselves(client: AsyncClient) -> None:
    """Prevents an admin locking themselves out of the system."""
    admin_email = f"admin_{uuid.uuid4()}@example.com"
    admin_token = await _make_admin(client, admin_email)

    users_response = await client.get("/api/v1/admin/users", headers=_headers(admin_token))
    own_id = next(
        u["id"] for u in users_response.json()["items"] if u["email"] == admin_email
    )

    response = await client.patch(
        f"/api/v1/admin/users/{own_id}/status",
        json={"is_active": False},
        headers=_headers(admin_token),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_admin_can_list_failed_documents(client: AsyncClient) -> None:
    token = await _make_admin(client, f"admin_{uuid.uuid4()}@example.com")

    response = await client.get(
        "/api/v1/admin/documents?status=failed", headers=_headers(token)
    )

    assert response.status_code == 200
    assert "items" in response.json()


@pytest.mark.asyncio
async def test_regular_user_cannot_promote_users(client: AsyncClient) -> None:
    """Privilege escalation must be blocked."""
    token = await _register_and_login(client, f"regular_{uuid.uuid4()}@example.com")

    response = await client.post(
        f"/api/v1/admin/users/{uuid.uuid4()}/promote", headers=_headers(token)
    )

    assert response.status_code == 403