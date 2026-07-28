"""Integration tests for authentication endpoints.

These go through the real HTTP layer (via the `client` fixture) and a
real database (via the `docker` marker), exercising the full chain:
route -> service -> repository -> database.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.docker


@pytest.mark.asyncio
async def test_register_creates_user(client: AsyncClient) -> None:
    """Registering with a new email should succeed and never leak the password."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": "supersecret123", "full_name": "Alice"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "alice@example.com"
    assert "password" not in body
    assert "hashed_password" not in body


@pytest.mark.asyncio
async def test_register_duplicate_email_fails(client: AsyncClient) -> None:
    """Registering the same email twice should fail with a conflict error."""
    payload = {"email": "bob@example.com", "password": "supersecret123"}
    await client.post("/api/v1/auth/register", json=payload)

    response = await client.post("/api/v1/auth/register", json=payload)

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_with_correct_credentials_returns_tokens(client: AsyncClient) -> None:
    """Logging in with the right password should return an access + refresh token."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "carol@example.com", "password": "correcthorse123"},
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "carol@example.com", "password": "correcthorse123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_with_wrong_password_fails(client: AsyncClient) -> None:
    """Logging in with the wrong password should return 401."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "dave@example.com", "password": "correcthorse123"},
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "dave@example.com", "password": "wrongpassword"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_requires_valid_token(client: AsyncClient) -> None:
    """Calling /me without a token should be rejected."""
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_profile_with_valid_token(client: AsyncClient) -> None:
    """Calling /me with a valid access token should return the user's profile."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "erin@example.com", "password": "correcthorse123"},
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": "erin@example.com", "password": "correcthorse123"},
    )
    access_token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"}
    )

    assert response.status_code == 200
    assert response.json()["email"] == "erin@example.com"


@pytest.mark.asyncio
async def test_refresh_returns_new_token_pair(client: AsyncClient) -> None:
    """Using a valid refresh token should return a brand new token pair."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "frank@example.com", "password": "correcthorse123"},
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": "frank@example.com", "password": "correcthorse123"},
    )
    old_refresh_token = login_response.json()["refresh_token"]

    response = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": old_refresh_token}
    )

    assert response.status_code == 200
    new_tokens = response.json()
    assert new_tokens["refresh_token"] != old_refresh_token  # rotated


@pytest.mark.asyncio
async def test_reusing_old_refresh_token_after_rotation_fails(client: AsyncClient) -> None:
    """Once a refresh token has been used (rotated), it can't be reused."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "grace@example.com", "password": "correcthorse123"},
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": "grace@example.com", "password": "correcthorse123"},
    )
    old_refresh_token = login_response.json()["refresh_token"]

    await client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh_token})
    second_attempt = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": old_refresh_token}
    )

    assert second_attempt.status_code == 401



@pytest.mark.asyncio
async def test_new_user_starts_unverified(client: AsyncClient) -> None:
    """A freshly registered user should have is_verified = False."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "henry@example.com", "password": "correcthorse123"},
    )
    assert response.json()["is_verified"] is False


@pytest.mark.asyncio
async def test_verify_email_with_invalid_token_fails(client: AsyncClient) -> None:
    """A made-up token should be rejected."""
    response = await client.post(
        "/api/v1/auth/verify-email", json={"token": "not-a-real-token"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_resend_verification_always_returns_success_message(client: AsyncClient) -> None:
    """Even for an unknown email, resend-verification should not error or leak info."""
    response = await client.post(
        "/api/v1/auth/resend-verification", json={"email": "nobody-here@example.com"}
    )
    assert response.status_code == 200
    assert "message" in response.json()


@pytest.mark.asyncio
async def test_forgot_password_always_returns_success_message(client: AsyncClient) -> None:
    """Even for an unknown email, forgot-password should not error or leak info."""
    response = await client.post(
        "/api/v1/auth/forgot-password", json={"email": "nobody-here@example.com"}
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_reset_password_with_invalid_token_fails(client: AsyncClient) -> None:
    """A made-up reset token should be rejected."""
    response = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": "not-a-real-token", "new_password": "newpassword123"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_full_password_reset_flow_via_repository(
    client: AsyncClient, db_session
) -> None:
    """End-to-end: register -> generate reset token directly (simulating
    clicking the emailed link) -> reset password -> old password stops
    working -> new password works.

    We fetch the token directly from the database instead of parsing logs,
    since the ConsoleEmailSender only logs the email during this test run.
    """
    from sqlalchemy import select

    from app.models.verification_token import TokenPurpose, VerificationToken

    await client.post(
        "/api/v1/auth/register",
        json={"email": "ivy@example.com", "password": "oldpassword123"},
    )

    await client.post("/api/v1/auth/forgot-password", json={"email": "ivy@example.com"})

    stmt = select(VerificationToken).where(
        VerificationToken.purpose == TokenPurpose.PASSWORD_RESET
    ).order_by(VerificationToken.created_at.desc())
    result = await db_session.execute(stmt)
    token_row = result.scalars().first()
    assert token_row is not None

    reset_response = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": token_row.token, "new_password": "newpassword456"},
    )
    assert reset_response.status_code == 200

    old_login = await client.post(
        "/api/v1/auth/login",
        json={"email": "ivy@example.com", "password": "oldpassword123"},
    )
    assert old_login.status_code == 401

    new_login = await client.post(
        "/api/v1/auth/login",
        json={"email": "ivy@example.com", "password": "newpassword456"},
    )
    assert new_login.status_code == 200