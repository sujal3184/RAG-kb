"""Tests for UserRepository.

Requires a real Postgres database (via Docker) — these are integration
tests, not pure unit tests, because repositories only make sense when
verified against a real database engine.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.user_repository import UserRepository

pytestmark = pytest.mark.docker


@pytest.mark.asyncio
async def test_create_and_get_user_by_id(db_session: AsyncSession) -> None:
    """A created user should be retrievable by their id."""
    repo = UserRepository(db_session)

    user = User(email="test@example.com", hashed_password="hashed", full_name="Test User")
    created = await repo.create(user)

    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.email == "test@example.com"
    assert fetched.is_active is True  # default value


@pytest.mark.asyncio
async def test_get_by_email_returns_none_when_missing(db_session: AsyncSession) -> None:
    """Looking up a non-existent email should return None, not raise."""
    repo = UserRepository(db_session)

    result = await repo.get_by_email("nobody@example.com")

    assert result is None


@pytest.mark.asyncio
async def test_get_by_email_finds_existing_user(db_session: AsyncSession) -> None:
    """A user should be findable by their exact email."""
    repo = UserRepository(db_session)
    await repo.create(User(email="findme@example.com", hashed_password="hashed"))

    result = await repo.get_by_email("findme@example.com")

    assert result is not None
    assert result.email == "findme@example.com"