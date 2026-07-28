"""Shared test setup.

For database tests, we use a real Postgres connection (via Docker) but
wrap each test in a transaction that gets ROLLED BACK at the end — so
tests never leave leftover data behind, and tests can run in any order.
"""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory, engine
from app.main import create_app


@pytest.fixture
def app():
    """Fresh app instance for each test."""
    return create_app()


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    """A fake HTTP client that talks to the app without a real server."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Provide a database session wrapped in a transaction that always
    rolls back, keeping the test database clean between tests.
    """
    async with engine.connect() as connection:
        await connection.begin()
        async with async_session_factory(bind=connection) as session:
            yield session
            await session.rollback()



@pytest.fixture(autouse=True)
async def _dispose_engine_after_test() -> AsyncIterator[None]:
    """Dispose the shared engine's connection pool after every test.

    On Windows, asyncpg connections are tied to the event loop that created
    them. pytest-asyncio creates a NEW event loop for every test function,
    so a connection opened during one test becomes unusable (and crashes
    on cleanup) once the next test's event loop starts. Disposing the pool
    after each test forces the next test to open fresh connections on its
    own event loop, avoiding this cross-loop connection reuse entirely.
    """
    yield
    await engine.dispose()