"""Shared test fixtures with proper isolation.

Every test runs inside a database transaction that is ROLLED BACK when
the test finishes — so tests never see each other's data, can assert on
absolute counts, and leave the database exactly as they found it.

The key piece is overriding FastAPI's `get_db_session` dependency so that
API requests made during a test use the SAME transaction the test itself
is using. Without that override, the app would open its own connection
and couldn't see the test's uncommitted setup data.
"""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.dependencies import get_db_session
from app.db.session import engine
from app.main import create_app


@pytest.fixture(autouse=True)
async def _dispose_engine_after_test() -> AsyncIterator[None]:
    """Dispose the engine's connection pool after every test.

    On Windows, asyncpg connections are bound to the event loop that
    created them, and pytest-asyncio creates a new loop per test — so a
    pooled connection from a previous test crashes when reused. Disposing
    forces fresh connections per test.
    """
    yield
    await engine.dispose()


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Provide a database session inside a transaction that always rolls back.

    Uses an explicit outer transaction on a single connection: everything
    the test does (setup data, API calls, assertions) happens inside it,
    and the rollback at the end undoes all of it.
    """
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session_factory = async_sessionmaker(bind=connection, expire_on_commit=False)

        async with session_factory() as session:
            yield session

        await transaction.rollback()


@pytest.fixture
def app(db_session: AsyncSession):
    """Provide a FastAPI app whose database dependency is overridden to
    use the test's transaction.

    This is what makes API tests properly isolated: a request handled by
    the app sees the test's setup data (uncommitted, but visible on the
    same connection), and anything the request writes is rolled back with
    everything else.
    """
    application = create_app()

    async def _override_get_db_session() -> AsyncIterator[AsyncSession]:
        # Deliberately does NOT commit — the test's outer transaction
        # controls the lifecycle, and rolls everything back at the end.
        yield db_session

    application.dependency_overrides[get_db_session] = _override_get_db_session
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    """An async HTTP client bound to the app, without a real server."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _mock_celery_task(monkeypatch):
    """Prevent tests from dispatching real Celery tasks.

    DocumentService.upload() enqueues process_document, which needs a
    reachable Redis broker. Tests run on the host where the broker
    hostname ('redis') doesn't resolve, and dispatching a real background
    job isn't what these tests are verifying anyway.
    """
    from app.tasks import document_processing

    monkeypatch.setattr(document_processing.process_document, "delay", lambda *a, **kw: None)