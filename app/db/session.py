"""Database engine and session management.

This file creates ONE database "engine" (a connection pool manager) for
the whole app's lifetime, and a factory for creating new database
"sessions" (a single unit-of-work, roughly: one request = one session).
"""

import logging

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    """Create the SQLAlchemy async engine.

    The engine manages a *pool* of database connections that get reused
    across requests, instead of opening a brand-new TCP connection to
    Postgres for every single query (which would be slow).

    Args:
        settings: app settings containing the database URL and pool sizing.

    Returns:
        A configured AsyncEngine, ready to create sessions from.
    """
    return create_async_engine(
        settings.database_url,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=True,  # checks a connection is alive before using it
        future=True,
    )


# Created once, at import time, using the cached settings singleton.
engine: AsyncEngine = create_engine(get_settings())

# A "factory" for creating new AsyncSession objects. `expire_on_commit=False`
# means objects we fetched stay usable after commit() — without this,
# accessing an attribute after commit would trigger a surprise extra query.
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)