"""FastAPI dependency for getting a database session per request.

Usage in a route or repository:

    async def some_endpoint(db: Annotated[AsyncSession, Depends(get_db_session)]):
        ...
"""

import logging
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory

logger = logging.getLogger(__name__)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield a database session, and guarantee cleanup afterwards.

    This function is a FastAPI dependency. FastAPI will:
    1. Run the code before `yield` (open the session).
    2. Pass the session into your route/service.
    3. After the request finishes, resume this function to run the code
       after `yield` (commit or rollback, then close).

    On success: commit any pending changes.
    On error: roll back so a half-finished operation never gets saved.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Database session rolled back due to an error")
            raise
        finally:
            await session.close()