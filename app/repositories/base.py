"""Generic base repository.

Implements common CRUD (Create, Read, Update, Delete) operations ONCE,
using Python generics, so every table-specific repository gets these for
free by inheriting from `BaseRepository[MyModel]`.

Why a repository at all, instead of calling SQLAlchemy directly in
services? It keeps ALL query code for a table in one file. If we later
need to add caching, change a query, or even swap databases, we touch
only the repository — services and API routes never change.
"""

import uuid
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

# `ModelType` is a stand-in for "any SQLAlchemy model class that inherits
# from Base". This is what makes BaseRepository reusable for User, and
# later for Document, Chunk, etc., while keeping full type-checking.
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Generic repository providing common database operations.

    Subclasses only need to set `model` and add any table-specific query
    methods (e.g. `get_by_email` for users).
    """

    model: type[ModelType]

    def __init__(self, session: AsyncSession) -> None:
        """Store the database session this repository will use.

        Args:
            session: an active AsyncSession, usually injected per-request
                via `get_db_session`.
        """
        self.session = session

    async def get_by_id(self, entity_id: uuid.UUID) -> ModelType | None:
        """Fetch a single row by its primary key, or None if not found."""
        return await self.session.get(self.model, entity_id)

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> list[ModelType]:
        """Fetch multiple rows with basic pagination."""
        stmt = select(self.model).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, entity: ModelType) -> ModelType:
        """Insert a new row.

        We `flush` (not `commit`) here — `flush` sends the SQL to the
        database and populates auto-generated fields (like `id`), but
        keeps the transaction open. Committing is the caller's/session
        dependency's job, so multiple repository calls can be grouped into
        one atomic transaction if needed.
        """
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def update(self, entity: ModelType) -> ModelType:
        """Save changes made to an already-fetched entity."""
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity: ModelType) -> None:
        """Remove a row from the database."""
        await self.session.delete(entity)
        await self.session.flush()