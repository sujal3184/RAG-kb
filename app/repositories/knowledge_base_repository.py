"""Repository for KnowledgeBase rows."""

import uuid

from sqlalchemy import func, select

from app.models.knowledge_base import KnowledgeBase
from app.repositories.base import BaseRepository


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    """Data access methods for knowledge bases."""

    model = KnowledgeBase

    async def list_for_user(
        self, owner_id: uuid.UUID, *, limit: int = 20, offset: int = 0
    ) -> list[KnowledgeBase]:
        """List knowledge bases belonging to a specific user, paginated.

        Ordered newest-first, which is almost always what users expect to
        see at the top of a list like this.
        """
        stmt = (
            select(KnowledgeBase)
            .where(KnowledgeBase.owner_id == owner_id)
            .order_by(KnowledgeBase.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_for_user(self, owner_id: uuid.UUID) -> int:
        """Count how many knowledge bases a user owns (used for pagination metadata)."""
        stmt = select(func.count()).select_from(KnowledgeBase).where(
            KnowledgeBase.owner_id == owner_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_for_user(
        self, kb_id: uuid.UUID, owner_id: uuid.UUID
    ) -> KnowledgeBase | None:
        """Fetch a single knowledge base, ONLY if it belongs to this user.

        Returns None both when the id doesn't exist at all AND when it
        exists but belongs to someone else — the caller can't tell which,
        which is exactly the point (see Module 5 design notes on 404 vs 403).
        """
        stmt = select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.owner_id == owner_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()