"""Repository for Chunk rows."""

import uuid

from sqlalchemy import select

from app.models.chunk import Chunk
from app.repositories.base import BaseRepository


class ChunkRepository(BaseRepository[Chunk]):
    """Data access methods for chunks."""

    model = Chunk

    async def list_for_knowledge_base(self, knowledge_base_id: uuid.UUID) -> list[Chunk]:
        """Fetch all chunks belonging to a knowledge base — used to build
        the BM25 keyword search corpus (Module 11) for that KB."""
        stmt = select(Chunk).where(Chunk.knowledge_base_id == knowledge_base_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())