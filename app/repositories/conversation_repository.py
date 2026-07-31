"""Repository for Conversation rows."""

import uuid

from sqlalchemy import func, select

from app.models.conversation import Conversation
from app.repositories.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """Data access methods for conversations."""

    model = Conversation

    async def list_for_knowledge_base(
        self, knowledge_base_id: uuid.UUID, *, limit: int = 20, offset: int = 0
    ) -> list[Conversation]:
        """List conversations within a knowledge base, newest first."""
        stmt = (
            select(Conversation)
            .where(Conversation.knowledge_base_id == knowledge_base_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_for_knowledge_base(self, knowledge_base_id: uuid.UUID) -> int:
        """Count conversations within a knowledge base."""
        stmt = select(func.count()).select_from(Conversation).where(
            Conversation.knowledge_base_id == knowledge_base_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_for_knowledge_base(
        self, conversation_id: uuid.UUID, knowledge_base_id: uuid.UUID
    ) -> Conversation | None:
        """Fetch a conversation, scoped to a specific knowledge base."""
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.knowledge_base_id == knowledge_base_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()