"""Repository for Message rows."""

import uuid

from sqlalchemy import select

from app.models.message import Message
from app.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Data access methods for messages."""

    model = Message

    async def list_for_conversation(
        self, conversation_id: uuid.UUID, *, limit: int | None = None
    ) -> list[Message]:
        """Fetch messages in a conversation, oldest first.

        Args:
            conversation_id: which conversation to fetch messages for.
            limit: if provided, return only the MOST RECENT `limit`
                messages (still returned oldest-first among themselves) —
                used to cap conversation history sent to the LLM.
        """
        stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(
            Message.created_at.asc()
        )
        result = await self.session.execute(stmt)
        messages = list(result.scalars().all())

        if limit is not None and len(messages) > limit:
            return messages[-limit:]
        return messages