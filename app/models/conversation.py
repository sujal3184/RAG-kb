"""Conversation database model.

A conversation belongs to one Knowledge Base and contains an ordered
sequence of Messages. Ownership is enforced transitively through the
parent KnowledgeBase's owner — no separate owner_id needed here.
"""

import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class Conversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A multi-turn conversation scoped to a single knowledge base."""

    __tablename__ = "conversations"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(
        String(255), nullable=True, doc="Optional display title, e.g. auto-generated from first message"
    )

    def __repr__(self) -> str:
        return f"Conversation(id={self.id}, knowledge_base_id={self.knowledge_base_id})"