"""Message database model.

Stores one turn of a conversation — either the user's question or the
assistant's reply. Shape mirrors app.llm.base.ChatMessage/MessageRole so
converting stored history into prompt-ready messages (Module 14) is a
direct, lossless mapping.
"""

import uuid
from enum import StrEnum

from sqlalchemy import Enum as SQLEnum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class MessageRole(StrEnum):
    """Who sent this message."""

    USER = "user"
    ASSISTANT = "assistant"


class Message(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single message within a conversation."""

    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(SQLEnum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return f"Message(id={self.id}, role={self.role}, conversation_id={self.conversation_id})"