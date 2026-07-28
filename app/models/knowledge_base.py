"""KnowledgeBase database model.

A KnowledgeBase is a named collection of documents, owned by exactly one
user. Future modules (Document Upload, Chunking, Embeddings, Retrieval)
will all attach their data to a specific KnowledgeBase.
"""

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class KnowledgeBase(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A named collection of documents belonging to one user."""

    __tablename__ = "knowledge_bases"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"KnowledgeBase(id={self.id}, name={self.name!r}, owner_id={self.owner_id})"