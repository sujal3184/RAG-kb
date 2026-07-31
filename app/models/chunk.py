"""Chunk database model.

Stores the text and position of each chunk produced during document
processing (Module 8), persisted here so BM25 keyword search (Module 11)
can load a knowledge base's full text corpus without re-parsing documents
or querying Qdrant just to get text back out.

Note: the actual EMBEDDING VECTOR lives in Qdrant (Module 10) — this
table exists purely to make chunk text queryable via Postgres for BM25
and other text-based needs.
"""

import uuid

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class Chunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single chunk of text extracted from a document."""

    __tablename__ = "chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return f"Chunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})"