"""Document database model.

Represents one uploaded file within a Knowledge Base. The `status` field
tracks its lifecycle: uploaded now (pending), processed later (Modules
7-10 extract text, chunk it, embed it), ready to search, or failed.
"""

import uuid
from enum import StrEnum

from sqlalchemy import BigInteger, Enum as SQLEnum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class DocumentStatus(StrEnum):
    """Lifecycle states of an uploaded document."""

    PENDING = "pending"        # uploaded, not yet processed
    PROCESSING = "processing"  # being parsed/chunked/embedded (Module 17)
    READY = "ready"            # fully processed, searchable
    FAILED = "failed"          # processing failed


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """An uploaded file within a knowledge base."""

    __tablename__ = "documents"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(20), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_ref: Mapped[str] = mapped_column(
        String(1000), nullable=False, doc="Reference used by FileStorage to locate this file"
    )
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True, doc="Populated if status is FAILED"
    )

    def __repr__(self) -> str:
        return f"Document(id={self.id}, filename={self.original_filename!r}, status={self.status})"