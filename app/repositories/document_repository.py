"""Repository for Document rows."""

import uuid

from sqlalchemy import func, select

from app.models.document import Document
from app.repositories.base import BaseRepository
from app.models.document import Document, DocumentStatus

class DocumentRepository(BaseRepository[Document]):
    """Data access methods for documents."""

    model = Document

    async def list_for_knowledge_base(
        self, knowledge_base_id: uuid.UUID, *, limit: int = 20, offset: int = 0
    ) -> list[Document]:
        """List documents within a specific knowledge base, newest first."""
        stmt = (
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_for_knowledge_base(self, knowledge_base_id: uuid.UUID) -> int:
        """Count documents within a specific knowledge base."""
        stmt = select(func.count()).select_from(Document).where(
            Document.knowledge_base_id == knowledge_base_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_for_knowledge_base(
        self, document_id: uuid.UUID, knowledge_base_id: uuid.UUID
    ) -> Document | None:
        """Fetch a document, scoped to a specific knowledge base.

        Returns None if the document doesn't exist OR belongs to a
        different knowledge base — same "don't reveal which" pattern used
        for knowledge base ownership in Module 5.
        """
        stmt = select(Document).where(
            Document.id == document_id, Document.knowledge_base_id == knowledge_base_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


    async def update_status(
        self, document_id, status: DocumentStatus, *, error_message: str | None = None
    ) -> None:
        """Update a document's processing status (and optional error message).

        Used by the background processing task (Module 17) to record
        progress: pending -> processing -> ready/failed.
        """
        document = await self.get_by_id(document_id)
        if document is None:
            return
        document.status = status
        document.error_message = error_message
        await self.update(document)



    async def count_by_status(self) -> dict[str, int]:
        """Count documents grouped by processing status, across all KBs."""
        stmt = select(Document.status, func.count()).group_by(Document.status)
        result = await self.session.execute(stmt)
        return {status.value: count for status, count in result.all()}

    async def list_by_status(
        self, status: DocumentStatus, *, limit: int = 50, offset: int = 0
    ) -> list[Document]:
        """List documents with a given processing status, across all KBs.

        Primarily used by admins to review failed documents and their
        error messages.
        """
        stmt = (
            select(Document)
            .where(Document.status == status)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_with_status(self, status: DocumentStatus) -> int:
        """Count documents with a specific status, across all KBs."""
        stmt = select(func.count()).select_from(Document).where(Document.status == status)
        result = await self.session.execute(stmt)
        return result.scalar_one()