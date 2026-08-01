"""Document upload business logic.

Coordinates file validation, storage, and database record-keeping.
Ownership is enforced by first confirming the parent Knowledge Base
belongs to the current user (delegated to KnowledgeBaseService — we don't
duplicate that ownership check here).
"""

import logging
import uuid

from app.core.exceptions import ValidationError
from app.models.document import Document, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.services.knowledge_base_service import KnowledgeBaseService
from app.storage.base import FileStorage
from app.tasks.document_processing import process_document
from app.services.conversation_service import ConversationService


logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        document_repo: DocumentRepository,
        kb_service: KnowledgeBaseService,
        file_storage: FileStorage,
        *,
        max_upload_size_bytes: int,
        allowed_extensions: set[str],
        conversation_service: ConversationService | None = None,
    ) -> None:
        """... (existing docstring, extended)

        Args:
            conversation_service: optional, used ONLY to invalidate cached
                RAG responses when documents change. Kept optional so
                DocumentService remains usable/testable without needing
                the full conversation pipeline wired up.
        """
        self.document_repo = document_repo
        self.kb_service = kb_service
        self.file_storage = file_storage
        self.max_upload_size_bytes = max_upload_size_bytes
        self.allowed_extensions = allowed_extensions
        self.conversation_service = conversation_service

    async def upload(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        owner_id: uuid.UUID,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> Document:
        """Validate, store, and record a newly uploaded file.

        Raises:
            NotFoundError: if the knowledge base doesn't exist or isn't
                owned by this user (raised by kb_service.get).
            ValidationError: if the file extension isn't allowed, or the
                file exceeds the configured size limit.
        """
        # Confirms the KB exists AND belongs to this user — reusing
        # Module 5's service instead of re-implementing that check here.
        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)

        extension = self._extract_extension(filename)
        if extension not in self.allowed_extensions:
            raise ValidationError(
                f"File type '.{extension}' is not supported. "
                f"Allowed types: {', '.join(sorted(self.allowed_extensions))}"
            )

        if len(content) > self.max_upload_size_bytes:
            max_mb = self.max_upload_size_bytes / (1024 * 1024)
            raise ValidationError(f"File exceeds the maximum allowed size of {max_mb:.0f}MB")

        document_id = uuid.uuid4()
        storage_path = f"{owner_id}/{knowledge_base_id}/{document_id}_{filename}"
        storage_ref = await self.file_storage.save(path=storage_path, content=content)

        document = Document(
            id=document_id,
            knowledge_base_id=knowledge_base_id,
            original_filename=filename,
            file_extension=extension,
            content_type=content_type,
            size_bytes=len(content),
            storage_ref=storage_ref,
            status=DocumentStatus.PENDING,
        )
        created = await self.document_repo.create(document)
        logger.info(
            "Document uploaded",
            extra={"document_id": str(created.id), "kb_id": str(knowledge_base_id)},
        )

        process_document.delay(str(created.id))

        if self.conversation_service is not None:
            await self.conversation_service.invalidate_response_cache(
                knowledge_base_id=knowledge_base_id
            )

        return created


    
    async def list(
        self, *, knowledge_base_id: uuid.UUID, owner_id: uuid.UUID, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        """List documents in a knowledge base, after verifying ownership."""
        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)

        items = await self.document_repo.list_for_knowledge_base(
            knowledge_base_id, limit=limit, offset=offset
        )
        total = await self.document_repo.count_for_knowledge_base(knowledge_base_id)
        return items, total

    async def get(
        self, *, document_id: uuid.UUID, knowledge_base_id: uuid.UUID, owner_id: uuid.UUID
    ) -> Document:
        """Fetch a single document, after verifying ownership of its knowledge base."""
        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)
        return await self._get_or_404(document_id, knowledge_base_id)

    async def delete(
        self, *, document_id: uuid.UUID, knowledge_base_id: uuid.UUID, owner_id: uuid.UUID
    ) -> None:
        """Delete a document's file and database record, after verifying ownership."""
        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)
        document = await self._get_or_404(document_id, knowledge_base_id)

        await self.file_storage.delete(storage_ref=document.storage_ref)
        await self.document_repo.delete(document)
        logger.info("Document deleted", extra={"document_id": str(document_id)})

        if self.conversation_service is not None:
            await self.conversation_service.invalidate_response_cache(
                knowledge_base_id=knowledge_base_id
            )
            

    async def _get_or_404(
        self, document_id: uuid.UUID, knowledge_base_id: uuid.UUID
    ) -> Document:
        """Shared helper: fetch a document scoped to its KB, or raise 404."""
        from app.core.exceptions import NotFoundError

        document = await self.document_repo.get_for_knowledge_base(document_id, knowledge_base_id)
        if document is None:
            raise NotFoundError("Document not found")
        return document

    @staticmethod
    def _extract_extension(filename: str) -> str:
        """Extract a lowercase file extension without the leading dot."""
        if "." not in filename:
            return ""
        return filename.rsplit(".", 1)[-1].lower()