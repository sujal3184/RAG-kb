"""Admin business logic — system-wide views and management operations.

Unlike every other service in this app, these methods deliberately are
NOT scoped to a single owner: admins see across all users. Authorization
is enforced at the route layer via the `require_admin` dependency, which
is the single place that gate lives.
"""

import logging
import uuid

from app.core.exceptions import NotFoundError, ValidationError
from app.models.document import Document, DocumentStatus
from app.models.user import User, UserRole
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


class AdminService:
    """System-wide administrative operations."""

    def __init__(
        self,
        user_repo: UserRepository,
        kb_repo: KnowledgeBaseRepository,
        document_repo: DocumentRepository,
    ) -> None:
        """Store the repositories this service reads across.

        Args:
            user_repo: for user listing and status management.
            kb_repo: for knowledge base counts.
            document_repo: for document processing oversight.
        """
        self.user_repo = user_repo
        self.kb_repo = kb_repo
        self.document_repo = document_repo

    async def get_system_stats(self) -> dict:
        """Gather high-level counts across the whole system.

        Returns:
            A dict of statistics suitable for building SystemStatsResponse.
        """
        total_users = await self.user_repo.count_all()
        user_status_counts = await self.user_repo.count_by_status()
        total_kbs = await self.kb_repo.count_all()
        documents_by_status = await self.document_repo.count_by_status()

        return {
            "total_users": total_users,
            "active_users": user_status_counts["active"],
            "verified_users": user_status_counts["verified"],
            "admin_users": user_status_counts["admins"],
            "total_knowledge_bases": total_kbs,
            "documents_by_status": documents_by_status,
        }

    async def list_users(self, *, limit: int, offset: int) -> tuple[list[User], int]:
        """List all users with pagination.

        Returns:
            A tuple of (users, total_count).
        """
        users = await self.user_repo.list_all_paginated(limit=limit, offset=offset)
        total = await self.user_repo.count_all()
        return users, total

    async def set_user_active_status(
        self, *, user_id: uuid.UUID, is_active: bool, acting_admin_id: uuid.UUID
    ) -> User:
        """Activate or deactivate a user account.

        Args:
            user_id: the account to modify.
            is_active: the new status.
            acting_admin_id: the admin making the change — used to prevent
                an admin from accidentally locking themselves out.

        Raises:
            NotFoundError: if no such user exists.
            ValidationError: if an admin tries to deactivate their own account.
        """
        if user_id == acting_admin_id and not is_active:
            raise ValidationError(
                "You cannot deactivate your own account. Ask another admin to do it."
            )

        user = await self.user_repo.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User not found")

        user.is_active = is_active
        updated = await self.user_repo.update(user)

        logger.info(
            "Admin changed user active status",
            extra={
                "target_user_id": str(user_id),
                "is_active": is_active,
                "acting_admin_id": str(acting_admin_id),
            },
        )
        return updated

    async def list_documents_by_status(
        self, *, status: DocumentStatus, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        """List documents across all knowledge bases with a given status.

        Primarily used to review failures — a failed document's
        `error_message` explains what went wrong during processing.

        Returns:
            A tuple of (documents, total_count).
        """
        documents = await self.document_repo.list_by_status(status, limit=limit, offset=offset)
        total = await self.document_repo.count_with_status(status)
        return documents, total

    async def retry_document_processing(self, *, document_id: uuid.UUID) -> Document:
        """Re-enqueue a document for processing.

        Resets its status to PENDING and clears any previous error, then
        re-dispatches the SAME Celery task used during normal upload
        (Module 17) — no processing logic is duplicated here.

        Raises:
            NotFoundError: if no such document exists.
            ValidationError: if the document is currently being processed.
        """
        from app.tasks.document_processing import process_document

        document = await self.document_repo.get_by_id(document_id)
        if document is None:
            raise NotFoundError("Document not found")

        if document.status == DocumentStatus.PROCESSING:
            raise ValidationError(
                "This document is currently being processed. Wait for it to finish before retrying."
            )

        document.status = DocumentStatus.PENDING
        document.error_message = None
        updated = await self.document_repo.update(document)

        process_document.delay(str(document.id))

        logger.info("Admin retried document processing", extra={"document_id": str(document_id)})
        return updated

    async def promote_to_admin(self, *, user_id: uuid.UUID) -> User:
        """Grant a user admin privileges.

        Raises:
            NotFoundError: if no such user exists.
        """
        user = await self.user_repo.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User not found")

        user.role = UserRole.ADMIN
        updated = await self.user_repo.update(user)
        logger.info("User promoted to admin", extra={"user_id": str(user_id)})
        return updated