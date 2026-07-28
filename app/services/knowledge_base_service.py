"""Knowledge Base business logic.

Enforces the core ownership rule: a user may only view, update, or delete
knowledge bases THEY created. This rule lives here — once — rather than
being repeated (and potentially forgotten) in every route.
"""

import logging
import uuid

from app.core.exceptions import NotFoundError
from app.models.knowledge_base import KnowledgeBase
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Handles create/read/update/delete for knowledge bases, scoped to an owner."""

    def __init__(self, kb_repo: KnowledgeBaseRepository) -> None:
        """Store the repository this service uses.

        Args:
            kb_repo: repository for reading/writing KnowledgeBase rows.
        """
        self.kb_repo = kb_repo

    async def create(
        self, *, owner_id: uuid.UUID, name: str, description: str | None
    ) -> KnowledgeBase:
        """Create a new knowledge base owned by the given user."""
        kb = KnowledgeBase(owner_id=owner_id, name=name, description=description)
        created = await self.kb_repo.create(kb)
        logger.info(
            "Knowledge base created",
            extra={"kb_id": str(created.id), "owner_id": str(owner_id)},
        )
        return created

    async def get(self, *, kb_id: uuid.UUID, owner_id: uuid.UUID) -> KnowledgeBase:
        """Fetch a single knowledge base owned by this user.

        Raises:
            NotFoundError: if it doesn't exist OR belongs to someone else
                (see repository docstring — these look identical on purpose).
        """
        return await self._get_owned_or_404(kb_id, owner_id)

    async def list(
        self, *, owner_id: uuid.UUID, limit: int, offset: int
    ) -> tuple[list[KnowledgeBase], int]:
        """List a user's knowledge bases, paginated.

        Returns:
            A tuple of (items, total_count) — the total count lets the
            client render pagination controls (e.g. "page 2 of 5").
        """
        items = await self.kb_repo.list_for_user(owner_id, limit=limit, offset=offset)
        total = await self.kb_repo.count_for_user(owner_id)
        return items, total

    async def update(
        self,
        *,
        kb_id: uuid.UUID,
        owner_id: uuid.UUID,
        name: str | None,
        description: str | None,
    ) -> KnowledgeBase:
        """Update a knowledge base's name and/or description.

        Only fields that are not None get changed — this supports partial
        updates (e.g. the client only wants to rename it, not touch the
        description).

        Raises:
            NotFoundError: if it doesn't exist or belongs to someone else.
        """
        kb = await self._get_owned_or_404(kb_id, owner_id)

        if name is not None:
            kb.name = name
        if description is not None:
            kb.description = description

        updated = await self.kb_repo.update(kb)
        logger.info("Knowledge base updated", extra={"kb_id": str(kb_id)})
        return updated

    async def delete(self, *, kb_id: uuid.UUID, owner_id: uuid.UUID) -> None:
        """Delete a knowledge base.

        Raises:
            NotFoundError: if it doesn't exist or belongs to someone else.
        """
        kb = await self._get_owned_or_404(kb_id, owner_id)
        await self.kb_repo.delete(kb)
        logger.info("Knowledge base deleted", extra={"kb_id": str(kb_id)})

    async def _get_owned_or_404(
        self, kb_id: uuid.UUID, owner_id: uuid.UUID
    ) -> KnowledgeBase:
        """Shared helper: fetch a KB by id, scoped to its owner, or raise 404.

        Every method that needs "this KB, and it must be mine" calls this
        ONE method, instead of repeating the same check four times.
        """
        kb = await self.kb_repo.get_for_user(kb_id, owner_id)
        if kb is None:
            raise NotFoundError("Knowledge base not found")
        return kb