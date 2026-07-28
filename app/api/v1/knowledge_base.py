"""Knowledge Base API endpoints.

All routes here require a logged-in user (via `get_current_user`), and
every operation is automatically scoped to that user's own knowledge
bases — enforced by KnowledgeBaseService, not repeated in each route.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import get_current_user, get_knowledge_base_service
from app.models.user import User
from app.schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdateRequest,
)
from app.services.knowledge_base_service import KnowledgeBaseService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post(
    "",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new knowledge base",
)
async def create_knowledge_base(
    body: KnowledgeBaseCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    kb_service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> KnowledgeBaseResponse:
    """Create a knowledge base owned by the currently logged-in user."""
    kb = await kb_service.create(
        owner_id=current_user.id, name=body.name, description=body.description
    )
    return KnowledgeBaseResponse.model_validate(kb)


@router.get(
    "",
    response_model=KnowledgeBaseListResponse,
    summary="List your knowledge bases",
)
async def list_knowledge_bases(
    current_user: Annotated[User, Depends(get_current_user)],
    kb_service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> KnowledgeBaseListResponse:
    """List knowledge bases owned by the currently logged-in user, paginated."""
    items, total = await kb_service.list(owner_id=current_user.id, limit=limit, offset=offset)
    return KnowledgeBaseListResponse(
        items=[KnowledgeBaseResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{kb_id}",
    response_model=KnowledgeBaseResponse,
    summary="Get a single knowledge base",
)
async def get_knowledge_base(
    kb_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    kb_service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> KnowledgeBaseResponse:
    """Get one knowledge base — only if it belongs to the current user."""
    kb = await kb_service.get(kb_id=kb_id, owner_id=current_user.id)
    return KnowledgeBaseResponse.model_validate(kb)


@router.patch(
    "/{kb_id}",
    response_model=KnowledgeBaseResponse,
    summary="Update a knowledge base's name or description",
)
async def update_knowledge_base(
    kb_id: uuid.UUID,
    body: KnowledgeBaseUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    kb_service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> KnowledgeBaseResponse:
    """Partially update a knowledge base — only fields provided are changed."""
    kb = await kb_service.update(
        kb_id=kb_id, owner_id=current_user.id, name=body.name, description=body.description
    )
    return KnowledgeBaseResponse.model_validate(kb)


@router.delete(
    "/{kb_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a knowledge base",
)
async def delete_knowledge_base(
    kb_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    kb_service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> None:
    """Delete a knowledge base — only if it belongs to the current user."""
    await kb_service.delete(kb_id=kb_id, owner_id=current_user.id)