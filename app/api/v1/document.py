"""Document upload API endpoints.

All routes require a logged-in user, and are scoped to a specific
Knowledge Base — every operation verifies (via DocumentService, which
delegates to KnowledgeBaseService) that the current user owns that KB.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.api.dependencies import get_current_user, get_document_service
from app.models.user import User
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/knowledge-bases/{knowledge_base_id}/documents", tags=["documents"]
)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document into a knowledge base",
)
async def upload_document(
    knowledge_base_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    file: Annotated[UploadFile, File(description="The file to upload")],
) -> DocumentResponse:
    """Upload a file into the given knowledge base.

    The file is validated (extension + size), stored via the configured
    FileStorage backend, and recorded with status "pending" — actual text
    extraction/chunking/embedding happens later (Module 17: background workers).
    """
    content = await file.read()
    document = await document_service.upload(
        knowledge_base_id=knowledge_base_id,
        owner_id=current_user.id,
        filename=file.filename or "unnamed_file",
        content_type=file.content_type or "application/octet-stream",
        content=content,
    )
    return DocumentResponse.model_validate(document)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents in a knowledge base",
)
async def list_documents(
    knowledge_base_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """List documents within a knowledge base, paginated."""
    items, total = await document_service.list(
        knowledge_base_id=knowledge_base_id,
        owner_id=current_user.id,
        limit=limit,
        offset=offset,
    )
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a single document's metadata",
)
async def get_document(
    knowledge_base_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    """Get metadata for a single document."""
    document = await document_service.get(
        document_id=document_id, knowledge_base_id=knowledge_base_id, owner_id=current_user.id
    )
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
async def delete_document(
    knowledge_base_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
) -> None:
    """Delete a document — removes both its stored file and database record."""
    await document_service.delete(
        document_id=document_id, knowledge_base_id=knowledge_base_id, owner_id=current_user.id
    )