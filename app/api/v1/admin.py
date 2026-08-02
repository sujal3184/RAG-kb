"""Admin API endpoints.

Every route here requires the admin role via `require_admin`. These
endpoints intentionally span all users — they are the operator's view of
the system, not a user's view of their own data.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_admin_service, require_admin
from app.models.document import DocumentStatus
from app.models.user import User
from app.schemas.admin import (
    AdminDocumentListResponse,
    AdminDocumentResponse,
    AdminUserListResponse,
    AdminUserResponse,
    SystemStatsResponse,
    UpdateUserStatusRequest,
)
from app.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    summary="Get system-wide statistics",
)
async def get_system_stats(
    _admin: Annotated[User, Depends(require_admin)],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
) -> SystemStatsResponse:
    """Return counts of users, knowledge bases, and documents by status."""
    stats = await admin_service.get_system_stats()
    return SystemStatsResponse(**stats)


@router.get(
    "/users",
    response_model=AdminUserListResponse,
    summary="List all users",
)
async def list_users(
    _admin: Annotated[User, Depends(require_admin)],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminUserListResponse:
    """List all registered users, newest first."""
    users, total = await admin_service.list_users(limit=limit, offset=offset)
    return AdminUserListResponse(
        items=[AdminUserResponse.model_validate(u) for u in users],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.patch(
    "/users/{user_id}/status",
    response_model=AdminUserResponse,
    summary="Activate or deactivate a user account",
)
async def update_user_status(
    user_id: uuid.UUID,
    body: UpdateUserStatusRequest,
    admin: Annotated[User, Depends(require_admin)],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
) -> AdminUserResponse:
    """Enable or disable a user's ability to log in.

    Deactivated users cannot log in, and their existing access tokens stop
    working (the `get_current_user` dependency checks `is_active`).
    """
    user = await admin_service.set_user_active_status(
        user_id=user_id, is_active=body.is_active, acting_admin_id=admin.id
    )
    return AdminUserResponse.model_validate(user)


@router.post(
    "/users/{user_id}/promote",
    response_model=AdminUserResponse,
    summary="Grant a user admin privileges",
)
async def promote_user_to_admin(
    user_id: uuid.UUID,
    _admin: Annotated[User, Depends(require_admin)],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
) -> AdminUserResponse:
    """Promote an existing user to the admin role."""
    user = await admin_service.promote_to_admin(user_id=user_id)
    return AdminUserResponse.model_validate(user)


@router.get(
    "/documents",
    response_model=AdminDocumentListResponse,
    summary="List documents by processing status across all knowledge bases",
)
async def list_documents_by_status(
    _admin: Annotated[User, Depends(require_admin)],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
    status: Annotated[DocumentStatus, Query(description="Filter by processing status")] = DocumentStatus.FAILED,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminDocumentListResponse:
    """List documents with a given status — defaults to FAILED, the most
    common admin use case (reviewing what went wrong and why)."""
    documents, total = await admin_service.list_documents_by_status(
        status=status, limit=limit, offset=offset
    )
    return AdminDocumentListResponse(
        items=[AdminDocumentResponse.model_validate(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/documents/{document_id}/retry",
    response_model=AdminDocumentResponse,
    summary="Retry processing for a failed document",
)
async def retry_document(
    document_id: uuid.UUID,
    _admin: Annotated[User, Depends(require_admin)],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
) -> AdminDocumentResponse:
    """Reset a document to PENDING and re-enqueue it for processing."""
    document = await admin_service.retry_document_processing(document_id=document_id)
    return AdminDocumentResponse.model_validate(document)