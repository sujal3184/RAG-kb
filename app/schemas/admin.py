"""Request/response models for admin endpoints."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr

from app.models.document import DocumentStatus
from app.models.user import UserRole


class SystemStatsResponse(BaseModel):
    """High-level system statistics."""

    total_users: int
    active_users: int
    verified_users: int
    admin_users: int
    total_knowledge_bases: int
    documents_by_status: dict[str, int]


class AdminUserResponse(BaseModel):
    """User data visible to admins — includes role and status fields
    that regular users don't see about each other."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    is_active: bool
    is_verified: bool
    role: UserRole
    created_at: datetime


class AdminUserListResponse(BaseModel):
    """A page of users."""

    items: list[AdminUserResponse]
    total: int
    limit: int
    offset: int


class UpdateUserStatusRequest(BaseModel):
    """Admin request to activate or deactivate a user account."""

    is_active: bool


class AdminDocumentResponse(BaseModel):
    """Document data for admin oversight — includes the error message
    that explains why processing failed."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    original_filename: str
    file_extension: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class AdminDocumentListResponse(BaseModel):
    """A page of documents."""

    items: list[AdminDocumentResponse]
    total: int
    limit: int
    offset: int


class MessageResponse(BaseModel):
    """Generic acknowledgement response."""

    message: str