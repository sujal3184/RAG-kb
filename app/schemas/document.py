"""Request/response models for the Document API."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """Public-facing document metadata (never includes storage_ref — an
    internal implementation detail clients don't need)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    original_filename: str
    file_extension: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """A page of documents, with pagination metadata."""

    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int