"""Request/response models for the Knowledge Base API."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseCreateRequest(BaseModel):
    """What the client sends to create a new knowledge base."""

    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)


class KnowledgeBaseUpdateRequest(BaseModel):
    """What the client sends to update a knowledge base.

    Both fields are optional — the client only sends what it wants to
    change (a "partial update" / PATCH-style request).
    """

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)


class KnowledgeBaseResponse(BaseModel):
    """Public-facing knowledge base data."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    owner_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseListResponse(BaseModel):
    """A page of knowledge bases, with pagination metadata."""

    items: list[KnowledgeBaseResponse]
    total: int
    limit: int
    offset: int