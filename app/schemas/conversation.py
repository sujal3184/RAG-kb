"""Request/response models for the Conversation API."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.message import MessageRole


class ConversationCreateRequest(BaseModel):
    """What the client sends to start a new conversation."""

    title: str | None = Field(default=None, max_length=255)


class ConversationResponse(BaseModel):
    """Public-facing conversation metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """A page of conversations, with pagination metadata."""

    items: list[ConversationResponse]
    total: int
    limit: int
    offset: int


class MessageResponse(BaseModel):
    """Public-facing message data."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    created_at: datetime


class SendMessageRequest(BaseModel):
    """What the client sends to ask a new question in a conversation."""

    content: str = Field(min_length=1, max_length=8000)


class SendMessageResponse(BaseModel):
    """The result of sending a message — the user's message plus the
    assistant's generated reply, along with citation sources."""

    user_message: MessageResponse
    assistant_message: MessageResponse
    sources: list[str] = Field(
        default_factory=list, description="Filenames of documents cited in the answer"
    )