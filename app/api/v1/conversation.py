"""Conversation API endpoints.

Nested under Knowledge Bases, same pattern as documents (Module 6) —
every operation verifies ownership via ConversationService, which
delegates to KnowledgeBaseService.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import get_conversation_service, get_current_user
from app.models.user import User
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services.conversation_service import ConversationService

router = APIRouter(
    prefix="/knowledge-bases/{knowledge_base_id}/conversations", tags=["conversations"]
)


@router.post(
    "", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED,
    summary="Start a new conversation",
)
async def create_conversation(
    knowledge_base_id: uuid.UUID,
    body: ConversationCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ConversationResponse:
    conversation = await conversation_service.create_conversation(
        knowledge_base_id=knowledge_base_id, owner_id=current_user.id, title=body.title
    )
    return ConversationResponse.model_validate(conversation)


@router.get("", response_model=ConversationListResponse, summary="List conversations")
async def list_conversations(
    knowledge_base_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationListResponse:
    items, total = await conversation_service.list_conversations(
        knowledge_base_id=knowledge_base_id, owner_id=current_user.id, limit=limit, offset=offset
    )
    return ConversationListResponse(
        items=[ConversationResponse.model_validate(item) for item in items],
        total=total, limit=limit, offset=offset,
    )


@router.get(
    "/{conversation_id}/messages", response_model=list[MessageResponse],
    summary="Get full message history for a conversation",
)
async def get_conversation_messages(
    knowledge_base_id: uuid.UUID,
    conversation_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> list[MessageResponse]:
    _, messages = await conversation_service.get_conversation_with_messages(
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        owner_id=current_user.id,
    )
    return [MessageResponse.model_validate(m) for m in messages]


@router.post(
    "/{conversation_id}/messages", response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send a message and get a RAG-generated reply",
)
async def send_message(
    knowledge_base_id: uuid.UUID,
    conversation_id: uuid.UUID,
    body: SendMessageRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> SendMessageResponse:
    """Send a user message and receive the assistant's generated reply.

    This triggers the full retrieval pipeline: hybrid search, reranking,
    context compression, prompt building, and LLM generation.
    """
    user_message, assistant_message, sources = await conversation_service.send_message(
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        owner_id=current_user.id,
        content=body.content,
    )
    return SendMessageResponse(
        user_message=MessageResponse.model_validate(user_message),
        assistant_message=MessageResponse.model_validate(assistant_message),
        sources=sources,
    )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
async def delete_conversation(
    knowledge_base_id: uuid.UUID,
    conversation_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> None:
    """Delete a conversation and its message history."""
    await conversation_service.delete_conversation(
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        owner_id=current_user.id,
    )



from fastapi.responses import StreamingResponse


@router.post(
    "/{conversation_id}/messages/stream",
    summary="Send a message and stream the RAG-generated reply as it's produced",
)
async def send_message_streaming(
    knowledge_base_id: uuid.UUID,
    conversation_id: uuid.UUID,
    body: SendMessageRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> StreamingResponse:
    """Stream the assistant's reply as Server-Sent Events (SSE).

    Each event's data is a raw text chunk. The client should concatenate
    chunks as they arrive to build the full response. A final event with
    data "[DONE]" signals the stream is complete.
    """

    async def event_generator():
        try:
            async for chunk in conversation_service.send_message_streaming(
                conversation_id=conversation_id,
                knowledge_base_id=knowledge_base_id,
                owner_id=current_user.id,
                content=body.content,
            ):
                # SSE format: each event is "data: <payload>\n\n". We
                # escape newlines within a chunk since SSE treats a blank
                # line as the event boundary.
                escaped = chunk.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            # SSE has no separate error channel — encode failures as a
            # special event the frontend can detect and surface.
            yield f"data: [ERROR] {str(exc)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables nginx response buffering, if ever deployed behind one
        },
    )