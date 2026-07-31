"""Integration tests for conversation endpoints.

Uses real Postgres + Qdrant + embedding/reranker models + Groq API, since
this exercises the FULL RAG pipeline end-to-end. Marked docker AND
external_api — requires Docker running and a valid GROQ_API_KEY.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk

pytestmark = [pytest.mark.docker, pytest.mark.external_api]


async def _register_login_and_create_kb(client: AsyncClient, email: str) -> tuple[str, str]:
    await client.post("/api/v1/auth/register", json={"email": email, "password": "correcthorse123"})
    login_response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "correcthorse123"}
    )
    token = login_response.json()["access_token"]
    kb_response = await client.post(
        "/api/v1/knowledge-bases", json={"name": "Chat Test KB"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return token, kb_response.json()["id"]


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _seed_chunks_and_vectors(db_session: AsyncSession, kb_id: str, texts: list[str]) -> None:
    """Insert a Document row, then Chunks into Postgres AND vectors into
    Qdrant, simulating what Module 17's background pipeline will
    eventually do automatically."""
    import uuid as uuid_module

    from app.api.dependencies import get_embedding_service, get_vector_store
    from app.models.document import Document, DocumentStatus
    from app.retrieval.base import VectorPoint

    kb_uuid = uuid_module.UUID(kb_id)

    # A Chunk's document_id is a real foreign key — we must create the
    # parent Document row first, or Postgres correctly rejects the chunk.
    document = Document(
        knowledge_base_id=kb_uuid,
        original_filename="seed_test_document.txt",
        file_extension="txt",
        content_type="text/plain",
        size_bytes=sum(len(t) for t in texts),
        storage_ref="test://seeded-in-memory",
        status=DocumentStatus.READY,
    )
    db_session.add(document)
    await db_session.flush()  # so document.id is populated before chunks reference it

    embedding_service = get_embedding_service()
    vector_store = get_vector_store()
    embed_result = await embedding_service.embed_texts(texts)

    points = []
    for i, text in enumerate(texts):
        chunk = Chunk(document_id=document.id, knowledge_base_id=kb_uuid, chunk_index=i, text=text)
        db_session.add(chunk)
        points.append(
            VectorPoint(
                chunk_id=str(chunk.id), document_id=document.id, knowledge_base_id=kb_uuid,
                chunk_index=i, text=text, vector=embed_result.vectors[i],
            )
        )
    await db_session.flush()
    await vector_store.upsert(
        knowledge_base_id=kb_uuid, points=points, vector_dimension=embed_result.dimension
    )
    await db_session.commit()


@pytest.mark.asyncio
async def test_create_conversation(client: AsyncClient) -> None:
    token, kb_id = await _register_login_and_create_kb(client, "convo_user1@example.com")

    response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/conversations",
        json={"title": "My first chat"},
        headers=_auth_headers(token),
    )

    assert response.status_code == 201
    assert response.json()["title"] == "My first chat"


@pytest.mark.asyncio
async def test_send_message_generates_rag_reply(client: AsyncClient, db_session) -> None:
    """End-to-end: seed content, create a conversation, send a message,
    and confirm a real, relevant assistant reply comes back with sources."""
    token, kb_id = await _register_login_and_create_kb(client, "convo_user2@example.com")
    await _seed_chunks_and_vectors(
        db_session, kb_id,
        ["Paris is the capital and most populous city of France."],
    )

    conv_response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/conversations", json={},
        headers=_auth_headers(token),
    )
    conversation_id = conv_response.json()["id"]

    response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/conversations/{conversation_id}/messages",
        json={"content": "What is the capital of France?"},
        headers=_auth_headers(token),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["user_message"]["content"] == "What is the capital of France?"
    assert "paris" in body["assistant_message"]["content"].lower()
    assert len(body["sources"]) > 0


@pytest.mark.asyncio
async def test_message_history_is_persisted_and_ordered(client: AsyncClient, db_session) -> None:
    token, kb_id = await _register_login_and_create_kb(client, "convo_user3@example.com")
    await _seed_chunks_and_vectors(db_session, kb_id, ["The sky appears blue due to Rayleigh scattering."])

    conv_response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/conversations", json={}, headers=_auth_headers(token)
    )
    conversation_id = conv_response.json()["id"]

    await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/conversations/{conversation_id}/messages",
        json={"content": "Why is the sky blue?"}, headers=_auth_headers(token),
    )

    history_response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}/conversations/{conversation_id}/messages",
        headers=_auth_headers(token),
    )

    messages = history_response.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_cannot_access_conversation_in_another_users_kb(client: AsyncClient) -> None:
    owner_token, kb_id = await _register_login_and_create_kb(client, "convo_owner@example.com")
    conv_response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/conversations", json={},
        headers=_auth_headers(owner_token),
    )
    conversation_id = conv_response.json()["id"]

    intruder_token, _ = await _register_login_and_create_kb(client, "convo_intruder@example.com")
    response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}/conversations/{conversation_id}/messages",
        headers=_auth_headers(intruder_token),
    )

    assert response.status_code == 404
    