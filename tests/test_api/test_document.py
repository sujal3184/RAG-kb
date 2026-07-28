"""Integration tests for document upload endpoints."""

import io

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.docker


async def _register_login_and_create_kb(client: AsyncClient, email: str) -> tuple[str, str]:
    """Helper: register, log in, create a knowledge base.

    Returns:
        A tuple of (access_token, knowledge_base_id).
    """
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correcthorse123"}
    )
    login_response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "correcthorse123"}
    )
    token = login_response.json()["access_token"]

    kb_response = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": "Test KB"},
        headers={"Authorization": f"Bearer {token}"},
    )
    kb_id = kb_response.json()["id"]
    return token, kb_id


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_upload_document_succeeds(client: AsyncClient) -> None:
    """Uploading a valid file should create a document with status pending."""
    token, kb_id = await _register_login_and_create_kb(client, "doc_user1@example.com")

    file_content = b"This is a test document about RAG systems."
    response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        headers=_auth_headers(token),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "test.txt"
    assert body["status"] == "pending"
    assert body["size_bytes"] == len(file_content)


@pytest.mark.asyncio
async def test_upload_rejects_disallowed_extension(client: AsyncClient) -> None:
    """Uploading an unsupported file type should fail with a clear error."""
    token, kb_id = await _register_login_and_create_kb(client, "doc_user2@example.com")

    response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("virus.exe", io.BytesIO(b"fake exe content"), "application/octet-stream")},
        headers=_auth_headers(token),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_rejects_oversized_file(client: AsyncClient) -> None:
    """Uploading a file larger than the configured limit should fail."""
    token, kb_id = await _register_login_and_create_kb(client, "doc_user3@example.com")

    # Settings default MAX_UPLOAD_SIZE_MB=50 — simulate exceeding it cheaply
    # by monkeypatching isn't needed; instead we rely on a smaller test
    # override. For simplicity here we just assert the endpoint enforces
    # SOME limit by checking a clearly oversized file is rejected.
    oversized_content = b"x" * (51 * 1024 * 1024)  # 51MB
    response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("big.txt", io.BytesIO(oversized_content), "text/plain")},
        headers=_auth_headers(token),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_requires_ownership_of_knowledge_base(client: AsyncClient) -> None:
    """A user cannot upload into a knowledge base they don't own."""
    _, kb_id = await _register_login_and_create_kb(client, "doc_owner@example.com")
    other_token, _ = await _register_login_and_create_kb(client, "doc_intruder@example.com")

    response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        headers=_auth_headers(other_token),
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_documents(client: AsyncClient) -> None:
    """Listing documents should return uploaded files for that knowledge base."""
    token, kb_id = await _register_login_and_create_kb(client, "doc_user4@example.com")
    await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("a.txt", io.BytesIO(b"content a"), "text/plain")},
        headers=_auth_headers(token),
    )
    await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("b.txt", io.BytesIO(b"content b"), "text/plain")},
        headers=_auth_headers(token),
    )

    response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}/documents", headers=_auth_headers(token)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    filenames = {item["original_filename"] for item in body["items"]}
    assert filenames == {"a.txt", "b.txt"}


@pytest.mark.asyncio
async def test_get_single_document(client: AsyncClient) -> None:
    """Fetching a specific document by id should return its metadata."""
    token, kb_id = await _register_login_and_create_kb(client, "doc_user5@example.com")
    upload_response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("fetchme.txt", io.BytesIO(b"content"), "text/plain")},
        headers=_auth_headers(token),
    )
    document_id = upload_response.json()["id"]

    response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}/documents/{document_id}",
        headers=_auth_headers(token),
    )

    assert response.status_code == 200
    assert response.json()["original_filename"] == "fetchme.txt"


@pytest.mark.asyncio
async def test_delete_document(client: AsyncClient) -> None:
    """Deleting a document should remove it, and it should 404 afterwards."""
    token, kb_id = await _register_login_and_create_kb(client, "doc_user6@example.com")
    upload_response = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("deleteme.txt", io.BytesIO(b"content"), "text/plain")},
        headers=_auth_headers(token),
    )
    document_id = upload_response.json()["id"]

    delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{kb_id}/documents/{document_id}",
        headers=_auth_headers(token),
    )
    assert delete_response.status_code == 204

    get_response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}/documents/{document_id}",
        headers=_auth_headers(token),
    )
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_cannot_access_documents_in_another_users_kb(client: AsyncClient) -> None:
    """A user cannot list or fetch documents inside someone else's knowledge base."""
    owner_token, kb_id = await _register_login_and_create_kb(client, "doc_owner2@example.com")
    await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("secret.txt", io.BytesIO(b"secret content"), "text/plain")},
        headers=_auth_headers(owner_token),
    )

    intruder_token, _ = await _register_login_and_create_kb(client, "doc_intruder2@example.com")
    response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}/documents", headers=_auth_headers(intruder_token)
    )

    assert response.status_code == 404