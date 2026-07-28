"""Integration tests for Knowledge Base endpoints.

Covers CRUD operations and, critically, the ownership isolation rule:
User A must never be able to see, edit, or delete User B's knowledge bases.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.docker


async def _register_and_login(client: AsyncClient, email: str) -> str:
    """Helper: register a fresh user and return their access token."""
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correcthorse123"}
    )
    login_response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "correcthorse123"}
    )
    return login_response.json()["access_token"]


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_create_knowledge_base(client: AsyncClient) -> None:
    """A logged-in user can create a knowledge base."""
    token = await _register_and_login(client, "kb_owner1@example.com")

    response = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": "My First KB", "description": "Testing things"},
        headers=_auth_headers(token),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "My First KB"
    assert body["description"] == "Testing things"


@pytest.mark.asyncio
async def test_create_knowledge_base_requires_auth(client: AsyncClient) -> None:
    """Creating a knowledge base without a token should be rejected."""
    response = await client.post("/api/v1/knowledge-bases", json={"name": "No Auth KB"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_returns_only_own_knowledge_bases(client: AsyncClient) -> None:
    """Listing knowledge bases should only return the current user's own KBs."""
    token_a = await _register_and_login(client, "kb_owner2@example.com")
    token_b = await _register_and_login(client, "kb_owner3@example.com")

    await client.post(
        "/api/v1/knowledge-bases", json={"name": "Owner A's KB"}, headers=_auth_headers(token_a)
    )
    await client.post(
        "/api/v1/knowledge-bases", json={"name": "Owner B's KB"}, headers=_auth_headers(token_b)
    )

    response = await client.get("/api/v1/knowledge-bases", headers=_auth_headers(token_a))

    assert response.status_code == 200
    names = [item["name"] for item in response.json()["items"]]
    assert "Owner A's KB" in names
    assert "Owner B's KB" not in names


@pytest.mark.asyncio
async def test_get_single_knowledge_base(client: AsyncClient) -> None:
    """Fetching a knowledge base by id should return it, if owned."""
    token = await _register_and_login(client, "kb_owner4@example.com")
    create_response = await client.post(
        "/api/v1/knowledge-bases", json={"name": "Fetch Me"}, headers=_auth_headers(token)
    )
    kb_id = create_response.json()["id"]

    response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}", headers=_auth_headers(token)
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Fetch Me"


@pytest.mark.asyncio
async def test_cannot_get_another_users_knowledge_base(client: AsyncClient) -> None:
    """A user must not be able to fetch someone else's knowledge base by id."""
    token_a = await _register_and_login(client, "kb_owner5@example.com")
    token_b = await _register_and_login(client, "kb_owner6@example.com")

    create_response = await client.post(
        "/api/v1/knowledge-bases", json={"name": "Private KB"}, headers=_auth_headers(token_a)
    )
    kb_id = create_response.json()["id"]

    response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}", headers=_auth_headers(token_b)
    )

    assert response.status_code == 404  # not 403 — see design notes


@pytest.mark.asyncio
async def test_update_knowledge_base(client: AsyncClient) -> None:
    """A user can partially update their own knowledge base."""
    token = await _register_and_login(client, "kb_owner7@example.com")
    create_response = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": "Old Name", "description": "Old description"},
        headers=_auth_headers(token),
    )
    kb_id = create_response.json()["id"]

    response = await client.patch(
        f"/api/v1/knowledge-bases/{kb_id}",
        json={"name": "New Name"},
        headers=_auth_headers(token),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "New Name"
    assert body["description"] == "Old description"  # untouched


@pytest.mark.asyncio
async def test_cannot_update_another_users_knowledge_base(client: AsyncClient) -> None:
    """A user must not be able to update someone else's knowledge base."""
    token_a = await _register_and_login(client, "kb_owner8@example.com")
    token_b = await _register_and_login(client, "kb_owner9@example.com")

    create_response = await client.post(
        "/api/v1/knowledge-bases", json={"name": "Untouchable"}, headers=_auth_headers(token_a)
    )
    kb_id = create_response.json()["id"]

    response = await client.patch(
        f"/api/v1/knowledge-bases/{kb_id}",
        json={"name": "Hacked Name"},
        headers=_auth_headers(token_b),
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_knowledge_base(client: AsyncClient) -> None:
    """A user can delete their own knowledge base, and it's then gone."""
    token = await _register_and_login(client, "kb_owner10@example.com")
    create_response = await client.post(
        "/api/v1/knowledge-bases", json={"name": "Delete Me"}, headers=_auth_headers(token)
    )
    kb_id = create_response.json()["id"]

    delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{kb_id}", headers=_auth_headers(token)
    )
    assert delete_response.status_code == 204

    get_response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}", headers=_auth_headers(token)
    )
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_cannot_delete_another_users_knowledge_base(client: AsyncClient) -> None:
    """A user must not be able to delete someone else's knowledge base."""
    token_a = await _register_and_login(client, "kb_owner11@example.com")
    token_b = await _register_and_login(client, "kb_owner12@example.com")

    create_response = await client.post(
        "/api/v1/knowledge-bases", json={"name": "Protected"}, headers=_auth_headers(token_a)
    )
    kb_id = create_response.json()["id"]

    response = await client.delete(
        f"/api/v1/knowledge-bases/{kb_id}", headers=_auth_headers(token_b)
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_pagination_metadata(client: AsyncClient) -> None:
    """The list endpoint should return correct pagination metadata."""
    token = await _register_and_login(client, "kb_owner13@example.com")
    for i in range(3):
        await client.post(
            "/api/v1/knowledge-bases", json={"name": f"KB {i}"}, headers=_auth_headers(token)
        )

    response = await client.get(
        "/api/v1/knowledge-bases?limit=2&offset=0", headers=_auth_headers(token)
    )

    body = response.json()
    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0