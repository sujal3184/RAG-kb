"""Test data factories.

Building a User -> KnowledgeBase -> Document chain by hand in every test
is repetitive and obscures what each test is actually about. These
helpers create valid entities with sensible defaults, overridable per
test.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User, UserRole


async def create_user(
    session: AsyncSession,
    *,
    email: str | None = None,
    password: str = "correcthorse123",
    role: UserRole = UserRole.USER,
    is_active: bool = True,
    is_verified: bool = True,
) -> User:
    """Create a user with sensible defaults.

    Args:
        session: the test's database session.
        email: defaults to a unique generated address.
        password: plain-text password, hashed before storage.
        role: user or admin.
        is_active: whether the account is enabled.
        is_verified: whether the email is verified.
    """
    user = User(
        email=email or f"test-{uuid.uuid4()}@example.com",
        hashed_password=hash_password(password),
        role=role,
        is_active=is_active,
        is_verified=is_verified,
    )
    session.add(user)
    await session.flush()
    return user


async def create_knowledge_base(
    session: AsyncSession, *, owner: User, name: str = "Test Knowledge Base"
) -> KnowledgeBase:
    """Create a knowledge base owned by the given user."""
    kb = KnowledgeBase(owner_id=owner.id, name=name)
    session.add(kb)
    await session.flush()
    return kb


async def create_document(
    session: AsyncSession,
    *,
    knowledge_base: KnowledgeBase,
    filename: str = "test.txt",
    status: DocumentStatus = DocumentStatus.READY,
    error_message: str | None = None,
) -> Document:
    """Create a document within a knowledge base."""
    document = Document(
        knowledge_base_id=knowledge_base.id,
        original_filename=filename,
        file_extension=filename.rsplit(".", 1)[-1],
        content_type="text/plain",
        size_bytes=100,
        storage_ref=f"test://{uuid.uuid4()}",
        status=status,
        error_message=error_message,
    )
    session.add(document)
    await session.flush()
    return document


async def login(client, email: str, password: str = "correcthorse123") -> str:
    """Log in via the API and return the access token."""
    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": password}
    )
    return response.json()["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    """Build an Authorization header from a token."""
    return {"Authorization": f"Bearer {token}"}