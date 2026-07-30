"""Shared FastAPI dependencies used across API routes."""

import uuid
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.core.security import TokenType, decode_token
from app.db.dependencies import get_db_session
from app.models.user import User
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.repositories.verification_token_repository import VerificationTokenRepository
from app.services.auth_service import AuthService
from app.services.email.base import EmailSender
from app.services.email.console_email_sender import ConsoleEmailSender

from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.knowledge_base_service import KnowledgeBaseService

from app.repositories.document_repository import DocumentRepository
from app.services.document_service import DocumentService
from app.storage.base import FileStorage
from app.storage.local_storage import LocalFileStorage

from functools import lru_cache

from app.embeddings.bge_m3_provider import BgeM3Provider
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.nomic_provider import NomicProvider

from app.retrieval.base import VectorStore
from app.retrieval.qdrant_store import QdrantVectorStore


oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{get_settings().API_V1_PREFIX}/auth/login")


def get_user_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserRepository:
    return UserRepository(db)


def get_refresh_token_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RefreshTokenRepository:
    return RefreshTokenRepository(db)


def get_verification_token_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> VerificationTokenRepository:
    return VerificationTokenRepository(db)


def get_email_sender() -> EmailSender:
    """Provide the email-sending implementation.

    This is the ONE place to change when a real email provider is added
    later — swap `ConsoleEmailSender()` for e.g. `SesEmailSender(settings)`,
    and nothing else in the app needs to change.
    """
    return ConsoleEmailSender()


def get_auth_service(
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    refresh_token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
    verification_token_repo: Annotated[
        VerificationTokenRepository, Depends(get_verification_token_repository)
    ],
    email_sender: Annotated[EmailSender, Depends(get_email_sender)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AuthService:
    return AuthService(
        user_repo, refresh_token_repo, verification_token_repo, email_sender, settings
    )


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    payload = decode_token(token, expected_type=TokenType.ACCESS, settings=settings)

    user_id = payload.get("sub")
    if user_id is None:
        raise AuthenticationError("Token missing subject claim")

    user = await user_repo.get_by_id(uuid.UUID(user_id))
    if user is None or not user.is_active:
        raise AuthenticationError("User no longer available")

    return user




def get_knowledge_base_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> KnowledgeBaseRepository:
    """Provide a KnowledgeBaseRepository bound to the current request's DB session."""
    return KnowledgeBaseRepository(db)


def get_knowledge_base_service(
    kb_repo: Annotated[KnowledgeBaseRepository, Depends(get_knowledge_base_repository)],
) -> KnowledgeBaseService:
    """Provide a KnowledgeBaseService with its repository wired up."""
    return KnowledgeBaseService(kb_repo)



def get_file_storage(
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileStorage:
    """Provide the file storage implementation.

    This is the ONE place to change when cloud storage is added later —
    swap `LocalFileStorage(...)` for e.g. `S3FileStorage(settings)`, and
    nothing else in the app needs to change.
    """
    return LocalFileStorage(base_path=settings.LOCAL_STORAGE_PATH)


def get_document_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentRepository:
    """Provide a DocumentRepository bound to the current request's DB session."""
    return DocumentRepository(db)


def get_document_service(
    document_repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    kb_service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    file_storage: Annotated[FileStorage, Depends(get_file_storage)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentService:
    """Provide a DocumentService with all its dependencies wired up."""
    return DocumentService(
        document_repo,
        kb_service,
        file_storage,
        max_upload_size_bytes=settings.max_upload_size_bytes,
        allowed_extensions=settings.allowed_upload_extensions_set,
    )



@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Provide a singleton EmbeddingService.

    Cached with lru_cache (not per-request) because embedding models are
    expensive to load — we want exactly ONE instance of each model loaded
    per running process, shared across all requests, not one per request.
    """
    settings = get_settings()
    primary = BgeM3Provider(
        settings.PRIMARY_EMBEDDING_MODEL,
        cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    fallback = NomicProvider(
        settings.FALLBACK_EMBEDDING_MODEL,
        cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    return EmbeddingService(primary, fallback)



@lru_cache
def get_vector_store() -> VectorStore:
    """Provide a singleton VectorStore.

    Cached with lru_cache — the Qdrant client manages its own connection
    pooling internally, so we want one shared client per process, not one
    per request (same reasoning as get_embedding_service in Module 9).
    """
    settings = get_settings()
    return QdrantVectorStore(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        collection_prefix=settings.QDRANT_COLLECTION_PREFIX,
    )