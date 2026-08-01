"""Shared FastAPI dependencies used across API routes."""

import uuid
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer , HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
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

from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import BgeRerankerProvider, RerankerProvider
from app.retrieval.reranking_service import RerankingService

from app.chunking.base import TokenCounter
from app.retrieval.context_compressor import ContextCompressor
from app.retrieval.deduplication import Deduplicator

from app.llm.prompt_builder import PromptBuilder

from app.llm.groq_provider import GroqProvider
from app.llm.llm_service import LLMService

from app.repositories.chunk_repository import ChunkRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranking_service import RerankingService
from app.services.conversation_service import ConversationService

from redis.asyncio import Redis

from app.core.cache import CacheService


# oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{get_settings().API_V1_PREFIX}/auth/login")
http_bearer_scheme = HTTPBearer()


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
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(http_bearer_scheme)],
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    token = credentials.credentials
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


@lru_cache
def get_redis_client() -> Redis:
    """Provide a singleton async Redis client, shared across the app.

    Cached with lru_cache — Redis clients manage their own connection
    pooling internally, so one shared instance per process is correct.
    """
    settings = get_settings()
    return Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)


def get_cache_service(
    redis_client: Annotated[Redis, Depends(get_redis_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CacheService:
    """Provide a CacheService wired to the shared Redis client."""
    return CacheService(redis_client, enabled=settings.CACHE_ENABLED)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Provide a singleton EmbeddingService, now with caching wired in."""
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
    redis_client = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)
    cache = CacheService(redis_client, enabled=settings.CACHE_ENABLED)
    return EmbeddingService(
        primary, fallback, cache=cache, cache_ttl_seconds=settings.EMBEDDING_CACHE_TTL_SECONDS
    )



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



@lru_cache
def get_bm25_store() -> BM25Store:
    """Provide a singleton BM25Store.

    BM25Store itself holds no per-request state (indexes are built
    on-the-fly per search call), so a single shared instance is safe and
    avoids pointless re-instantiation.
    """
    return BM25Store()


def get_hybrid_retriever(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    bm25_store: Annotated[BM25Store, Depends(get_bm25_store)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HybridRetriever:
    """Provide a HybridRetriever with all its dependencies wired up."""
    return HybridRetriever(
        vector_store,
        bm25_store,
        embedding_service,
        top_k_per_method=settings.HYBRID_TOP_K_PER_METHOD,
        rrf_k=settings.HYBRID_DENSE_WEIGHT_RRF_K,
    )

@lru_cache
def get_reranker_provider() -> RerankerProvider:
    """Provide a singleton RerankerProvider.

    Cached with lru_cache — same reasoning as embedding models (Module 9):
    the underlying model is expensive to load and safe to share across
    requests.
    """
    settings = get_settings()
    return BgeRerankerProvider(
        settings.RERANKER_MODEL,
        cache_dir=settings.RERANKER_MODEL_CACHE_DIR,
        batch_size=settings.RERANKER_BATCH_SIZE,
    )


def get_reranking_service(
    reranker: Annotated[RerankerProvider, Depends(get_reranker_provider)],
) -> RerankingService:
    """Provide a RerankingService with its reranker wired up."""
    return RerankingService(reranker)


@lru_cache
def get_token_counter() -> TokenCounter:
    """Provide a singleton TokenCounter, shared across chunking (Module 8)
    and context compression (this module) so 'token' always means the
    same thing throughout the app."""
    settings = get_settings()
    return TokenCounter(settings.TOKENIZER_ENCODING)


def get_deduplicator(
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> Deduplicator:
    """Provide a Deduplicator with its embedding service wired up."""
    return Deduplicator(embedding_service)


def get_context_compressor(
    deduplicator: Annotated[Deduplicator, Depends(get_deduplicator)],
    token_counter: Annotated[TokenCounter, Depends(get_token_counter)],
) -> ContextCompressor:
    """Provide a ContextCompressor with its dependencies wired up."""
    return ContextCompressor(deduplicator, token_counter)


def get_prompt_builder(
    token_counter: Annotated[TokenCounter, Depends(get_token_counter)],
) -> PromptBuilder:
    """Provide a PromptBuilder with its token counter wired up."""
    return PromptBuilder(token_counter)


@lru_cache
def get_llm_service() -> LLMService:
    """Provide a singleton LLMService.

    Cached with lru_cache — same reasoning as get_embedding_service
    (Module 9): the underlying Groq client connections are cheap to
    reuse across requests, and we want consistent configuration process-wide.
    """
    settings = get_settings()
    primary = GroqProvider(
        api_key=settings.GROQ_API_KEY,
        model=settings.PRIMARY_LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        timeout_seconds=settings.LLM_REQUEST_TIMEOUT_SECONDS,
    )
    fallback = GroqProvider(
        api_key=settings.GROQ_API_KEY,
        model=settings.FALLBACK_LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        timeout_seconds=settings.LLM_REQUEST_TIMEOUT_SECONDS,
    )
    return LLMService(primary, fallback, max_retries=settings.LLM_MAX_RETRIES)



def get_conversation_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> ConversationRepository:
    return ConversationRepository(db)


def get_message_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> MessageRepository:
    return MessageRepository(db)


def get_chunk_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChunkRepository:
    return ChunkRepository(db)


def get_hybrid_retriever_service(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    bm25_store: Annotated[BM25Store, Depends(get_bm25_store)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HybridRetriever:
    """Provide a HybridRetriever — separate name from Module 11's original
    get_hybrid_retriever to avoid a naming collision now that this module
    also needs one wired identically; both simply build the same class."""
    return HybridRetriever(
        vector_store,
        bm25_store,
        embedding_service,
        top_k_per_method=settings.HYBRID_TOP_K_PER_METHOD,
        rrf_k=settings.HYBRID_DENSE_WEIGHT_RRF_K,
    )


def get_reranking_service_instance(
    reranker: Annotated[RerankerProvider, Depends(get_reranker_provider)],
) -> RerankingService:
    return RerankingService(reranker)


def get_conversation_service(
    conversation_repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    message_repo: Annotated[MessageRepository, Depends(get_message_repository)],
    chunk_repo: Annotated[ChunkRepository, Depends(get_chunk_repository)],
    document_repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    kb_service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    hybrid_retriever: Annotated[HybridRetriever, Depends(get_hybrid_retriever_service)],
    reranking_service: Annotated[RerankingService, Depends(get_reranking_service_instance)],
    context_compressor: Annotated[ContextCompressor, Depends(get_context_compressor)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    cache: Annotated[CacheService, Depends(get_cache_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversationService:
    """Provide a fully-wired ConversationService, now with response caching."""
    return ConversationService(
        conversation_repo,
        message_repo,
        chunk_repo,
        document_repo,
        kb_service,
        hybrid_retriever,
        reranking_service,
        context_compressor,
        prompt_builder,
        llm_service,
        max_history_messages=settings.MAX_CONVERSATION_HISTORY_MESSAGES,
        retrieval_top_k=settings.HYBRID_TOP_K_PER_METHOD,
        rerank_top_k=10,
        similarity_threshold=settings.DEDUPLICATION_SIMILARITY_THRESHOLD,
        max_context_tokens=settings.MAX_CONTEXT_TOKENS,
        cache=cache,
        response_cache_ttl_seconds=settings.RAG_RESPONSE_CACHE_TTL_SECONDS,
    )

def get_document_service(
    document_repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    kb_service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    file_storage: Annotated[FileStorage, Depends(get_file_storage)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentService:
    """Provide a DocumentService, now wired to invalidate the response
    cache whenever documents change."""
    return DocumentService(
        document_repo,
        kb_service,
        file_storage,
        max_upload_size_bytes=settings.max_upload_size_bytes,
        allowed_extensions=settings.allowed_upload_extensions_set,
        conversation_service=conversation_service,
    )


