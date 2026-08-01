"""Application settings."""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_NAME: str = "Knowledge Base RAG"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)

    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_JSON: bool = False

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "kb_admin"
    POSTGRES_PASSWORD: str = "kb_dev_password"
    POSTGRES_DB: str = "knowledge_base"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    # --- Security / JWT ---------------------------------------------------
    SECRET_KEY: str = Field(
        default="dev-only-secret-change-me",
        description="Used to sign JWTs. MUST be a long random string in production.",
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --- Email verification / password reset -------------------------------
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 24
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 1

     # Base URL of the (future) frontend, used to build links inside emails,
    # e.g. f"{FRONTEND_BASE_URL}/verify-email?token=xyz"
    FRONTEND_BASE_URL: str = "http://localhost:3000"

    # --- File storage -------------------------------------------------------
    # Where uploaded files are stored on disk. Inside Docker this should be
    # a mounted volume path so files survive container restarts.
    LOCAL_STORAGE_PATH: str = "./data/uploads"

    # --- Document upload limits ----------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_UPLOAD_EXTENSIONS: str = "pdf,docx,txt,md,html,csv"

    # --- Chunking -------------------------------------------------------
    DEFAULT_CHUNK_SIZE_TOKENS: int = 500
    DEFAULT_CHUNK_OVERLAP_TOKENS: int = 50
    # The tokenizer encoding used to COUNT tokens for chunk sizing. This is
    # independent of which embedding model we eventually use (Module 9) —
    # it's just a consistent, fast way to measure "how much text is this."
    TOKENIZER_ENCODING: str = "cl100k_base"

    # --- Embeddings -------------------------------------------------------
    PRIMARY_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    FALLBACK_EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_BATCH_SIZE: int = 32
    # Where sentence-transformers caches downloaded model weights.
    EMBEDDING_MODEL_CACHE_DIR: str = "./data/models"

    # --- Qdrant -----------------------------------------------------------
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_PREFIX: str = "kb_"
    # How many results to fetch per search call by default (can be
    # overridden per-call by future retrieval logic in Module 11).
    QDRANT_DEFAULT_TOP_K: int = 10


     # --- Hybrid Retrieval ---------------------------------------------------
    HYBRID_DENSE_WEIGHT_RRF_K: int = 60
    # RRF's "k" constant — a standard damping factor (60 is the widely-used
    # default from the original RRF paper) controlling how much lower-ranked
    # results still contribute to the fused score.
    HYBRID_TOP_K_PER_METHOD: int = 20
    # How many results each individual method (dense/BM25) contributes
    # BEFORE fusion — kept higher than the final desired result count so
    # fusion has enough candidates to work with.


    # --- Reranking ----------------------------------------------------------
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MODEL_CACHE_DIR: str = "./data/models"
    RERANKER_BATCH_SIZE: int = 16

    # --- Context Compression -------------------------------------------------
    DEDUPLICATION_SIMILARITY_THRESHOLD: float = 0.92
    # Chunks with cosine similarity above this threshold are considered
    # near-duplicates; only the highest-ranked one is kept.
    MAX_CONTEXT_TOKENS: int = 4000
    # Hard cap on total tokens across all chunks passed to the LLM prompt.

     # --- Prompt Builder -------------------------------------------------------
    MAX_CONVERSATION_HISTORY_MESSAGES: int = 10
    # How many prior turns (user+assistant pairs) to include for context,

     # --- Groq LLM -----------------------------------------------------------
    GROQ_API_KEY: str = "change-me"
    PRIMARY_LLM_MODEL: str = "llama-3.3-70b-versatile"
    FALLBACK_LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_OUTPUT_TOKENS: int = 1024
    LLM_REQUEST_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 2


    # --- Redis / Celery -------------------------------------------------------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CELERY_TASK_MAX_RETRIES: int = 3
    CELERY_TASK_RETRY_BACKOFF_SECONDS: int = 10

    @computed_field  # type: ignore[prop-decorator]
    @property
    def redis_url(self) -> str:
        """Build the Redis connection URL, used as both Celery's broker
        and result backend."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def max_upload_size_bytes(self) -> int:
        """Convert the configured MB limit into bytes for easy comparison."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def allowed_upload_extensions_set(self) -> set[str]:
        """Parse the comma-separated extension list into a lowercase set."""
        return {ext.strip().lower() for ext in self.ALLOWED_UPLOAD_EXTENSIONS.split(",")}
    

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    return Settings()