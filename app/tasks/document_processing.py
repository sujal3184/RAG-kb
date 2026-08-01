"""Celery task: process an uploaded document into searchable chunks.

Runs the full pipeline built across Modules 7-10:
    FileStorage.read() -> Loader.load() -> Chunker.chunk()
    -> EmbeddingService.embed_texts() -> save Chunk rows (Postgres)
    -> VectorStore.upsert() (Qdrant)

This is a SYNC Celery task (Celery's execution model) that runs an async
event loop internally via asyncio.run() — the standard pattern for using
async application code from Celery tasks.
"""

import asyncio
import logging
import uuid

from celery import Task

from app.chunking.factory import ChunkingFactory, ChunkingStrategyType
from app.config.settings import get_settings
from app.core.exceptions import AppException
from app.db.session import async_session_factory
from app.embeddings.bge_m3_provider import BgeM3Provider
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.nomic_provider import NomicProvider
from app.loaders.exceptions import LoaderError
from app.loaders.factory import LoaderFactory
from app.models.chunk import Chunk
from app.models.document import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.retrieval.base import VectorPoint
from app.retrieval.exceptions import VectorStoreError
from app.retrieval.qdrant_store import QdrantVectorStore
from app.storage.local_storage import LocalFileStorage
from app.workers.celery_app import celery_app
from app.observability.metrics import documents_processed_total


logger = logging.getLogger(__name__)

# Errors that mean "this specific document can never be processed" —
# retrying would just fail again identically. These are NOT retried.
_PERMANENT_FAILURE_TYPES = (LoaderError,)

# Errors that likely represent a TRANSIENT infrastructure hiccup
# (connection issues, temporary unavailability) — worth retrying.
_RETRYABLE_FAILURE_TYPES = (VectorStoreError, ConnectionError, TimeoutError)


class DocumentProcessingTask(Task):
    """Base Celery task class providing shared, lazily-constructed
    dependencies (models, storage, vector store) reused across task runs
    within the same worker process, avoiding reloading embedding models
    on every single document."""

    _embedding_service: EmbeddingService | None = None
    _vector_store: QdrantVectorStore | None = None
    _file_storage: LocalFileStorage | None = None
    _chunking_factory: ChunkingFactory | None = None

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
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
            self._embedding_service = EmbeddingService(primary, fallback)
        return self._embedding_service

    @property
    def vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            settings = get_settings()
            self._vector_store = QdrantVectorStore(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                collection_prefix=settings.QDRANT_COLLECTION_PREFIX,
            )
        return self._vector_store

    @property
    def file_storage(self) -> LocalFileStorage:
        if self._file_storage is None:
            settings = get_settings()
            self._file_storage = LocalFileStorage(base_path=settings.LOCAL_STORAGE_PATH)
        return self._file_storage

    @property
    def chunking_factory(self) -> ChunkingFactory:
        if self._chunking_factory is None:
            self._chunking_factory = ChunkingFactory(get_settings())
        return self._chunking_factory


@celery_app.task(
    base=DocumentProcessingTask,
    bind=True,
    name="app.tasks.document_processing.process_document",
    max_retries=None,  # actual retry limit is read from settings at runtime, see below
)
def process_document(self: DocumentProcessingTask, document_id: str) -> None:
    """Celery task entrypoint — process a single uploaded document.

    Args:
        document_id: string UUID of the Document to process (Celery task
            arguments must be JSON-serializable, so we pass a string
            rather than a uuid.UUID object).
    """
    try:
        asyncio.run(_process_document_async(self, uuid.UUID(document_id)))
    finally:
        # Each asyncio.run() call creates a NEW event loop. The shared
        # engine's connection pool must not hold onto connections tied to
        # a loop that's about to be destroyed — otherwise the NEXT task
        # run in this same worker process crashes trying to reuse a
        # connection bound to a dead event loop. Disposing forces fresh
        # connections on the next task's own event loop.
        from app.db.session import engine

        asyncio.run(engine.dispose())


async def _process_document_async(task: DocumentProcessingTask, document_id: uuid.UUID) -> None:
    """The actual async pipeline logic, run inside the sync Celery task
    via asyncio.run()."""
    settings = get_settings()

    async with async_session_factory() as session:
        document_repo = DocumentRepository(session)

        document = await document_repo.get_by_id(document_id)
        if document is None:
            logger.warning("Document not found, skipping", extra={"document_id": str(document_id)})
            return

        await document_repo.update_status(document_id, DocumentStatus.PROCESSING)
        await session.commit()

        try:
            content = await task.file_storage.read(storage_ref=document.storage_ref)

            loader = LoaderFactory.get_loader(document.file_extension)
            loaded = loader.load(content)

            chunker = task.chunking_factory.get_strategy(ChunkingStrategyType.RECURSIVE)
            text_chunks = chunker.chunk(loaded.text)

            if not text_chunks:
                await document_repo.update_status(
                    document_id, DocumentStatus.FAILED,
                    error_message="No extractable text content found in this document.",
                )
                await session.commit()
                return

            embed_result = await task.embedding_service.embed_texts(
                [c.text for c in text_chunks]
            )

            chunk_rows: list[Chunk] = []
            vector_points: list[VectorPoint] = []
            for i, (text_chunk, vector) in enumerate(zip(text_chunks, embed_result.vectors, strict=True)):
                chunk_row = Chunk(
                    document_id=document.id,
                    knowledge_base_id=document.knowledge_base_id,
                    chunk_index=i,
                    text=text_chunk.text,
                )
                session.add(chunk_row)
                chunk_rows.append(chunk_row)

            await session.flush()  # populate chunk_row.id for each chunk before building points

            for chunk_row, vector in zip(chunk_rows, embed_result.vectors, strict=True):
                vector_points.append(
                    VectorPoint(
                        chunk_id=str(chunk_row.id),
                        document_id=document.id,
                        knowledge_base_id=document.knowledge_base_id,
                        chunk_index=chunk_row.chunk_index,
                        text=chunk_row.text,
                        vector=vector,
                    )
                )

            await task.vector_store.upsert(
                knowledge_base_id=document.knowledge_base_id,
                points=vector_points,
                vector_dimension=embed_result.dimension,
            )

            await document_repo.update_status(document_id, DocumentStatus.READY)
            await session.commit()
            documents_processed_total.labels(status="ready").inc()

            logger.info(
                "Document processed successfully",
                extra={"document_id": str(document_id), "chunk_count": len(chunk_rows)},
            )

        except _PERMANENT_FAILURE_TYPES as exc:
            # This document can never succeed (e.g. corrupted/unsupported
            # content) — record the failure, do NOT retry.
            await session.rollback()
            await document_repo.update_status(
                document_id, DocumentStatus.FAILED, error_message=str(exc)
            )
            await session.commit()
            documents_processed_total.labels(status="failed").inc()
            logger.warning(
                "Document processing failed permanently",
                extra={"document_id": str(document_id), "error": str(exc)},
            )

        except _RETRYABLE_FAILURE_TYPES as exc:
            # Likely transient infrastructure issue — let Celery retry
            # this task with backoff, up to the configured max attempts.
            await session.rollback()
            logger.warning(
                "Document processing failed with a retryable error",
                extra={
                    "document_id": str(document_id),
                    "error": str(exc),
                    "retry_count": task.request.retries,
                },
            )
            if task.request.retries >= settings.CELERY_TASK_MAX_RETRIES:
                await document_repo.update_status(
                    document_id, DocumentStatus.FAILED,
                    error_message=f"Processing failed after {task.request.retries} retries: {exc}",
                )
                await session.commit()
                documents_processed_total.labels(status="failed").inc()
                return
            raise task.retry(
                exc=exc, countdown=settings.CELERY_TASK_RETRY_BACKOFF_SECONDS
            ) from exc

        except AppException as exc:
            # Any other known application error — treat as permanent to
            # avoid infinite retries on something we don't recognize as
            # safely retryable.
            await session.rollback()
            await document_repo.update_status(
                document_id, DocumentStatus.FAILED, error_message=str(exc)
            )
            await session.commit()
            documents_processed_total.labels(status="failed").inc()
            logger.error(
                "Document processing failed with an unexpected application error",
                extra={"document_id": str(document_id), "error": str(exc)},
            )