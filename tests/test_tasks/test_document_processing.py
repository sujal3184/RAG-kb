"""Integration tests for the document processing Celery task.

These call the task function DIRECTLY (synchronously, in-process) rather
than through a real Celery worker + broker — this tests the actual
processing LOGIC without needing a running Celery worker process, while
still exercising real Postgres, Qdrant, and embedding models.
"""

import uuid

import pytest
from sqlalchemy import select

from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.storage.local_storage import LocalFileStorage
from app.tasks.document_processing import _process_document_async, DocumentProcessingTask

pytestmark = pytest.mark.docker


class FakeCeleryRequest:
    """Minimal stand-in for Celery's task.request, used only to read
    `.retries` inside the task's retryable-error branch."""

    retries = 0


class FakeTaskForDirectInvocation(DocumentProcessingTask):
    """Lets us call the task's async logic directly without a real Celery
    worker/broker, while still using real embedding/vector store
    dependencies via the base class's lazy properties."""

    request = FakeCeleryRequest()

    def retry(self, exc=None, countdown=None):
        raise exc


@pytest.fixture
async def seeded_document(tmp_path) -> Document:
    """Create a real User -> KnowledgeBase -> Document chain, with an
    actual text file saved via LocalFileStorage, ready for processing.

    Uses its OWN independent session (not the shared db_session fixture)
    and commits immediately, since the task under test opens its own
    separate database connection — exactly like a real Celery worker
    process would. Using the transactional db_session fixture here would
    leave this data invisible to the task's connection.
    """
    from app.db.session import async_session_factory

    async with async_session_factory() as session:
        user = await UserRepository(session).create(
            User(email=f"task_test_{uuid.uuid4()}@example.com", hashed_password="hashed")
        )
        kb = KnowledgeBase(owner_id=user.id, name="Task Test KB")
        session.add(kb)
        await session.flush()

        storage = LocalFileStorage(base_path=str(tmp_path))
        content = b"Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris."
        storage_ref = await storage.save(path="test_doc.txt", content=content)

        document = Document(
            knowledge_base_id=kb.id,
            original_filename="test_doc.txt",
            file_extension="txt",
            content_type="text/plain",
            size_bytes=len(content),
            storage_ref=storage_ref,
            status=DocumentStatus.PENDING,
        )
        session.add(document)
        await session.flush()
        await session.commit()

        return document


@pytest.mark.asyncio
async def test_processing_creates_chunks_and_marks_ready(seeded_document, tmp_path) -> None:
    """A valid document should end up with status READY and real Chunk rows."""
    from app.db.session import async_session_factory

    task = FakeTaskForDirectInvocation()
    task._file_storage = LocalFileStorage(base_path=str(tmp_path))

    await _process_document_async(task, seeded_document.id)

    async with async_session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.id == seeded_document.id)
        )
        updated_document = result.scalar_one()
        assert updated_document.status == DocumentStatus.READY
        assert updated_document.error_message is None

        chunks_result = await session.execute(
            select(Chunk).where(Chunk.document_id == seeded_document.id)
        )
        chunks = list(chunks_result.scalars().all())
        assert len(chunks) > 0
        assert any("Paris" in c.text for c in chunks)


@pytest.mark.asyncio
async def test_missing_document_is_skipped_gracefully(db_session) -> None:
    """Processing a document_id that doesn't exist should not raise."""
    task = FakeTaskForDirectInvocation()

    await _process_document_async(task, uuid.uuid4())  # should simply log and return


@pytest.mark.asyncio
async def test_unsupported_content_marks_document_failed(tmp_path) -> None:
    """A document whose loader fails (e.g. corrupted content) should end
    up FAILED with a populated error_message, not stuck in PROCESSING."""
    from app.db.session import async_session_factory

    async with async_session_factory() as session:
        user = await UserRepository(session).create(
            User(email=f"task_fail_{uuid.uuid4()}@example.com", hashed_password="hashed")
        )
        kb = KnowledgeBase(owner_id=user.id, name="Failing Task KB")
        session.add(kb)
        await session.flush()

        storage = LocalFileStorage(base_path=str(tmp_path))
        storage_ref = await storage.save(path="broken.pdf", content=b"not a real pdf at all")

        document = Document(
            knowledge_base_id=kb.id,
            original_filename="broken.pdf",
            file_extension="pdf",
            content_type="application/pdf",
            size_bytes=20,
            storage_ref=storage_ref,
            status=DocumentStatus.PENDING,
        )
        session.add(document)
        await session.flush()
        await session.commit()
        document_id = document.id

    task = FakeTaskForDirectInvocation()
    task._file_storage = storage

    await _process_document_async(task, document_id)

    async with async_session_factory() as verify_session:
        result = await verify_session.execute(select(Document).where(Document.id == document_id))
        updated_document = result.scalar_one()
        assert updated_document.status == DocumentStatus.FAILED
        assert updated_document.error_message is not None