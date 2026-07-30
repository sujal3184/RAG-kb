"""Integration tests for QdrantVectorStore.

Requires a real Qdrant instance (via Docker) — these are integration
tests exercising the actual Qdrant client and collection lifecycle.
"""

import uuid

import pytest

from app.config.settings import get_settings
from app.retrieval.base import VectorPoint
from app.retrieval.qdrant_store import QdrantVectorStore

pytestmark = pytest.mark.docker


@pytest.fixture
async def vector_store():
    """Provide a QdrantVectorStore, cleaning up any test collection afterwards."""
    settings = get_settings()
    store = QdrantVectorStore(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        collection_prefix="test_kb_",
    )
    yield store


def _make_point(
    document_id: uuid.UUID, kb_id: uuid.UUID, chunk_index: int, vector: list[float], text: str
) -> VectorPoint:
    return VectorPoint(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        knowledge_base_id=kb_id,
        chunk_index=chunk_index,
        text=text,
        vector=vector,
    )


@pytest.mark.asyncio
async def test_upsert_and_search_returns_most_similar_vector(vector_store) -> None:
    """Searching with a vector close to a stored one should return it first."""
    kb_id = uuid.uuid4()
    document_id = uuid.uuid4()

    points = [
        _make_point(document_id, kb_id, 0, [1.0, 0.0, 0.0, 0.0], "About cats"),
        _make_point(document_id, kb_id, 1, [0.0, 1.0, 0.0, 0.0], "About stock markets"),
    ]
    await vector_store.upsert(knowledge_base_id=kb_id, points=points, vector_dimension=4)

    results = await vector_store.search(
        knowledge_base_id=kb_id, query_vector=[0.9, 0.1, 0.0, 0.0], top_k=1
    )

    assert len(results) == 1
    assert results[0].text == "About cats"

    await vector_store.delete_collection(knowledge_base_id=kb_id)


@pytest.mark.asyncio
async def test_search_on_nonexistent_collection_returns_empty(vector_store) -> None:
    """Searching a knowledge base with no embedded documents should return
    an empty list, not raise an error."""
    kb_id = uuid.uuid4()

    results = await vector_store.search(
        knowledge_base_id=kb_id, query_vector=[0.1, 0.2, 0.3, 0.4], top_k=5
    )

    assert results == []


@pytest.mark.asyncio
async def test_upsert_same_document_chunk_overwrites_not_duplicates(vector_store) -> None:
    """Re-embedding the same document/chunk index should overwrite the
    existing vector, not create a duplicate point."""
    kb_id = uuid.uuid4()
    document_id = uuid.uuid4()

    original = _make_point(document_id, kb_id, 0, [1.0, 0.0, 0.0, 0.0], "Original text")
    await vector_store.upsert(knowledge_base_id=kb_id, points=[original], vector_dimension=4)

    updated = _make_point(document_id, kb_id, 0, [1.0, 0.0, 0.0, 0.0], "Updated text")
    await vector_store.upsert(knowledge_base_id=kb_id, points=[updated], vector_dimension=4)

    results = await vector_store.search(
        knowledge_base_id=kb_id, query_vector=[1.0, 0.0, 0.0, 0.0], top_k=10
    )

    assert len(results) == 1
    assert results[0].text == "Updated text"

    await vector_store.delete_collection(knowledge_base_id=kb_id)


@pytest.mark.asyncio
async def test_delete_by_document_removes_only_that_documents_vectors(vector_store) -> None:
    """Deleting one document's vectors should leave other documents in
    the same knowledge base untouched."""
    kb_id = uuid.uuid4()
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    points = [
        _make_point(doc_a, kb_id, 0, [1.0, 0.0, 0.0, 0.0], "Document A content"),
        _make_point(doc_b, kb_id, 0, [0.0, 1.0, 0.0, 0.0], "Document B content"),
    ]
    await vector_store.upsert(knowledge_base_id=kb_id, points=points, vector_dimension=4)

    await vector_store.delete_by_document(knowledge_base_id=kb_id, document_id=doc_a)

    results = await vector_store.search(
        knowledge_base_id=kb_id, query_vector=[0.5, 0.5, 0.0, 0.0], top_k=10
    )

    assert len(results) == 1
    assert results[0].text == "Document B content"

    await vector_store.delete_collection(knowledge_base_id=kb_id)


@pytest.mark.asyncio
async def test_search_can_filter_by_document_id(vector_store) -> None:
    """Providing document_id to search should restrict results to that document only."""
    kb_id = uuid.uuid4()
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    points = [
        _make_point(doc_a, kb_id, 0, [1.0, 0.0, 0.0, 0.0], "From document A"),
        _make_point(doc_b, kb_id, 0, [1.0, 0.0, 0.0, 0.0], "From document B"),
    ]
    await vector_store.upsert(knowledge_base_id=kb_id, points=points, vector_dimension=4)

    results = await vector_store.search(
        knowledge_base_id=kb_id,
        query_vector=[1.0, 0.0, 0.0, 0.0],
        top_k=10,
        document_id=doc_a,
    )

    assert len(results) == 1
    assert results[0].text == "From document A"

    await vector_store.delete_collection(knowledge_base_id=kb_id)


@pytest.mark.asyncio
async def test_delete_collection_removes_all_vectors(vector_store) -> None:
    """Deleting a whole collection should make subsequent searches return empty."""
    kb_id = uuid.uuid4()
    document_id = uuid.uuid4()

    points = [_make_point(document_id, kb_id, 0, [1.0, 0.0, 0.0, 0.0], "Some content")]
    await vector_store.upsert(knowledge_base_id=kb_id, points=points, vector_dimension=4)

    await vector_store.delete_collection(knowledge_base_id=kb_id)

    results = await vector_store.search(
        knowledge_base_id=kb_id, query_vector=[1.0, 0.0, 0.0, 0.0], top_k=10
    )
    assert results == []