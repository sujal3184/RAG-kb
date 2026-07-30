"""Tests for HybridRetriever's fusion orchestration, using fake
dense/BM25/embedding dependencies to avoid needing Qdrant or real models.
"""

import uuid

import pytest

from app.embeddings.base import EmbeddingResult
from app.retrieval.base import SearchResult
from app.retrieval.bm25_store import BM25Document, BM25Result
from app.retrieval.hybrid_retriever import HybridRetriever


class FakeVectorStore:
    """A fake VectorStore returning pre-configured dense search results."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def search(self, **kwargs) -> list[SearchResult]:
        return self._results

    async def upsert(self, **kwargs) -> None:
        pass

    async def delete_by_document(self, **kwargs) -> None:
        pass

    async def delete_collection(self, **kwargs) -> None:
        pass


class FakeBM25Store:
    """A fake BM25Store returning pre-configured keyword search results."""

    def __init__(self, results: list[BM25Result]) -> None:
        self._results = results

    def search(self, **kwargs) -> list[BM25Result]:
        return self._results


class FakeEmbeddingService:
    """A fake EmbeddingService avoiding real model inference."""

    async def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_chunk_found_by_both_methods_ranks_first() -> None:
    """A chunk appearing in BOTH dense and BM25 results should be ranked
    above a chunk found by only one method."""
    document_id = uuid.uuid4()
    shared_chunk_id = "shared-chunk"
    dense_only_chunk_id = "dense-only-chunk"

    dense_results = [
        SearchResult(
            chunk_id=shared_chunk_id, document_id=document_id, chunk_index=0,
            text="Found by both methods", score=0.9,
        ),
        SearchResult(
            chunk_id=dense_only_chunk_id, document_id=document_id, chunk_index=1,
            text="Found only by dense search", score=0.8,
        ),
    ]
    bm25_results = [
        BM25Result(
            chunk_id=shared_chunk_id, document_id=document_id, chunk_index=0,
            text="Found by both methods", score=5.0,
        ),
    ]

    retriever = HybridRetriever(
        FakeVectorStore(dense_results),
        FakeBM25Store(bm25_results),
        FakeEmbeddingService(),
        top_k_per_method=10,
        rrf_k=60,
    )

    results = await retriever.retrieve(
        knowledge_base_id=uuid.uuid4(),
        query="test query",
        bm25_documents=[],
        top_k=10,
    )

    assert results[0].chunk_id == shared_chunk_id
    assert set(results[0].source_methods) == {"dense", "bm25"}


@pytest.mark.asyncio
async def test_chunk_found_by_only_bm25_still_appears_in_results() -> None:
    """A chunk found ONLY by BM25 (e.g. an exact keyword match dense
    search missed) should still appear in the final fused results."""
    document_id = uuid.uuid4()
    bm25_only_chunk_id = "bm25-only-chunk"

    bm25_results = [
        BM25Result(
            chunk_id=bm25_only_chunk_id, document_id=document_id, chunk_index=0,
            text="Contains exact product code SKU-99999", score=8.0,
        ),
    ]

    retriever = HybridRetriever(
        FakeVectorStore([]),
        FakeBM25Store(bm25_results),
        FakeEmbeddingService(),
        top_k_per_method=10,
        rrf_k=60,
    )

    results = await retriever.retrieve(
        knowledge_base_id=uuid.uuid4(),
        query="SKU-99999",
        bm25_documents=[],
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].chunk_id == bm25_only_chunk_id
    assert results[0].source_methods == ["bm25"]


@pytest.mark.asyncio
async def test_respects_top_k_limit() -> None:
    """The final result count should never exceed the requested top_k,
    even if more candidates were available from either method."""
    document_id = uuid.uuid4()
    dense_results = [
        SearchResult(chunk_id=f"chunk-{i}", document_id=document_id, chunk_index=i, text=f"text {i}", score=1.0 - i * 0.1)
        for i in range(10)
    ]

    retriever = HybridRetriever(
        FakeVectorStore(dense_results),
        FakeBM25Store([]),
        FakeEmbeddingService(),
        top_k_per_method=10,
        rrf_k=60,
    )

    results = await retriever.retrieve(
        knowledge_base_id=uuid.uuid4(),
        query="test",
        bm25_documents=[],
        top_k=3,
    )

    assert len(results) == 3


@pytest.mark.asyncio
async def test_no_results_from_either_method_returns_empty() -> None:
    retriever = HybridRetriever(
        FakeVectorStore([]),
        FakeBM25Store([]),
        FakeEmbeddingService(),
        top_k_per_method=10,
        rrf_k=60,
    )

    results = await retriever.retrieve(
        knowledge_base_id=uuid.uuid4(),
        query="anything",
        bm25_documents=[],
        top_k=10,
    )

    assert results == []