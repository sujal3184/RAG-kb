"""Fast unit tests for Deduplicator, using a fake embedding service so no
real model inference is needed."""

import uuid

import pytest

from app.embeddings.base import EmbeddingResult
from app.retrieval.base import RankedChunk
from app.retrieval.deduplication import Deduplicator


class FakeEmbeddingService:
    """Returns pre-configured vectors for known chunk texts, so we can
    control similarity relationships deterministically in tests."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        vectors = [self._vectors_by_text[text] for text in texts]
        return EmbeddingResult(vectors=vectors, model_name="fake", dimension=len(vectors[0]))


def _make_chunk(chunk_id: str, text: str) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id, document_id=uuid.uuid4(), chunk_index=0, text=text, rerank_score=1.0
    )


@pytest.mark.asyncio
async def test_removes_near_duplicate_keeping_first() -> None:
    """Two chunks with nearly identical vectors should collapse to one,
    keeping whichever appeared first (already the higher-ranked one)."""
    chunks = [
        _make_chunk("first", "Paris is the capital of France."),
        _make_chunk("duplicate", "The capital city of France is Paris."),
    ]
    vectors = {
        "Paris is the capital of France.": [1.0, 0.0, 0.0],
        "The capital city of France is Paris.": [0.99, 0.01, 0.0],  # nearly identical direction
    }
    deduplicator = Deduplicator(FakeEmbeddingService(vectors))

    result = await deduplicator.deduplicate(chunks, similarity_threshold=0.95)

    assert len(result) == 1
    assert result[0].chunk_id == "first"


@pytest.mark.asyncio
async def test_keeps_distinct_chunks() -> None:
    """Chunks with clearly different content/vectors should both be kept."""
    chunks = [
        _make_chunk("about-paris", "Paris is the capital of France."),
        _make_chunk("about-bananas", "Bananas are rich in potassium."),
    ]
    vectors = {
        "Paris is the capital of France.": [1.0, 0.0, 0.0],
        "Bananas are rich in potassium.": [0.0, 1.0, 0.0],
    }
    deduplicator = Deduplicator(FakeEmbeddingService(vectors))

    result = await deduplicator.deduplicate(chunks, similarity_threshold=0.95)

    assert len(result) == 2


@pytest.mark.asyncio
async def test_single_chunk_returned_unchanged() -> None:
    chunks = [_make_chunk("only-one", "Some text")]
    deduplicator = Deduplicator(FakeEmbeddingService({"Some text": [1.0, 0.0]}))

    result = await deduplicator.deduplicate(chunks, similarity_threshold=0.9)

    assert result == chunks


@pytest.mark.asyncio
async def test_empty_list_returns_empty_list() -> None:
    deduplicator = Deduplicator(FakeEmbeddingService({}))
    result = await deduplicator.deduplicate([], similarity_threshold=0.9)
    assert result == []