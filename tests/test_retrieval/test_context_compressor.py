"""Fast unit tests for ContextCompressor's orchestration logic."""

import uuid

import pytest

from app.chunking.base import TokenCounter
from app.retrieval.base import RankedChunk
from app.retrieval.context_compressor import ContextCompressor


class PassthroughDeduplicator:
    """A fake deduplicator that returns chunks unchanged — isolates
    ContextCompressor's token-budget logic from deduplication behavior,
    which is already tested separately in test_deduplication.py."""

    async def deduplicate(self, chunks, *, similarity_threshold: float):
        return chunks


def _make_chunk(chunk_id: str, text: str, score: float = 1.0) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id, document_id=uuid.uuid4(), chunk_index=0, text=text, rerank_score=score
    )


@pytest.fixture
def token_counter() -> TokenCounter:
    return TokenCounter("cl100k_base")


@pytest.mark.asyncio
async def test_includes_all_chunks_when_within_budget(token_counter: TokenCounter) -> None:
    chunks = [_make_chunk("a", "Short text one."), _make_chunk("b", "Short text two.")]
    compressor = ContextCompressor(PassthroughDeduplicator(), token_counter)

    result = await compressor.compress(
        chunks, similarity_threshold=0.9, max_context_tokens=1000
    )

    assert len(result) == 2


@pytest.mark.asyncio
async def test_stops_including_chunks_once_budget_exceeded(token_counter: TokenCounter) -> None:
    long_text = " ".join(f"word{i}" for i in range(100))  # ~100+ tokens
    chunks = [
        _make_chunk("first", long_text),
        _make_chunk("second", long_text),
        _make_chunk("third", long_text),
    ]
    compressor = ContextCompressor(PassthroughDeduplicator(), token_counter)

    # Budget only large enough for roughly 1-2 chunks of this size
    result = await compressor.compress(
        chunks, similarity_threshold=0.9, max_context_tokens=150
    )

    assert len(result) < len(chunks)
    total_tokens = sum(c.token_count for c in result)
    assert total_tokens <= 150


@pytest.mark.asyncio
async def test_never_splits_a_chunk_partially(token_counter: TokenCounter) -> None:
    """Every included chunk's full original text should be preserved —
    compression trims WHICH chunks are included, never a chunk's content."""
    chunks = [_make_chunk("a", "This is a complete sentence that should not be cut.")]
    compressor = ContextCompressor(PassthroughDeduplicator(), token_counter)

    result = await compressor.compress(
        chunks, similarity_threshold=0.9, max_context_tokens=1000
    )

    assert result[0].text == "This is a complete sentence that should not be cut."


@pytest.mark.asyncio
async def test_empty_input_returns_empty_list(token_counter: TokenCounter) -> None:
    compressor = ContextCompressor(PassthroughDeduplicator(), token_counter)
    result = await compressor.compress([], similarity_threshold=0.9, max_context_tokens=1000)
    assert result == []


@pytest.mark.asyncio
async def test_preserves_rerank_score_and_order(token_counter: TokenCounter) -> None:
    chunks = [
        _make_chunk("high", "Top result text.", score=0.9),
        _make_chunk("low", "Lower result text.", score=0.3),
    ]
    compressor = ContextCompressor(PassthroughDeduplicator(), token_counter)

    result = await compressor.compress(
        chunks, similarity_threshold=0.9, max_context_tokens=1000
    )

    assert result[0].chunk_id == "high"
    assert result[0].rerank_score == 0.9