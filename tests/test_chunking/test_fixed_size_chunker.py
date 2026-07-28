"""Tests for FixedSizeChunker."""

import pytest

from app.chunking.base import TokenCounter
from app.chunking.exceptions import ChunkingError
from app.chunking.fixed_size_chunker import FixedSizeChunker


@pytest.fixture
def token_counter() -> TokenCounter:
    return TokenCounter("cl100k_base")


def test_splits_text_into_expected_number_of_chunks(token_counter: TokenCounter) -> None:
    chunker = FixedSizeChunker(chunk_size_tokens=10, chunk_overlap_tokens=2, token_counter=token_counter)
    text = " ".join(f"word{i}" for i in range(50))  # roughly 50+ tokens

    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 10


def test_empty_text_returns_no_chunks(token_counter: TokenCounter) -> None:
    chunker = FixedSizeChunker(chunk_size_tokens=10, chunk_overlap_tokens=2, token_counter=token_counter)
    assert chunker.chunk("   ") == []


def test_rejects_overlap_greater_than_or_equal_to_chunk_size(token_counter: TokenCounter) -> None:
    with pytest.raises(ChunkingError):
        FixedSizeChunker(chunk_size_tokens=10, chunk_overlap_tokens=10, token_counter=token_counter)


def test_consecutive_chunks_overlap(token_counter: TokenCounter) -> None:
    chunker = FixedSizeChunker(chunk_size_tokens=20, chunk_overlap_tokens=5, token_counter=token_counter)
    text = " ".join(f"word{i}" for i in range(60))

    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    # Some trailing tokens of chunk 1 should reappear at the start of chunk 2
    first_chunk_tail = token_counter.encode(chunks[0].text)[-5:]
    second_chunk_head = token_counter.encode(chunks[1].text)[:5]
    assert first_chunk_tail == second_chunk_head