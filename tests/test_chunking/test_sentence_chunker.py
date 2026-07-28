"""Tests for SentenceChunker."""

import pytest

from app.chunking.base import TokenCounter
from app.chunking.sentence_chunker import SentenceChunker


@pytest.fixture
def token_counter() -> TokenCounter:
    return TokenCounter("cl100k_base")


def test_never_splits_a_sentence_in_half(token_counter: TokenCounter) -> None:
    chunker = SentenceChunker(chunk_size_tokens=15, chunk_overlap_tokens=3, token_counter=token_counter)
    text = (
        "This is the first sentence about RAG systems. "
        "This is the second sentence about embeddings. "
        "This is the third sentence about retrieval."
    )

    chunks = chunker.chunk(text)

    full_reconstructed = " ".join(c.text for c in chunks)
    for sentence in [
        "This is the first sentence about RAG systems.",
        "This is the second sentence about embeddings.",
        "This is the third sentence about retrieval.",
    ]:
        assert sentence in full_reconstructed


def test_empty_text_returns_no_chunks(token_counter: TokenCounter) -> None:
    chunker = SentenceChunker(chunk_size_tokens=15, chunk_overlap_tokens=3, token_counter=token_counter)
    assert chunker.chunk("") == []


def test_single_short_sentence_returns_one_chunk(token_counter: TokenCounter) -> None:
    chunker = SentenceChunker(chunk_size_tokens=50, chunk_overlap_tokens=5, token_counter=token_counter)
    chunks = chunker.chunk("Just one short sentence.")

    assert len(chunks) == 1
    assert chunks[0].metadata["sentence_count"] == 1