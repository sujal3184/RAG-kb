"""Tests for RecursiveChunker."""

import pytest

from app.chunking.base import TokenCounter
from app.chunking.recursive_chunker import RecursiveChunker


@pytest.fixture
def token_counter() -> TokenCounter:
    return TokenCounter("cl100k_base")


def test_keeps_short_paragraphs_together(token_counter: TokenCounter) -> None:
    chunker = RecursiveChunker(chunk_size_tokens=100, chunk_overlap_tokens=10, token_counter=token_counter)
    text = "Paragraph one is short.\n\nParagraph two is also short."

    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert "Paragraph one is short." in chunks[0].text
    assert "Paragraph two is also short." in chunks[0].text


def test_splits_across_paragraphs_when_budget_exceeded(token_counter: TokenCounter) -> None:
    chunker = RecursiveChunker(chunk_size_tokens=15, chunk_overlap_tokens=3, token_counter=token_counter)
    paragraph_a = "This is paragraph A with several words in it for testing purposes."
    paragraph_b = "This is paragraph B with different words for testing the chunker."
    text = f"{paragraph_a}\n\n{paragraph_b}"

    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 15


def test_handles_oversized_single_paragraph_via_sentence_fallback(
    token_counter: TokenCounter,
) -> None:
    chunker = RecursiveChunker(chunk_size_tokens=10, chunk_overlap_tokens=2, token_counter=token_counter)
    long_paragraph = (
        "This is sentence one of a very long paragraph. "
        "This is sentence two of the same long paragraph. "
        "This is sentence three, still in one paragraph."
    )

    chunks = chunker.chunk(long_paragraph)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 10


def test_all_text_is_preserved_across_chunks(token_counter: TokenCounter) -> None:
    """No sentence/word should be silently dropped during recursive splitting."""
    chunker = RecursiveChunker(chunk_size_tokens=20, chunk_overlap_tokens=4, token_counter=token_counter)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."

    chunks = chunker.chunk(text)
    combined = " ".join(c.text for c in chunks)

    assert "First paragraph here." in combined
    assert "Second paragraph here." in combined
    assert "Third paragraph here." in combined