"""Chunking interfaces and shared building blocks.

A `ChunkingStrategy` splits a single block of text into smaller `Chunk`
objects, each within a target token budget. All strategies share the same
tokenizer-based size measurement (via tiktoken) so "chunk size" means the
same thing regardless of which strategy produced the chunk.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import tiktoken


@dataclass
class Chunk:
    """A single chunk of text produced by a chunking strategy.

    Attributes:
        text: the chunk's actual text content.
        start_index: character offset where this chunk begins in the
            original source text (useful for "jump to source" features
            and for verifying chunks in tests).
        end_index: character offset where this chunk ends in the
            original source text.
        token_count: how many tokens this chunk contains, per the
            configured tokenizer.
        metadata: any additional info a strategy wants to attach (e.g.
            which paragraph/page this came from).
    """

    text: str
    start_index: int
    end_index: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenCounter:
    """Wraps tiktoken to count and encode/decode tokens consistently.

    Every chunker uses THIS class (not tiktoken directly) so the token
    counting logic — and which encoding is used — lives in exactly one
    place.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Load the tokenizer encoding.

        Args:
            encoding_name: the tiktoken encoding to use. "cl100k_base" is
                the encoding used by GPT-3.5/GPT-4-family models and is a
                reasonable, widely-compatible default for measuring text
                size even when the final embedding/LLM model differs.
        """
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of token ids."""
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Convert a list of token ids back into text."""
        return self._encoding.decode(tokens)


class ChunkingStrategy(ABC):
    """Abstract base class for splitting text into chunks."""

    def __init__(
        self,
        *,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        token_counter: TokenCounter,
    ) -> None:
        """Store the sizing configuration shared by all strategies.

        Args:
            chunk_size_tokens: target maximum tokens per chunk.
            chunk_overlap_tokens: how many tokens of overlap to include
                between consecutive chunks, to preserve context across
                chunk boundaries.
            token_counter: shared tokenizer wrapper for consistent sizing.

        Raises:
            ChunkingError: if the overlap is not smaller than the chunk size
                (which would cause chunks to never make forward progress).
        """
        from app.chunking.exceptions import ChunkingError

        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ChunkingError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens, "
                "otherwise chunking would never advance through the text"
            )

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.token_counter = token_counter

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into a list of Chunk objects.

        Args:
            text: the full text to split (typically a LoadedDocument.text
                from Module 7).

        Returns:
            An ordered list of Chunk objects covering the input text.
        """
        raise NotImplementedError