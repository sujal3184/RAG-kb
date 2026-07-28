"""Chunking factory — creates the appropriate ChunkingStrategy.

Centralizes strategy selection AND sizing configuration, so callers don't
need to import tiktoken or wire up TokenCounter themselves.
"""

from enum import StrEnum

from app.chunking.base import ChunkingStrategy, TokenCounter
from app.chunking.fixed_size_chunker import FixedSizeChunker
from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.sentence_chunker import SentenceChunker
from app.config.settings import Settings


class ChunkingStrategyType(StrEnum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"


class ChunkingFactory:
    """Creates a configured ChunkingStrategy instance."""

    _STRATEGIES: dict[ChunkingStrategyType, type[ChunkingStrategy]] = {
        ChunkingStrategyType.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategyType.SENTENCE: SentenceChunker,
        ChunkingStrategyType.RECURSIVE: RecursiveChunker,
    }

    def __init__(self, settings: Settings) -> None:
        """Store settings used for default sizing and tokenizer configuration.

        Args:
            settings: app settings providing chunk size/overlap defaults
                and which tiktoken encoding to use.
        """
        self.settings = settings
        self._token_counter = TokenCounter(settings.TOKENIZER_ENCODING)

    def get_strategy(
        self,
        strategy_type: ChunkingStrategyType = ChunkingStrategyType.RECURSIVE,
        *,
        chunk_size_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
    ) -> ChunkingStrategy:
        """Create a chunking strategy instance.

        Args:
            strategy_type: which strategy to use. Defaults to RECURSIVE,
                the safest general-purpose choice.
            chunk_size_tokens: override the configured default chunk size.
            chunk_overlap_tokens: override the configured default overlap.

        Returns:
            A ready-to-use ChunkingStrategy instance.
        """
        strategy_class = self._STRATEGIES[strategy_type]
        return strategy_class(
            chunk_size_tokens=chunk_size_tokens or self.settings.DEFAULT_CHUNK_SIZE_TOKENS,
            chunk_overlap_tokens=chunk_overlap_tokens
            or self.settings.DEFAULT_CHUNK_OVERLAP_TOKENS,
            token_counter=self._token_counter,
        )