"""Tests for ChunkingFactory."""

from app.chunking.factory import ChunkingFactory, ChunkingStrategyType
from app.chunking.fixed_size_chunker import FixedSizeChunker
from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.sentence_chunker import SentenceChunker
from app.config.settings import get_settings


def test_defaults_to_recursive_chunker() -> None:
    factory = ChunkingFactory(get_settings())
    strategy = factory.get_strategy()
    assert isinstance(strategy, RecursiveChunker)


def test_can_request_fixed_size_chunker() -> None:
    factory = ChunkingFactory(get_settings())
    strategy = factory.get_strategy(ChunkingStrategyType.FIXED_SIZE)
    assert isinstance(strategy, FixedSizeChunker)


def test_can_request_sentence_chunker() -> None:
    factory = ChunkingFactory(get_settings())
    strategy = factory.get_strategy(ChunkingStrategyType.SENTENCE)
    assert isinstance(strategy, SentenceChunker)


def test_override_chunk_size_and_overlap() -> None:
    factory = ChunkingFactory(get_settings())
    strategy = factory.get_strategy(chunk_size_tokens=123, chunk_overlap_tokens=12)
    assert strategy.chunk_size_tokens == 123
    assert strategy.chunk_overlap_tokens == 12