"""Fast unit tests for RerankingService, using a fake reranker to avoid
loading any real cross-encoder model."""

import uuid

import pytest

from app.core.exceptions import AppException
from app.retrieval.base import RetrievedChunk
from app.retrieval.reranking_service import RerankingService


class FakeRerankerProvider:
    """A fake reranker returning pre-configured scores, or raising an
    error if configured to simulate failure."""

    def __init__(self, scores: list[float] | None = None, *, should_fail: bool = False) -> None:
        self._scores = scores
        self._should_fail = should_fail

    async def score(self, *, query: str, candidates: list[str]) -> list[float]:
        if self._should_fail:
            raise AppException("Fake reranker failure")
        return self._scores if self._scores is not None else [0.0] * len(candidates)


def _make_chunk(chunk_id: str, text: str, fused_score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=text,
        fused_score=fused_score,
        source_methods=["dense"],
    )


@pytest.mark.asyncio
async def test_reorders_candidates_by_reranker_score() -> None:
    """A candidate that was ranked lower by hybrid retrieval, but scores
    higher from the reranker, should end up first after reranking."""
    candidates = [
        _make_chunk("low-relevance", "Something mostly unrelated"),
        _make_chunk("high-relevance", "Exactly what the query is about"),
    ]
    # Reranker scores in candidate order: first gets 0.1, second gets 0.9
    reranker = FakeRerankerProvider(scores=[0.1, 0.9])
    service = RerankingService(reranker)

    results = await service.rerank(query="test query", candidates=candidates, top_k=10)

    assert results[0].chunk_id == "high-relevance"
    assert results[0].rerank_score == 0.9


@pytest.mark.asyncio
async def test_respects_top_k() -> None:
    candidates = [_make_chunk(f"chunk-{i}", f"text {i}") for i in range(5)]
    reranker = FakeRerankerProvider(scores=[0.5, 0.9, 0.1, 0.7, 0.3])
    service = RerankingService(reranker)

    results = await service.rerank(query="test", candidates=candidates, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"  # score 0.9
    assert results[1].chunk_id == "chunk-3"  # score 0.7


@pytest.mark.asyncio
async def test_empty_candidates_returns_empty_list() -> None:
    service = RerankingService(FakeRerankerProvider(scores=[]))
    results = await service.rerank(query="test", candidates=[], top_k=10)
    assert results == []


@pytest.mark.asyncio
async def test_falls_back_to_original_order_when_reranker_fails() -> None:
    """If the reranker itself fails, results should degrade gracefully to
    the original hybrid-fused order rather than raising an error."""
    candidates = [
        _make_chunk("first", "text one", fused_score=0.9),
        _make_chunk("second", "text two", fused_score=0.5),
    ]
    failing_reranker = FakeRerankerProvider(should_fail=True)
    service = RerankingService(failing_reranker)

    results = await service.rerank(query="test", candidates=candidates, top_k=10)

    assert len(results) == 2
    assert results[0].chunk_id == "first"  # original order preserved
    assert results[0].rerank_score == 0.9  # fused_score used as stand-in


@pytest.mark.asyncio
async def test_preserves_source_methods_metadata() -> None:
    candidates = [_make_chunk("chunk-1", "some text")]
    candidates[0].source_methods = ["dense", "bm25"]
    reranker = FakeRerankerProvider(scores=[0.8])
    service = RerankingService(reranker)

    results = await service.rerank(query="test", candidates=candidates, top_k=10)

    assert results[0].source_methods == ["dense", "bm25"]