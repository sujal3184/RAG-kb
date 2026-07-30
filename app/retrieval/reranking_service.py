"""Reranking service — re-scores hybrid retrieval results using a
cross-encoder for improved final ranking quality.

Degrades gracefully: if the reranker fails, results are returned in their
original hybrid-fused order rather than raising an error, since reranking
improves quality but isn't required for the pipeline to function.
"""

import logging

from app.core.exceptions import AppException
from app.retrieval.base import RankedChunk, RetrievedChunk
from app.retrieval.reranker import RerankerProvider

logger = logging.getLogger(__name__)


class RerankingService:
    """Re-ranks a shortlist of retrieved chunks using a cross-encoder reranker."""

    def __init__(self, reranker: RerankerProvider) -> None:
        """Store the reranker provider this service uses.

        Args:
            reranker: the cross-encoder reranker to score candidates with.
        """
        self.reranker = reranker

    async def rerank(
        self, *, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RankedChunk]:
        """Re-score and re-order candidate chunks using the cross-encoder.

        Args:
            query: the original search query.
            candidates: chunks from hybrid retrieval (Module 11) to re-rank.
            top_k: how many top results to return after reranking.

        Returns:
            A list of RankedChunk, ordered by the reranker's relevance
            score (best first). If reranking fails, falls back to
            returning the input order unchanged (truncated to top_k).
        """
        if not candidates:
            return []

        try:
            scores = await self.reranker.score(
                query=query, candidates=[c.text for c in candidates]
            )
        except AppException as exc:
            logger.warning(
                "Reranking failed — falling back to original retrieval order",
                extra={"error": str(exc)},
            )
            return self._fallback_without_scores(candidates, top_k)

        ranked = [
            RankedChunk(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                chunk_index=candidate.chunk_index,
                text=candidate.text,
                rerank_score=score,
                source_methods=candidate.source_methods,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]

        ranked.sort(key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _fallback_without_scores(
        candidates: list[RetrievedChunk], top_k: int
    ) -> list[RankedChunk]:
        """Build RankedChunk results preserving original order, used when
        the reranker itself fails and we degrade gracefully."""
        return [
            RankedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
                rerank_score=c.fused_score,  # reuse fused score as a stand-in
                source_methods=c.source_methods,
            )
            for c in candidates[:top_k]
        ]