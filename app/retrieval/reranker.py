"""Reranker provider interface and BAAI/bge-reranker-v2-m3 implementation.

A reranker (cross-encoder) scores a query against ONE candidate at a time
by processing both together, producing far more accurate relevance
judgments than bi-encoder-based retrieval alone. This makes it too slow
to run over an entire knowledge base, but ideal for re-scoring a small
shortlist of candidates that retrieval already narrowed down.
"""

import asyncio
import logging
from abc import ABC, abstractmethod

from sentence_transformers import CrossEncoder

from app.retrieval.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class RerankerProvider(ABC):
    """Abstract base class for a cross-encoder reranking model."""

    @abstractmethod
    async def score(self, *, query: str, candidates: list[str]) -> list[float]:
        """Score how relevant each candidate text is to the query.

        Args:
            query: the search query.
            candidates: candidate texts to score against the query.

        Returns:
            A list of relevance scores, one per candidate, in the SAME
            order as the input `candidates` list (higher = more relevant).
        """
        raise NotImplementedError


class BgeRerankerProvider(RerankerProvider):
    """Reranker backed by BAAI/bge-reranker-v2-m3, via sentence-transformers'
    CrossEncoder."""

    def __init__(self, model_id: str, *, cache_dir: str, batch_size: int) -> None:
        """Store configuration; the model is loaded lazily on first use.

        Args:
            model_id: HuggingFace model identifier (e.g.
                "BAAI/bge-reranker-v2-m3").
            cache_dir: where to cache downloaded model weights.
            batch_size: how many query-candidate pairs to score per
                internal model call.
        """
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model: CrossEncoder | None = None

    async def score(self, *, query: str, candidates: list[str]) -> list[float]:
        """Score each candidate's relevance to the query using the cross-encoder.

        Raises:
            VectorStoreError: if the model fails to load or run.
        """
        if not candidates:
            return []

        try:
            model = await asyncio.to_thread(self._get_or_load_model)
            pairs = [[query, candidate] for candidate in candidates]
            scores = await asyncio.to_thread(
                lambda: model.predict(pairs, batch_size=self._batch_size).tolist()
            )
        except Exception as exc:
            raise VectorStoreError(f"Reranking failed: {exc}") from exc

        return scores

    def _get_or_load_model(self) -> CrossEncoder:
        """Load the cross-encoder model on first use, then reuse it."""
        if self._model is None:
            logger.info("Loading reranker model", extra={"model": self._model_id})
            self._model = CrossEncoder(self._model_id, cache_folder=self._cache_dir)
            logger.info("Reranker model loaded", extra={"model": self._model_id})
        return self._model