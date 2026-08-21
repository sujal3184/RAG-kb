"""Cohere Rerank API implementation of RerankerProvider.

Cohere's rerank endpoint runs on hosted GPU infrastructure, making it
dramatically faster than CPU-based local cross-encoder inference — this
has consistently been the largest latency contributor in this pipeline
(20-40s locally vs. typically under 1s via this API).

Same RerankerProvider interface as BgeRerankerProvider (Module 12), so
RerankingService's fail-open fallback behavior is unaffected by this swap.
"""

import logging

import cohere

from app.retrieval.exceptions import VectorStoreError
from app.retrieval.reranker import RerankerProvider

logger = logging.getLogger(__name__)


class CohereRerankerProvider(RerankerProvider):
    """Reranker backed by Cohere's hosted Rerank API."""

    def __init__(self, api_key: str, *, model: str = "rerank-v3.5") -> None:
        """Configure the Cohere client.

        Args:
            api_key: Cohere API key.
            model: which Cohere rerank model to use. "rerank-v3.5" is
                their current general-purpose multilingual model.
        """
        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model

    async def score(self, *, query: str, candidates: list[str]) -> list[float]:
        """Score how relevant each candidate is to the query via Cohere's API.

        Raises:
            VectorStoreError: if the API call fails.
        """
        if not candidates:
            return []

        try:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=candidates,
            )
        except Exception as exc:
            raise VectorStoreError(f"Cohere reranking failed: {exc}") from exc

        # Cohere returns results sorted by relevance with each result's
        # ORIGINAL index — we need scores back in the SAME ORDER as the
        # input candidates list, since RerankingService.rerank() zips
        # scores against candidates positionally.
        scores_by_index = {result.index: result.relevance_score for result in response.results}
        return [scores_by_index.get(i, 0.0) for i in range(len(candidates))]