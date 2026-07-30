"""Near-duplicate chunk removal via embedding similarity.

Two chunks rarely have identical text, but can carry near-identical
meaning (e.g., the same fact stated in two different source documents).
Comparing their embeddings catches this in a way exact text matching
cannot.
"""

import logging

import numpy as np

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.base import RankedChunk

logger = logging.getLogger(__name__)


class Deduplicator:
    """Removes near-duplicate chunks from a ranked list, keeping the
    highest-ranked representative of each duplicate cluster."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """Store the embedding service used to compare chunk similarity.

        Args:
            embedding_service: used to embed chunk text for cosine
                similarity comparison (Module 9).
        """
        self.embedding_service = embedding_service

    async def deduplicate(
        self, chunks: list[RankedChunk], *, similarity_threshold: float
    ) -> list[RankedChunk]:
        """Remove near-duplicate chunks, keeping the highest-ranked of each group.

        Args:
            chunks: chunks already ordered by relevance (best first) —
                order MATTERS here, since we keep whichever chunk in a
                duplicate cluster appears FIRST.
            similarity_threshold: cosine similarity above which two
                chunks are considered near-duplicates (0.0 to 1.0).

        Returns:
            A filtered list preserving the original relative order, with
            near-duplicates removed.
        """
        if len(chunks) <= 1:
            return chunks

        embed_result = await self.embedding_service.embed_texts([c.text for c in chunks])
        vectors = np.array(embed_result.vectors)

        kept_indices: list[int] = []
        kept_vectors: list[np.ndarray] = []

        for i, vector in enumerate(vectors):
            is_duplicate_of_kept = any(
                self._cosine_similarity(vector, kept_vector) >= similarity_threshold
                for kept_vector in kept_vectors
            )
            if not is_duplicate_of_kept:
                kept_indices.append(i)
                kept_vectors.append(vector)

        removed_count = len(chunks) - len(kept_indices)
        if removed_count > 0:
            logger.info(
                "Removed near-duplicate chunks during compression",
                extra={"removed_count": removed_count, "remaining_count": len(kept_indices)},
            )

        return [chunks[i] for i in kept_indices]

    @staticmethod
    def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Since embeddings from EmbeddingService are already normalized
        (Module 9's `normalize_embeddings=True`), this simplifies to a
        plain dot product — but we compute it generally here in case that
        assumption ever changes.
        """
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))