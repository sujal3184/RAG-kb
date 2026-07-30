"""Hybrid retriever combining dense (vector) and BM25 (keyword) search.

Runs both search methods concurrently, then merges their rankings using
Reciprocal Rank Fusion — giving results that benefit from BOTH semantic
understanding and exact keyword matching.
"""

import asyncio
import logging
import uuid

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.base import RetrievedChunk, VectorStore
from app.retrieval.bm25_store import BM25Document, BM25Store
from app.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves the most relevant chunks for a query using both dense
    vector search and BM25 keyword search, fused via RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedding_service: EmbeddingService,
        *,
        top_k_per_method: int,
        rrf_k: int,
    ) -> None:
        """Store the dependencies and tuning parameters this retriever needs.

        Args:
            vector_store: for dense (embedding-based) search.
            bm25_store: for keyword-based search.
            embedding_service: to embed the query text for dense search.
            top_k_per_method: how many results EACH method contributes
                before fusion (kept higher than the final desired count).
            rrf_k: RRF's damping constant (see fusion.py).
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedding_service = embedding_service
        self.top_k_per_method = top_k_per_method
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        query: str,
        bm25_documents: list[BM25Document],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query, using hybrid search.

        Args:
            knowledge_base_id: which knowledge base to search within.
            query: the user's search query text.
            bm25_documents: the full set of chunks to run BM25 over — this
                must be supplied by the caller (typically loaded from
                Postgres for this knowledge base), since BM25Store builds
                its index on demand rather than maintaining one itself.
            top_k: how many final, fused results to return.

        Returns:
            A list of RetrievedChunk, ordered by fused relevance (best first).
        """
        query_embedding_task = self.embedding_service.embed_text(query)
        bm25_search_task = asyncio.to_thread(
            self.bm25_store.search,
            documents=bm25_documents,
            query=query,
            top_k=self.top_k_per_method,
        )

        query_vector, bm25_results = await asyncio.gather(
            query_embedding_task, bm25_search_task
        )

        dense_results = await self.vector_store.search(
            knowledge_base_id=knowledge_base_id,
            query_vector=query_vector,
            top_k=self.top_k_per_method,
        )

        dense_ranked_ids = [r.chunk_id for r in dense_results]
        bm25_ranked_ids = [r.chunk_id for r in bm25_results]

        fused_scores = reciprocal_rank_fusion(
            [dense_ranked_ids, bm25_ranked_ids], k=self.rrf_k
        )

        chunk_lookup = self._build_chunk_lookup(dense_results, bm25_results)
        source_methods_by_chunk = self._track_source_methods(dense_ranked_ids, bm25_ranked_ids)

        fused_results = [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=chunk_lookup[chunk_id].document_id,
                chunk_index=chunk_lookup[chunk_id].chunk_index,
                text=chunk_lookup[chunk_id].text,
                fused_score=score,
                source_methods=source_methods_by_chunk[chunk_id],
            )
            for chunk_id, score in fused_scores.items()
            if chunk_id in chunk_lookup
        ]

        fused_results.sort(key=lambda r: r.fused_score, reverse=True)
        return fused_results[:top_k]

    @staticmethod
    def _build_chunk_lookup(dense_results: list, bm25_results: list) -> dict:
        """Build a chunk_id -> result lookup, so we can recover full chunk
        data (text, document_id, etc.) after fusion has only scores + ids."""
        lookup = {}
        for result in dense_results:
            lookup[result.chunk_id] = result
        for result in bm25_results:
            lookup.setdefault(result.chunk_id, result)
        return lookup

    @staticmethod
    def _track_source_methods(
        dense_ranked_ids: list[str], bm25_ranked_ids: list[str]
    ) -> dict[str, list[str]]:
        """Track which method(s) surfaced each chunk_id, for observability."""
        source_methods: dict[str, list[str]] = {}
        for chunk_id in dense_ranked_ids:
            source_methods.setdefault(chunk_id, []).append("dense")
        for chunk_id in bm25_ranked_ids:
            source_methods.setdefault(chunk_id, []).append("bm25")
        return source_methods