"""Vector store interface and shared data types.

A `VectorStore` handles storing, searching, and deleting vector
embeddings, keyed by knowledge base. Callers never interact with Qdrant's
client library directly — only through this interface — so swapping
vector database backends later touches only one new implementation file.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorPoint:
    """A single vector to be stored, with its associated metadata.

    Attributes:
        chunk_id: a stable identifier for the chunk this vector represents
            (used to derive a deterministic point ID — see design notes).
        document_id: which Document this chunk came from.
        knowledge_base_id: which Knowledge Base this belongs to (also
            determines which Qdrant collection is used).
        chunk_index: this chunk's position within its document (0-based),
            used to preserve ordering for later context reconstruction.
        text: the chunk's actual text content, stored in the payload so
            search results are immediately usable without a second lookup.
        vector: the embedding vector itself.
        extra_metadata: any additional payload fields (e.g. page number).
    """

    chunk_id: str
    document_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    chunk_index: int
    text: str
    vector: list[float]
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result returned from the vector store.

    Attributes:
        chunk_id: identifier of the matched chunk.
        document_id: which Document this chunk came from.
        chunk_index: this chunk's position within its document.
        text: the chunk's text content.
        score: similarity score (higher = more similar, for cosine
            similarity this ranges roughly from -1 to 1, though in
            practice typically 0 to 1 for normalized text embeddings).
        extra_metadata: any additional payload fields stored with this point.
    """

    chunk_id: str
    document_id: uuid.UUID
    chunk_index: int
    text: str
    score: float
    extra_metadata: dict[str, Any] = field(default_factory=dict)


# Add this new dataclass alongside VectorPoint and SearchResult:

@dataclass
class RetrievedChunk:
    """A chunk retrieved via hybrid search, with fused ranking information.

    Attributes:
        chunk_id: identifier of the retrieved chunk.
        document_id: which Document this chunk came from.
        chunk_index: this chunk's position within its document.
        text: the chunk's text content.
        fused_score: the final combined score after Reciprocal Rank Fusion
            — higher means more relevant overall.
        source_methods: which retrieval method(s) found this chunk
            ("dense", "bm25", or both) — useful for debugging/observability.
        extra_metadata: any additional payload fields.
    """

    chunk_id: str
    document_id: uuid.UUID
    chunk_index: int
    text: str
    fused_score: float
    source_methods: list[str] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)


# Add this new dataclass alongside RetrievedChunk:

@dataclass
class RankedChunk:
    """A chunk after reranking, with the reranker's relevance score.

    Attributes:
        chunk_id: identifier of the chunk.
        document_id: which Document this chunk came from.
        chunk_index: this chunk's position within its document.
        text: the chunk's text content.
        rerank_score: the cross-encoder's relevance score for this
            specific query + chunk pair (higher = more relevant). Unlike
            fused_score from hybrid retrieval, this score is NOT
            comparable across different queries — it's only meaningful
            for ranking candidates within the SAME query.
        source_methods: preserved from the original RetrievedChunk, for
            observability (which retrieval method(s) originally surfaced
            this chunk before reranking).
    """

    chunk_id: str
    document_id: uuid.UUID
    chunk_index: int
    text: str
    rerank_score: float
    source_methods: list[str] = field(default_factory=list)
    


class VectorStore(ABC):
    """Abstract base class for storing and searching vector embeddings."""

    @abstractmethod
    async def upsert(
        self, *, knowledge_base_id: uuid.UUID, points: list[VectorPoint], vector_dimension: int
    ) -> None:
        """Insert or update vectors within a knowledge base's collection.

        Args:
            knowledge_base_id: which knowledge base these vectors belong to.
            points: the vectors and metadata to store.
            vector_dimension: dimension of the vectors being stored — used
                to create the collection correctly if it doesn't exist yet.
        """
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        query_vector: list[float],
        top_k: int,
        document_id: uuid.UUID | None = None,
    ) -> list[SearchResult]:
        """Find the most similar vectors within a knowledge base.

        Args:
            knowledge_base_id: which knowledge base to search within.
            query_vector: the embedding to search for similar vectors against.
            top_k: maximum number of results to return.
            document_id: if provided, restrict results to this document only.

        Returns:
            A list of SearchResult, ordered by similarity (best first).
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_by_document(
        self, *, knowledge_base_id: uuid.UUID, document_id: uuid.UUID
    ) -> None:
        """Delete all vectors belonging to a specific document.

        Called when a Document is deleted (Module 6), so its vectors
        don't linger in Qdrant as orphaned data.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_collection(self, *, knowledge_base_id: uuid.UUID) -> None:
        """Delete an entire knowledge base's collection.

        Called when a Knowledge Base is deleted (Module 5), removing all
        of its vectors in one operation.
        """
        raise NotImplementedError



