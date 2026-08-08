"""Qdrant implementation of the VectorStore interface."""

import logging
import uuid

from qdrant_client import AsyncQdrantClient, models

from app.retrieval.base import SearchResult, VectorPoint, VectorStore
from app.retrieval.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Stores and searches vector embeddings using Qdrant, one collection
    per knowledge base."""

    def __init__(self, host: str, port: int, *, collection_prefix: str, api_key: str | None = None) -> None:
        """Set up the Qdrant client connection.

        Args:
            host: Qdrant server hostname.
            port: Qdrant server REST port.
            collection_prefix: prefix used when deriving collection names
                from knowledge base ids (e.g. "kb_" -> "kb_<uuid>").
            api_key: required for Qdrant Cloud (auth-protected); leave None
                for local Docker Qdrant (no auth). When set, connections
                automatically use HTTPS, since Qdrant Cloud requires it.
        """
        self._client = AsyncQdrantClient(host=host, port=port, api_key=api_key, https=bool(api_key))
        self._collection_prefix = collection_prefix

        
    def _collection_name(self, knowledge_base_id: uuid.UUID) -> str:
        """Derive a deterministic Qdrant collection name for a knowledge base."""
        return f"{self._collection_prefix}{knowledge_base_id}"

    def _point_id(self, document_id: uuid.UUID, chunk_index: int) -> str:
        """Derive a deterministic point ID from document + chunk position.

        Using UUID5 (name-based, deterministic) means re-embedding the
        same document produces the SAME point IDs, so upserting naturally
        overwrites old vectors instead of creating duplicates.
        """
        name = f"{document_id}:{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

    async def _ensure_collection_exists(
        self, collection_name: str, vector_dimension: int
    ) -> None:
        """Create the collection if it doesn't already exist.

        Args:
            collection_name: the Qdrant collection to check/create.
            vector_dimension: vector size to configure the collection with,
                if it needs to be created.
        """
        exists = await self._client.collection_exists(collection_name)
        if not exists:
            logger.info(
                "Creating Qdrant collection",
                extra={"collection": collection_name, "dimension": vector_dimension},
            )
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension, distance=models.Distance.COSINE
                ),
            )

    async def upsert(
        self, *, knowledge_base_id: uuid.UUID, points: list[VectorPoint], vector_dimension: int
    ) -> None:
        """Insert or update vectors within a knowledge base's collection.

        Raises:
            VectorStoreError: if the Qdrant operation fails.
        """
        if not points:
            return

        collection_name = self._collection_name(knowledge_base_id)

        try:
            await self._ensure_collection_exists(collection_name, vector_dimension)

            qdrant_points = [
                models.PointStruct(
                    id=self._point_id(point.document_id, point.chunk_index),
                    vector=point.vector,
                    payload={
                        "chunk_id": point.chunk_id,
                        "document_id": str(point.document_id),
                        "knowledge_base_id": str(point.knowledge_base_id),
                        "chunk_index": point.chunk_index,
                        "text": point.text,
                        **point.extra_metadata,
                    },
                )
                for point in points
            ]

            await self._client.upsert(collection_name=collection_name, points=qdrant_points)
            logger.info(
                "Upserted vectors into Qdrant",
                extra={"collection": collection_name, "count": len(points)},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert vectors into Qdrant: {exc}") from exc

    async def search(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        query_vector: list[float],
        top_k: int,
        document_id: uuid.UUID | None = None,
    ) -> list[SearchResult]:
        """Find the most similar vectors within a knowledge base.

        Raises:
            VectorStoreError: if the Qdrant operation fails.
        """
        collection_name = self._collection_name(knowledge_base_id)

        try:
            exists = await self._client.collection_exists(collection_name)
            if not exists:
                # No documents have been embedded into this KB yet —
                # an empty result set is the correct answer, not an error.
                return []

            query_filter = None
            if document_id is not None:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=str(document_id))
                        )
                    ]
                )

            response = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            return [self._to_search_result(point) for point in response.points]
        except Exception as exc:
            raise VectorStoreError(f"Failed to search Qdrant: {exc}") from exc

    async def delete_by_document(
        self, *, knowledge_base_id: uuid.UUID, document_id: uuid.UUID
    ) -> None:
        """Delete all vectors belonging to a specific document.

        Raises:
            VectorStoreError: if the Qdrant operation fails.
        """
        collection_name = self._collection_name(knowledge_base_id)

        try:
            exists = await self._client.collection_exists(collection_name)
            if not exists:
                return  # Nothing to delete — treat as a no-op, not an error.

            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=str(document_id)),
                            )
                        ]
                    )
                ),
            )
            logger.info(
                "Deleted document vectors from Qdrant",
                extra={"collection": collection_name, "document_id": str(document_id)},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document vectors from Qdrant: {exc}") from exc

    async def delete_collection(self, *, knowledge_base_id: uuid.UUID) -> None:
        """Delete an entire knowledge base's collection.

        Raises:
            VectorStoreError: if the Qdrant operation fails.
        """
        collection_name = self._collection_name(knowledge_base_id)

        try:
            exists = await self._client.collection_exists(collection_name)
            if exists:
                await self._client.delete_collection(collection_name)
                logger.info("Deleted Qdrant collection", extra={"collection": collection_name})
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete Qdrant collection: {exc}") from exc

    @staticmethod
    def _to_search_result(point: models.ScoredPoint) -> SearchResult:
        """Convert a Qdrant ScoredPoint into our own SearchResult type.

        Keeping this conversion in one place means the rest of the app
        never sees Qdrant's own result types directly.
        """
        payload = point.payload or {}
        known_keys = {"chunk_id", "document_id", "knowledge_base_id", "chunk_index", "text"}
        extra_metadata = {k: v for k, v in payload.items() if k not in known_keys}

        return SearchResult(
            chunk_id=payload.get("chunk_id", str(point.id)),
            document_id=uuid.UUID(payload["document_id"]),
            chunk_index=payload.get("chunk_index", 0),
            text=payload.get("text", ""),
            score=point.score,
            extra_metadata=extra_metadata,
        )