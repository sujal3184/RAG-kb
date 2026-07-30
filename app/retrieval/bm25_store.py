"""BM25 keyword search.

Provides classic term-frequency-based keyword search, complementing dense
vector search (which excels at meaning but can miss exact terms like
product codes or rare technical vocabulary).

The BM25 index is built ON DEMAND from a provided list of chunks, rather
than being persisted — see Module design notes for the reasoning. This
keeps BM25Store simple and stateless from the caller's perspective: hand
it the chunks to search over, get results back.
"""

import logging
import re
import uuid
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.retrieval.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

# Simple whitespace/punctuation tokenizer. BM25 doesn't need linguistic
# sophistication — just consistent, reasonable word splitting.
_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass
class BM25Document:
    """A single document (chunk) to be indexed/searched by BM25.

    Attributes:
        chunk_id: identifier of this chunk.
        document_id: which Document this chunk came from.
        chunk_index: this chunk's position within its document.
        text: the chunk's text content to be searched.
    """

    chunk_id: str
    document_id: uuid.UUID
    chunk_index: int
    text: str


@dataclass
class BM25Result:
    """A single BM25 search result.

    Attributes:
        chunk_id: identifier of the matched chunk.
        document_id: which Document this chunk came from.
        chunk_index: this chunk's position within its document.
        text: the chunk's text content.
        score: raw BM25 score (unbounded, higher = more relevant).
    """

    chunk_id: str
    document_id: uuid.UUID
    chunk_index: int
    text: str
    score: float


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens for BM25 indexing/querying."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Store:
    """Builds and searches an in-memory BM25 index over a set of chunks."""

    def search(
        self, *, documents: list[BM25Document], query: str, top_k: int
    ) -> list[BM25Result]:
        """Build a BM25 index over the given documents and search it.

        Args:
            documents: the chunks to search over (typically all chunks in
                a specific knowledge base).
            query: the search query text.
            top_k: maximum number of results to return.

        Returns:
            A list of BM25Result, ordered by relevance (best first).
            Returns an empty list if there are no documents to search.

        Raises:
            VectorStoreError: if BM25 indexing/search fails unexpectedly.
        """
        if not documents:
            return []

        try:
            tokenized_corpus = [_tokenize(doc.text) for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = _tokenize(query)
            scores = bm25.get_scores(tokenized_query)

            scored_documents = sorted(
                zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True
            )

            results = [
                BM25Result(
                    chunk_id=doc.chunk_id,
                    document_id=doc.document_id,
                    chunk_index=doc.chunk_index,
                    text=doc.text,
                    score=float(score),
                )
                for doc, score in scored_documents[:top_k]
                if score > 0  # exclude zero-relevance results entirely
            ]
            return results
        except Exception as exc:
            raise VectorStoreError(f"BM25 search failed: {exc}") from exc