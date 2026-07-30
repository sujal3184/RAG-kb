"""Context compression pipeline.

Runs deduplication, then enforces a hard token budget, producing the
final, lean chunk list that Module 14 (Prompt Builder) will assemble into
the LLM's context.
"""

import logging

from app.chunking.base import TokenCounter
from app.retrieval.base import CompressedChunk, RankedChunk
from app.retrieval.deduplication import Deduplicator

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Compresses a ranked chunk list into a lean, deduplicated,
    budget-constrained final context."""

    def __init__(self, deduplicator: Deduplicator, token_counter: TokenCounter) -> None:
        """Store the dependencies this compressor uses.

        Args:
            deduplicator: removes near-duplicate chunks.
            token_counter: measures each chunk's token count for budget
                enforcement (reusing the same tokenizer wrapper from
                Module 8, so "token" means the same thing everywhere).
        """
        self.deduplicator = deduplicator
        self.token_counter = token_counter

    async def compress(
        self,
        chunks: list[RankedChunk],
        *,
        similarity_threshold: float,
        max_context_tokens: int,
    ) -> list[CompressedChunk]:
        """Deduplicate and token-budget-limit a ranked chunk list.

        Args:
            chunks: reranked chunks (Module 12 output), best first.
            similarity_threshold: cosine similarity threshold for
                deduplication (see Deduplicator).
            max_context_tokens: hard cap on total tokens across all
                returned chunks combined.

        Returns:
            A list of CompressedChunk, deduplicated and trimmed to fit
            within max_context_tokens — never splitting a chunk mid-text.
        """
        if not chunks:
            return []

        deduplicated = await self.deduplicator.deduplicate(
            chunks, similarity_threshold=similarity_threshold
        )

        result: list[CompressedChunk] = []
        total_tokens = 0

        for chunk in deduplicated:
            token_count = self.token_counter.count(chunk.text)

            if total_tokens + token_count > max_context_tokens:
                logger.info(
                    "Stopping context assembly — token budget reached",
                    extra={"included_chunks": len(result), "total_tokens": total_tokens},
                )
                break

            result.append(
                CompressedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    rerank_score=chunk.rerank_score,
                    token_count=token_count,
                )
            )
            total_tokens += token_count

        return result