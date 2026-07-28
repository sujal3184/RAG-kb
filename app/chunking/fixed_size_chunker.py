"""Fixed-size chunking strategy.

Splits text into fixed-size token windows with configurable overlap,
without regard for sentence or paragraph boundaries. Simple, predictable,
and a reasonable choice for dense/structured text (e.g. CSV-derived text)
where "sentences" aren't a meaningful concept anyway.
"""

from app.chunking.base import Chunk, ChunkingStrategy


class FixedSizeChunker(ChunkingStrategy):
    """Splits text into fixed-size, overlapping token windows."""

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into fixed-size token windows with overlap.

        Args:
            text: the full text to split.

        Returns:
            A list of Chunk objects, each up to `chunk_size_tokens` long,
            with `chunk_overlap_tokens` of repeated content between
            consecutive chunks.
        """
        if not text.strip():
            return []

        tokens = self.token_counter.encode(text)
        chunks: list[Chunk] = []

        step = self.chunk_size_tokens - self.chunk_overlap_tokens
        position = 0

        while position < len(tokens):
            window_tokens = tokens[position : position + self.chunk_size_tokens]
            chunk_text = self.token_counter.decode(window_tokens)

            start_index = text.find(chunk_text[:50]) if chunk_text else -1
            # Fallback for cases where decoding introduces minor spacing
            # differences that make an exact substring match fail.
            if start_index == -1:
                start_index = 0
            end_index = start_index + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=len(window_tokens),
                )
            )

            position += step

        return chunks