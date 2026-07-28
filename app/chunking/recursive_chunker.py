"""Recursive, structure-aware chunking strategy.

Tries to split text along the most "natural" boundary first, falling back
to progressively smaller units only when needed:

    1. Paragraph breaks (double newlines)
    2. Sentence boundaries (reusing SentenceChunker's splitter)
    3. Fixed-size token windows (reusing FixedSizeChunker, as a last resort)

This mirrors the widely-used "recursive character text splitter" pattern:
it respects document structure when present, but still guarantees every
chunk fits the token budget no matter how unstructured the input text is.
"""

from app.chunking.base import Chunk, ChunkingStrategy
from app.chunking.fixed_size_chunker import FixedSizeChunker
from app.chunking.sentence_chunker import SentenceChunker


class RecursiveChunker(ChunkingStrategy):
    """Splits text by paragraph, then sentence, then fixed-size, as needed."""

    def chunk(self, text: str) -> list[Chunk]:
        """Split text using the most natural boundary that fits the token budget.

        Args:
            text: the full text to split.

        Returns:
            A list of Chunk objects, preferring paragraph-aligned chunks,
            falling back to sentence- or token-based splitting as needed.
        """
        if not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        chunks: list[Chunk] = []
        current_paragraphs: list[str] = []
        current_token_count = 0

        for paragraph in paragraphs:
            paragraph_tokens = self.token_counter.count(paragraph)

            if paragraph_tokens > self.chunk_size_tokens:
                # This single paragraph is too big on its own — flush
                # whatever we were building, then break the paragraph
                # down further using sentence-level chunking.
                if current_paragraphs:
                    chunks.append(self._build_chunk(current_paragraphs, text))
                    current_paragraphs, current_token_count = [], 0

                chunks.extend(self._split_oversized_paragraph(paragraph, text))
                continue

            would_exceed_budget = (
                current_token_count + paragraph_tokens > self.chunk_size_tokens
                and current_paragraphs
            )
            if would_exceed_budget:
                chunks.append(self._build_chunk(current_paragraphs, text))
                current_paragraphs, current_token_count = [], 0

            current_paragraphs.append(paragraph)
            current_token_count += paragraph_tokens

        if current_paragraphs:
            chunks.append(self._build_chunk(current_paragraphs, text))

        return chunks

    def _split_oversized_paragraph(self, paragraph: str, original_text: str) -> list[Chunk]:
        """Break a too-large paragraph down using sentence-aware chunking,
        falling back to fixed-size chunking if it's STILL too large
        (e.g. one enormous run-on sentence with no punctuation)."""
        sentence_chunker = SentenceChunker(
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
            token_counter=self.token_counter,
        )
        sentence_chunks = sentence_chunker.chunk(paragraph)

        final_chunks: list[Chunk] = []
        fixed_chunker = FixedSizeChunker(
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
            token_counter=self.token_counter,
        )
        for sentence_chunk in sentence_chunks:
            if sentence_chunk.token_count > self.chunk_size_tokens:
                final_chunks.extend(fixed_chunker.chunk(sentence_chunk.text))
            else:
                final_chunks.append(sentence_chunk)

        return final_chunks

    def _build_chunk(self, paragraphs: list[str], original_text: str) -> Chunk:
        """Join paragraphs into a single Chunk, locating its position in the source text."""
        chunk_text = "\n\n".join(paragraphs)
        start_index = original_text.find(paragraphs[0]) if paragraphs else 0
        start_index = max(start_index, 0)
        end_index = start_index + len(chunk_text)

        return Chunk(
            text=chunk_text,
            start_index=start_index,
            end_index=end_index,
            token_count=self.token_counter.count(chunk_text),
            metadata={"paragraph_count": len(paragraphs)},
        )