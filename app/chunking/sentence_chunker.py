"""Sentence-aware chunking strategy.

Groups whole sentences together until the token budget is reached, never
splitting a sentence mid-way. Better suited than FixedSizeChunker for
narrative/prose text where sentence integrity matters for readability and
embedding quality.
"""

import re

from app.chunking.base import Chunk, ChunkingStrategy

# A deliberately simple sentence-boundary regex: split after '.', '!', or
# '?' followed by whitespace and a capital letter or end of string. This
# is NOT linguistically perfect (e.g., "Dr. Smith arrived." may split
# incorrectly after "Dr."), but is dependency-light and good enough for
# chunking purposes, where occasional imperfect boundaries have minimal
# impact. A future extension could swap in nltk/spacy for higher accuracy.
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class SentenceChunker(ChunkingStrategy):
    """Splits text into chunks made of whole sentences, respecting a token budget."""

    def chunk(self, text: str) -> list[Chunk]:
        """Group whole sentences into chunks up to the configured token budget.

        Args:
            text: the full text to split.

        Returns:
            A list of Chunk objects, each containing one or more complete
            sentences, sized to fit within `chunk_size_tokens`.
        """
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = self.token_counter.count(sentence)

            would_exceed_budget = (
                current_token_count + sentence_tokens > self.chunk_size_tokens
                and current_sentences
            )
            if would_exceed_budget:
                chunks.append(self._build_chunk(current_sentences, text))
                current_sentences = self._carry_over_for_overlap(current_sentences)
                current_token_count = sum(
                    self.token_counter.count(s) for s in current_sentences
                )

            current_sentences.append(sentence)
            current_token_count += sentence_tokens

        if current_sentences:
            chunks.append(self._build_chunk(current_sentences, text))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using a simple regex heuristic."""
        raw_sentences = _SENTENCE_BOUNDARY_PATTERN.split(text.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

    def _carry_over_for_overlap(self, sentences: list[str]) -> list[str]:
        """Choose trailing sentences from the previous chunk to repeat at
        the start of the next chunk, up to the configured overlap budget."""
        carried: list[str] = []
        carried_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.token_counter.count(sentence)
            if carried_tokens + sentence_tokens > self.chunk_overlap_tokens and carried:
                break
            carried.insert(0, sentence)
            carried_tokens += sentence_tokens

        return carried

    def _build_chunk(self, sentences: list[str], original_text: str) -> Chunk:
        """Join sentences into a single Chunk, locating its position in the source text."""
        chunk_text = " ".join(sentences)
        start_index = original_text.find(sentences[0]) if sentences else 0
        start_index = max(start_index, 0)
        end_index = start_index + len(chunk_text)

        return Chunk(
            text=chunk_text,
            start_index=start_index,
            end_index=end_index,
            token_count=self.token_counter.count(chunk_text),
            metadata={"sentence_count": len(sentences)},
        )