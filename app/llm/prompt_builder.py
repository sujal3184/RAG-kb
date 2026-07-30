"""Prompt builder — assembles compressed chunks + user query (+ optional
conversation history) into a final, structured prompt ready for an LLM.
"""

import logging

from app.chunking.base import TokenCounter
from app.llm.base import ChatMessage, MessageRole, PromptResult, SourceReference
from app.llm.templates import (
    CONTEXT_SECTION_HEADER,
    NO_CONTEXT_FALLBACK_NOTICE,
    RAG_SYSTEM_PROMPT,
    SOURCE_BLOCK_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)
from app.retrieval.base import CompressedChunk

logger = logging.getLogger(__name__)


class ChunkWithSource:
    """Pairs a CompressedChunk with the filename of its source document.

    PromptBuilder needs the filename for citation display, but
    CompressedChunk (Module 13) only carries document_id — the caller
    (Module 17's eventual pipeline wiring) is responsible for looking up
    filenames via DocumentRepository and providing them here.
    """

    def __init__(self, chunk: CompressedChunk, source_filename: str) -> None:
        self.chunk = chunk
        self.source_filename = source_filename


class PromptBuilder:
    """Builds structured, citation-ready prompts from compressed context chunks."""

    def __init__(self, token_counter: TokenCounter) -> None:
        """Store the token counter used for context-size bookkeeping.

        Args:
            token_counter: shared tokenizer wrapper (Module 8), used here
                purely for reporting total context token usage.
        """
        self.token_counter = token_counter

    def build(
        self,
        *,
        query: str,
        chunks: list[ChunkWithSource],
        conversation_history: list[ChatMessage] | None = None,
    ) -> PromptResult:
        """Assemble a final prompt from a query, source chunks, and optional history.

        Args:
            query: the user's current question.
            chunks: compressed, source-labeled chunks (Module 13's output,
                paired with their document's filename).
            conversation_history: prior turns of the conversation, if any
                (populated starting Module 16) — inserted between the
                system prompt and the current user message.

        Returns:
            A PromptResult containing the full message list ready to send
            to an LLM, plus source metadata for later citation mapping.
        """
        sources = self._build_source_references(chunks)
        context_section, total_context_tokens = self._build_context_section(chunks, sources)

        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=RAG_SYSTEM_PROMPT)
        ]

        if conversation_history:
            messages.extend(conversation_history)

        user_content = USER_PROMPT_TEMPLATE.format(context_section=context_section, query=query)
        messages.append(ChatMessage(role=MessageRole.USER, content=user_content))

        logger.info(
            "Prompt built",
            extra={
                "source_count": len(sources),
                "context_tokens": total_context_tokens,
                "history_messages": len(conversation_history) if conversation_history else 0,
            },
        )

        return PromptResult(
            messages=messages, sources=sources, total_context_tokens=total_context_tokens
        )

    def _build_source_references(self, chunks: list[ChunkWithSource]) -> list[SourceReference]:
        """Assign a citation marker ("Source 1", "Source 2"...) to each chunk."""
        return [
            SourceReference(
                marker=f"Source {i + 1}",
                document_id=str(item.chunk.document_id),
                chunk_id=item.chunk.chunk_id,
                source_filename=item.source_filename,
            )
            for i, item in enumerate(chunks)
        ]

    def _build_context_section(
        self, chunks: list[ChunkWithSource], sources: list[SourceReference]
    ) -> tuple[str, int]:
        """Format all chunks into the labeled "Context:" section of the prompt.

        Returns:
            A tuple of (formatted context section text, total token count
            of just the source content — useful for observability).
        """
        if not chunks:
            return CONTEXT_SECTION_HEADER + NO_CONTEXT_FALLBACK_NOTICE, 0

        blocks = []
        total_tokens = 0
        for item, source in zip(chunks, sources, strict=True):
            block = SOURCE_BLOCK_TEMPLATE.format(
                marker=source.marker, filename=source.source_filename, text=item.chunk.text
            )
            blocks.append(block)
            total_tokens += item.chunk.token_count

        context_section = CONTEXT_SECTION_HEADER + "\n".join(blocks)
        return context_section, total_tokens