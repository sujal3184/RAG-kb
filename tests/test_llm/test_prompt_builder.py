"""Tests for PromptBuilder — pure computation, no external services needed."""

import uuid

from app.chunking.base import TokenCounter
from app.llm.base import ChatMessage, MessageRole
from app.llm.prompt_builder import ChunkWithSource, PromptBuilder
from app.retrieval.base import CompressedChunk


def _make_chunk_with_source(
    text: str, filename: str, chunk_id: str | None = None
) -> ChunkWithSource:
    chunk = CompressedChunk(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=text,
        rerank_score=0.9,
        token_count=len(text.split()),
    )
    return ChunkWithSource(chunk, filename)


def _builder() -> PromptBuilder:
    return PromptBuilder(TokenCounter("cl100k_base"))


def test_includes_system_prompt_first() -> None:
    result = _builder().build(query="What is RAG?", chunks=[])

    assert result.messages[0].role == MessageRole.SYSTEM
    assert "ONLY the information provided" in result.messages[0].content


def test_labels_sources_sequentially() -> None:
    chunks = [
        _make_chunk_with_source("Paris is the capital of France.", "geography.pdf"),
        _make_chunk_with_source("The Eiffel Tower was built in 1889.", "history.docx"),
    ]
    result = _builder().build(query="Tell me about Paris", chunks=chunks)

    user_message = result.messages[-1].content
    assert "[Source 1]" in user_message
    assert "[Source 2]" in user_message
    assert "geography.pdf" in user_message
    assert "history.docx" in user_message


def test_source_references_map_correctly() -> None:
    chunks = [_make_chunk_with_source("Some fact.", "notes.txt", chunk_id="chunk-abc")]
    result = _builder().build(query="A question", chunks=chunks)

    assert len(result.sources) == 1
    assert result.sources[0].marker == "Source 1"
    assert result.sources[0].chunk_id == "chunk-abc"
    assert result.sources[0].source_filename == "notes.txt"


def test_empty_chunks_includes_fallback_notice() -> None:
    result = _builder().build(query="Anything?", chunks=[])

    user_message = result.messages[-1].content
    assert "No relevant documents were found" in user_message
    assert result.total_context_tokens == 0
    assert result.sources == []


def test_query_appears_in_final_message() -> None:
    result = _builder().build(query="What is the capital of France?", chunks=[])

    assert "What is the capital of France?" in result.messages[-1].content


def test_conversation_history_inserted_between_system_and_user_message() -> None:
    history = [
        ChatMessage(role=MessageRole.USER, content="Hi there"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hello! How can I help?"),
    ]
    result = _builder().build(query="Follow-up question", chunks=[], conversation_history=history)

    assert result.messages[0].role == MessageRole.SYSTEM
    assert result.messages[1] == history[0]
    assert result.messages[2] == history[1]
    assert result.messages[3].role == MessageRole.USER
    assert "Follow-up question" in result.messages[3].content


def test_no_conversation_history_means_only_system_and_user_messages() -> None:
    result = _builder().build(query="A question", chunks=[])

    assert len(result.messages) == 2
    assert result.messages[0].role == MessageRole.SYSTEM
    assert result.messages[1].role == MessageRole.USER


def test_total_context_tokens_sums_chunk_token_counts() -> None:
    chunks = [
        _make_chunk_with_source("one two three", "a.txt"),
        _make_chunk_with_source("four five", "b.txt"),
    ]
    result = _builder().build(query="test", chunks=chunks)

    expected_total = sum(item.chunk.token_count for item in chunks)
    assert result.total_context_tokens == expected_total