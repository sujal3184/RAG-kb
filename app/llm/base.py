"""Shared types for LLM prompt construction and (later) LLM responses.

Kept separate from any specific LLM provider (Module 15 will add Groq
integration) so this module stays provider-agnostic — the same
PromptResult shape could be sent to Groq, OpenAI, Anthropic, or any other
chat-completions-style API.
"""

from dataclasses import dataclass
from enum import StrEnum

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class MessageRole(StrEnum):
    """Standard chat message roles, matching the OpenAI-compatible format
    used by Groq (Module 15) and most other LLM providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """A single message in a chat-style LLM conversation.

    Attributes:
        role: who "said" this message (system instructions, the user, or
            a prior assistant response).
        content: the message text.
    """

    role: MessageRole
    content: str


@dataclass
class SourceReference:
    """Metadata about one source chunk included in a built prompt.

    Kept alongside the prompt itself so that once the LLM responds
    (Module 15) and its answer references "[Source 2]", we can map that
    citation marker back to real document/chunk information to show the
    user (e.g. "this came from quarterly_report.pdf").

    Attributes:
        marker: the citation label used in the prompt (e.g. "Source 1").
        document_id: which Document this chunk came from.
        chunk_id: identifier of the specific chunk.
        source_filename: the original filename, shown to the LLM and
            available for later display to the user.
    """

    marker: str
    document_id: str
    chunk_id: str
    source_filename: str


@dataclass
class PromptResult:
    """The final, ready-to-send prompt, plus bookkeeping metadata.

    Attributes:
        messages: the full list of chat messages to send to the LLM,
            in order (system message first, then conversation history,
            then the current user message).
        sources: metadata for every source chunk included, for later
            citation mapping once the LLM responds.
        total_context_tokens: total tokens consumed by the source context
            portion of the prompt (useful for observability/debugging).
    """

    messages: list[ChatMessage]
    sources: list[SourceReference]
    total_context_tokens: int



@dataclass
class LLMResponse:
    """A complete (non-streaming) response from an LLM.

    Attributes:
        content: the generated answer text.
        model_name: which model actually produced this response — useful
            for logging and for showing the user "answered by X" if
            desired.
        input_tokens: how many tokens the prompt consumed (as reported by
            the provider), for cost/usage tracking.
        output_tokens: how many tokens the response consumed.
    """

    content: str
    model_name: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    """Abstract base class for a single LLM backend."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The identifier of the model this provider uses."""
        raise NotImplementedError

    @abstractmethod
    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Send messages to the LLM and get back a complete response.

        Args:
            messages: the full conversation, including system instructions.

        Returns:
            An LLMResponse with the generated content.

        Raises:
            LLMError: if the request fails.
        """
        raise NotImplementedError

    @abstractmethod
    def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Send messages to the LLM and stream back the response token-by-token.

        Args:
            messages: the full conversation, including system instructions.

        Yields:
            Successive text chunks as they're generated.

        Raises:
            LLMError: if the request fails.
        """
        raise NotImplementedError