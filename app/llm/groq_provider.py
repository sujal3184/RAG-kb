"""Groq API implementation of the LLMProvider interface.

Groq exposes an OpenAI-compatible chat completions API, known for very
fast inference. This provider is model-agnostic within Groq — the SAME
class is used for both the primary and fallback provider in LLMService,
just configured with different model names.
"""

import logging
from collections.abc import AsyncIterator

from groq import APIError, APIStatusError, APITimeoutError, AsyncGroq

from app.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.llm.exceptions import LLMError

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """LLM provider backed by Groq's chat completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> None:
        """Configure the Groq client and model parameters.

        Args:
            api_key: Groq API key.
            model: which Groq-hosted model to use (e.g.
                "llama-3.3-70b-versatile").
            temperature: sampling temperature — lower is more deterministic,
                which we generally want for a RAG assistant answering
                factual questions from provided context.
            max_output_tokens: cap on how many tokens the response may contain.
            timeout_seconds: how long to wait for a response before timing out.
        """
        self._client = AsyncGroq(api_key=api_key, timeout=timeout_seconds)
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    @property
    def model_name(self) -> str:
        return self._model

    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Send messages to Groq and get back a complete response.

        Raises:
            LLMError: if the Groq API request fails.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
            )
        except (APIError, APIStatusError, APITimeoutError) as exc:
            raise LLMError(f"Groq request failed for model '{self._model}': {exc}") from exc

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model_name=self._model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Send messages to Groq and stream back the response token-by-token.

        Raises:
            LLMError: if the Groq API request fails.
        """
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (APIError, APIStatusError, APITimeoutError) as exc:
            raise LLMError(f"Groq streaming request failed for model '{self._model}': {exc}") from exc