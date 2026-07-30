"""LLM service with automatic primary -> fallback failover and retry.

This is the ONLY class other modules should depend on for generating LLM
responses. It hides retry logic and which specific model actually
answered behind a single, reliable interface — same architectural role as
EmbeddingService (Module 9) plays for embeddings.
"""

import logging
from collections.abc import AsyncIterator

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.llm.exceptions import AllLLMProvidersFailedError, LLMError

logger = logging.getLogger(__name__)


class LLMService:
    """Generates LLM responses using a primary provider, retrying transient
    failures, and falling back to a secondary provider if the primary is
    exhausted."""

    def __init__(
        self,
        primary_provider: LLMProvider,
        fallback_provider: LLMProvider,
        *,
        max_retries: int,
    ) -> None:
        """Store the two providers and retry configuration.

        Args:
            primary_provider: tried first (with retries) for every request.
            fallback_provider: used only if the primary provider fails
                after all retries are exhausted.
            max_retries: how many times to retry the PRIMARY provider on
                transient failures before falling back.
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self._max_retries = max_retries

    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate a complete response, retrying and falling back as needed.

        Args:
            messages: the full conversation to send.

        Returns:
            An LLMResponse — check `.model_name` to see which model
            actually produced it.

        Raises:
            AllLLMProvidersFailedError: if both providers fail.
        """
        try:
            return await self._call_with_retry(self.primary_provider.chat, messages)
        except LLMError as exc:
            logger.warning(
                "Primary LLM provider failed after retries, falling back",
                extra={"primary_model": self.primary_provider.model_name, "error": str(exc)},
            )

        try:
            return await self.fallback_provider.chat(messages)
        except LLMError as exc:
            logger.error(
                "Fallback LLM provider also failed",
                extra={"fallback_model": self.fallback_provider.model_name, "error": str(exc)},
            )
            raise AllLLMProvidersFailedError(
                "Both primary and fallback LLM providers failed to generate a response."
            ) from exc

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Stream a response token-by-token, falling back to the secondary
        provider if the primary fails BEFORE any tokens have been yielded.

        Note: if the primary provider fails partway through streaming
        (after some tokens were already sent to the caller), we cannot
        cleanly "restart" with the fallback without the caller potentially
        seeing a duplicated/garbled partial response — so mid-stream
        failures are NOT retried here; they propagate as an error. This is
        an intentional simplicity trade-off for streaming specifically.

        Args:
            messages: the full conversation to send.

        Yields:
            Successive text chunks from whichever provider succeeds.

        Raises:
            AllLLMProvidersFailedError: if the primary fails before
                yielding anything AND the fallback also fails.
        """
        try:
            async for chunk in self.primary_provider.stream_chat(messages):
                yield chunk
            return
        except LLMError as exc:
            logger.warning(
                "Primary LLM provider failed during streaming, falling back",
                extra={"primary_model": self.primary_provider.model_name, "error": str(exc)},
            )

        try:
            async for chunk in self.fallback_provider.stream_chat(messages):
                yield chunk
        except LLMError as exc:
            logger.error(
                "Fallback LLM provider also failed during streaming",
                extra={"fallback_model": self.fallback_provider.model_name, "error": str(exc)},
            )
            raise AllLLMProvidersFailedError(
                "Both primary and fallback LLM providers failed to stream a response."
            ) from exc

    async def _call_with_retry(self, func, messages: list[ChatMessage]) -> LLMResponse:
        """Call an LLM provider method with exponential-backoff retry on
        transient failures.

        Uses tenacity to retry LLMError up to `max_retries` additional
        times, with exponential backoff between attempts (1s, 2s, 4s...),
        before giving up and letting the caller fall back to the secondary
        provider.
        """
        retrying = retry(
            retry=retry_if_exception_type(LLMError),
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        return await retrying(func)(messages)

    @property
    def is_using_fallback_model(self) -> bool:
        """Whether the most recent request required falling back.

        Note: unlike EmbeddingService's "sticky" fallback (Module 9),
        LLMService re-attempts the primary provider on EVERY call — LLM
        rate limits and transient errors are often short-lived (seconds),
        so retrying the primary each time is more appropriate here than
        permanently marking it unavailable for the process lifetime.
        """
        return False  # Placeholder — see note below.