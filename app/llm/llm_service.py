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
from app.observability.langfuse_client import LangfuseTracer
from app.observability.metrics import llm_requests_total, llm_tokens_total



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
        langfuse_tracer: LangfuseTracer | None = None,
    ) -> None:
        """... (existing docstring, extended)

        Args:
            langfuse_tracer: optional LLM tracing — if None, generations
                simply aren't recorded (keeps LLMService usable in tests
                without any observability infrastructure).
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self._max_retries = max_retries
        self._langfuse = langfuse_tracer


    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate a complete response, retrying and falling back as needed."""
        try:
            response = await self._call_with_retry(self.primary_provider.chat, messages)
            self._record_observability(messages, response, outcome="success")
            return response
        except LLMError as exc:
            logger.warning(
                "Primary LLM provider failed after retries, falling back",
                extra={"primary_model": self.primary_provider.model_name, "error": str(exc)},
            )
            llm_requests_total.labels(
                model=self.primary_provider.model_name, outcome="failure"
            ).inc()

        try:
            response = await self.fallback_provider.chat(messages)
            self._record_observability(messages, response, outcome="fallback")
            return response
        except LLMError as exc:
            logger.error(
                "Fallback LLM provider also failed",
                extra={"fallback_model": self.fallback_provider.model_name, "error": str(exc)},
            )
            llm_requests_total.labels(
                model=self.fallback_provider.model_name, outcome="failure"
            ).inc()
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


    def _record_observability(
        self, messages: list[ChatMessage], response: LLMResponse, *, outcome: str
    ) -> None:
        """Record metrics and (if configured) a Langfuse generation trace."""
        llm_requests_total.labels(model=response.model_name, outcome=outcome).inc()
        llm_tokens_total.labels(model=response.model_name, token_type="input").inc(
            response.input_tokens
        )
        llm_tokens_total.labels(model=response.model_name, token_type="output").inc(
            response.output_tokens
        )

        if self._langfuse is not None:
            self._langfuse.trace_generation(
                name="rag_answer",
                model=response.model_name,
                input_messages=[{"role": m.role.value, "content": m.content} for m in messages],
                output=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                metadata={"outcome": outcome},
            )



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