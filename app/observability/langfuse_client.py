"""Langfuse client for LLM-specific tracing.

Langfuse records the exact prompt sent to the LLM, the response received,
token counts, latency, and model used — surfaced in a purpose-built UI
for debugging RAG answer quality (as opposed to OpenTelemetry, which
focuses on system-level timing).

Fail-open: if Langfuse is disabled or unreachable, all operations become
no-ops rather than raising.
"""

import logging
from functools import lru_cache

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Thin wrapper around the Langfuse SDK, safe to use even when
    Langfuse is disabled or misconfigured."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the Langfuse client if enabled and configured.

        Args:
            settings: app settings providing Langfuse credentials/host.
        """
        self._client = None
        self._enabled = settings.LANGFUSE_ENABLED

        if not self._enabled:
            logger.info("Langfuse tracing is disabled")
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            logger.info("Langfuse tracing configured", extra={"host": settings.LANGFUSE_HOST})
        except Exception as exc:
            logger.warning(
                "Failed to initialize Langfuse, continuing without LLM tracing",
                extra={"error": str(exc)},
            )
            self._enabled = False

    def trace_generation(
        self,
        *,
        name: str,
        model: str,
        input_messages: list[dict],
        output: str,
        input_tokens: int,
        output_tokens: int,
        metadata: dict | None = None,
    ) -> None:
        """Record one LLM generation (prompt + response) in Langfuse.

        Args:
            name: a label for this generation (e.g. "rag_answer").
            model: which model produced the response.
            input_messages: the prompt messages sent to the LLM.
            output: the generated response text.
            input_tokens: prompt token count reported by the provider.
            output_tokens: completion token count reported by the provider.
            metadata: any extra context worth attaching (knowledge base id,
                conversation id, cache status, etc.).
        """
        if not self._enabled or self._client is None:
            return

        try:
            trace = self._client.trace(name=name, metadata=metadata or {})
            trace.generation(
                name=name,
                model=model,
                input=input_messages,
                output=output,
                usage={"input": input_tokens, "output": output_tokens},
            )
        except Exception as exc:
            logger.warning("Failed to record Langfuse trace", extra={"error": str(exc)})

    def flush(self) -> None:
        """Flush any buffered traces — called on application shutdown so
        in-flight traces aren't lost when the process exits."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:
                logger.warning("Failed to flush Langfuse traces", extra={"error": str(exc)})


@lru_cache
def get_langfuse_tracer() -> LangfuseTracer:
    """Provide a singleton LangfuseTracer."""
    return LangfuseTracer(get_settings())