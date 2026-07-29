"""Embedding service with automatic primary -> fallback failover.

This is the ONLY class other modules (Module 10: Qdrant integration,
Module 17: background workers) should depend on. It hides which specific
model actually produced the embeddings behind a single, reliable
`embed_texts` method.
"""

import logging

from app.embeddings.base import EmbeddingProvider, EmbeddingResult
from app.embeddings.exceptions import AllEmbeddingProvidersFailedError, EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embeds text using a primary provider, falling back to a secondary
    provider if the primary fails."""

    def __init__(
        self, primary_provider: EmbeddingProvider, fallback_provider: EmbeddingProvider
    ) -> None:
        """Store the two providers to try, in order.

        Args:
            primary_provider: tried first for every request.
            fallback_provider: used only if the primary provider raises
                an EmbeddingError.
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        # Once the primary fails, we stop retrying it for the rest of this
        # service instance's lifetime — see Module design notes on why.
        self._primary_marked_unavailable = False

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts, using the primary model if possible.

        Args:
            texts: the texts to embed.

        Returns:
            An EmbeddingResult — check `.model_name` to see which model
            actually produced these vectors.

        Raises:
            AllEmbeddingProvidersFailedError: if BOTH the primary and
                fallback providers fail.
        """
        if not texts:
            return EmbeddingResult(vectors=[], model_name="", dimension=0)

        if not self._primary_marked_unavailable:
            try:
                return await self.primary_provider.embed(texts)
            except EmbeddingError as exc:
                logger.warning(
                    "Primary embedding provider failed, switching to fallback "
                    "for the remainder of this session",
                    extra={"primary_model": self.primary_provider.model_name, "error": str(exc)},
                )
                self._primary_marked_unavailable = True

        try:
            return await self.fallback_provider.embed(texts)
        except EmbeddingError as exc:
            logger.error(
                "Fallback embedding provider also failed",
                extra={"fallback_model": self.fallback_provider.model_name, "error": str(exc)},
            )
            raise AllEmbeddingProvidersFailedError(
                "Both primary and fallback embedding providers failed. "
                "Text could not be embedded."
            ) from exc

    async def embed_text(self, text: str) -> list[float]:
        """Convenience wrapper to embed a single piece of text.

        Args:
            text: a single piece of text to embed.

        Returns:
            A single embedding vector.
        """
        result = await self.embed_texts([text])
        return result.vectors[0]

    @property
    def is_using_fallback(self) -> bool:
        """Whether the primary provider has failed and fallback is now active.

        Useful for health checks / observability (Module 19) to surface
        "the system is degraded but still working" as a warning signal.
        """
        return self._primary_marked_unavailable