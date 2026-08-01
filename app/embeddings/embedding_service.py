"""Embedding service with automatic primary -> fallback failover, and
optional caching of query embeddings.
"""

import logging

from app.core.cache import CacheService
from app.embeddings.base import EmbeddingProvider, EmbeddingResult
from app.embeddings.exceptions import AllEmbeddingProvidersFailedError, EmbeddingError

logger = logging.getLogger(__name__)

_CACHE_NAMESPACE = "embedding"


class EmbeddingService:
    """Embeds text using a primary provider, falling back to a secondary
    provider if the primary fails. Optionally caches single-text embeddings
    (the common case for query embedding in retrieval) to avoid recomputing
    the same embedding repeatedly."""

    def __init__(
        self,
        primary_provider: EmbeddingProvider,
        fallback_provider: EmbeddingProvider,
        *,
        cache: CacheService | None = None,
        cache_ttl_seconds: int = 86400,
    ) -> None:
        """Store the two providers and optional cache.

        Args:
            primary_provider: tried first for every request.
            fallback_provider: used only if the primary provider fails.
            cache: optional CacheService — if None, caching is simply
                skipped (embeddings are always computed fresh). This
                keeps EmbeddingService usable without Redis at all (e.g.
                in unit tests), matching our "caching is optional
                infrastructure" principle.
            cache_ttl_seconds: how long cached embeddings remain valid.
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self._primary_marked_unavailable = False
        self._cache = cache
        self._cache_ttl_seconds = cache_ttl_seconds

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts, using the primary model if possible.

        Note: caching only applies to SINGLE-text requests (via
        `embed_text`) — batch embedding (used during document processing,
        Module 17) always computes fresh, since batches are rarely
        identical across calls and caching them would add complexity for
        little benefit.
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
        """Embed a single piece of text, using the cache if available.

        This is the method called for QUERY embedding during retrieval
        (Module 11) — exactly the case where the same or very similar
        queries repeating makes caching worthwhile.
        """
        if self._cache is not None:
            cache_key = CacheService.build_key(_CACHE_NAMESPACE, text)
            cached_vector = await self._cache.get(cache_key)
            if cached_vector is not None:
                logger.debug("Embedding cache hit")
                return cached_vector

        result = await self.embed_texts([text])
        vector = result.vectors[0]

        if self._cache is not None:
            cache_key = CacheService.build_key(_CACHE_NAMESPACE, text)
            await self._cache.set(cache_key, vector, ttl_seconds=self._cache_ttl_seconds)

        return vector

    @property
    def is_using_fallback(self) -> bool:
        """Whether the primary provider has failed and fallback is now active."""
        return self._primary_marked_unavailable