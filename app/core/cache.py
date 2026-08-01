"""Generic caching service backed by Redis.

Every part of the app that wants to cache something (query embeddings,
RAG responses, etc.) goes through THIS class rather than talking to Redis
directly — keeping serialization, key-building, error handling, and TTL
logic in one place.

Caching is FAIL-OPEN: if Redis is unreachable or errors, cache operations
log a warning and behave as a "miss" (get) or a no-op (set) rather than
raising — caching must never be a hard dependency for the app to work.
"""

import hashlib
import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class CacheService:
    """Get/set/delete cached JSON-serializable values in Redis, with TTLs."""

    def __init__(self, redis_client: Redis, *, enabled: bool) -> None:
        """Store the Redis client and whether caching is enabled at all.

        Args:
            redis_client: an async Redis client instance.
            enabled: if False, every operation behaves as a no-op/miss —
                lets caching be disabled entirely via configuration
                without changing any calling code.
        """
        self._redis = redis_client
        self._enabled = enabled

    async def get(self, key: str) -> Any | None:
        """Fetch a cached value, or None if missing/expired/unavailable.

        Args:
            key: the cache key (see `build_key` for constructing one).

        Returns:
            The deserialized JSON value, or None on a cache miss OR if
            Redis itself is unreachable (fail-open).
        """
        if not self._enabled:
            return None

        try:
            raw_value = await self._redis.get(key)
        except RedisError as exc:
            logger.warning("Cache read failed, treating as miss", extra={"error": str(exc)})
            return None

        if raw_value is None:
            return None

        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            logger.warning("Cached value was not valid JSON, ignoring", extra={"key": key})
            return None

    async def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        """Store a JSON-serializable value with a time-to-live.

        Args:
            key: the cache key.
            value: any JSON-serializable value.
            ttl_seconds: how long until this entry automatically expires.
        """
        if not self._enabled:
            return

        try:
            serialized = json.dumps(value)
            await self._redis.set(key, serialized, ex=ttl_seconds)
        except (RedisError, TypeError) as exc:
            logger.warning("Cache write failed, continuing without caching", extra={"error": str(exc)})

    async def delete_by_prefix(self, prefix: str) -> None:
        """Delete all cache keys starting with a given prefix.

        Used for cache invalidation — e.g. clearing every cached RAG
        response for a knowledge base when its documents change.

        Args:
            prefix: keys matching "{prefix}*" will be deleted.
        """
        if not self._enabled:
            return

        try:
            keys_to_delete = [key async for key in self._redis.scan_iter(match=f"{prefix}*")]
            if keys_to_delete:
                await self._redis.delete(*keys_to_delete)
                logger.info(
                    "Invalidated cache entries", extra={"prefix": prefix, "count": len(keys_to_delete)}
                )
        except RedisError as exc:
            logger.warning("Cache invalidation failed", extra={"error": str(exc)})

    @staticmethod
    def build_key(*parts: str) -> str:
        """Build a deterministic, compact cache key from arbitrary parts.

        Uses a SHA-256 hash of the joined parts rather than the raw text
        itself — avoids issues with very long values, special characters,
        or key-length limits, while still being fully deterministic (the
        same inputs always produce the same key).

        Args:
            parts: strings to combine into the key (e.g. a namespace,
                a knowledge base id, and query text).

        Returns:
            A cache key string like "namespace:kb_id:<hash>".
        """
        *prefix_parts, content = parts
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
        return ":".join([*prefix_parts, content_hash])