"""Integration tests for CacheService, using a real Redis instance."""

import uuid

import pytest
from redis.asyncio import Redis

from app.core.cache import CacheService

pytestmark = pytest.mark.docker


from collections.abc import AsyncIterator


@pytest.fixture
async def cache_service() -> AsyncIterator[CacheService]:
    redis_client = Redis(host="localhost", port=6379, db=1)  # separate DB from Celery's db=0
    service = CacheService(redis_client, enabled=True)
    yield service
    await redis_client.flushdb()
    await redis_client.aclose()


@pytest.mark.asyncio
async def test_set_and_get_round_trip(cache_service: CacheService) -> None:
    key = f"test:{uuid.uuid4()}"
    await cache_service.set(key, {"answer": 42}, ttl_seconds=60)

    result = await cache_service.get(key)

    assert result == {"answer": 42}


@pytest.mark.asyncio
async def test_missing_key_returns_none(cache_service: CacheService) -> None:
    result = await cache_service.get(f"nonexistent:{uuid.uuid4()}")
    assert result is None


@pytest.mark.asyncio
async def test_disabled_cache_always_misses(cache_service: CacheService) -> None:
    cache_service._enabled = False
    key = f"test:{uuid.uuid4()}"

    await cache_service.set(key, {"value": 1}, ttl_seconds=60)
    result = await cache_service.get(key)

    assert result is None


@pytest.mark.asyncio
async def test_delete_by_prefix_removes_matching_keys(cache_service: CacheService) -> None:
    prefix = f"group:{uuid.uuid4()}:"
    await cache_service.set(f"{prefix}a", "value_a", ttl_seconds=60)
    await cache_service.set(f"{prefix}b", "value_b", ttl_seconds=60)
    await cache_service.set("unrelated:key", "value_c", ttl_seconds=60)

    await cache_service.delete_by_prefix(prefix)

    assert await cache_service.get(f"{prefix}a") is None
    assert await cache_service.get(f"{prefix}b") is None
    assert await cache_service.get("unrelated:key") == "value_c"


def test_build_key_is_deterministic() -> None:
    key1 = CacheService.build_key("namespace", "some query text")
    key2 = CacheService.build_key("namespace", "some query text")
    key3 = CacheService.build_key("namespace", "different query text")

    assert key1 == key2
    assert key1 != key3


def test_build_key_includes_prefix_parts() -> None:
    key = CacheService.build_key("namespace", "kb-123", "query text")
    assert key.startswith("namespace:kb-123:")