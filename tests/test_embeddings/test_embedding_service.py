"""Fast unit tests for EmbeddingService's fallback orchestration logic."""

import pytest

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.exceptions import AllEmbeddingProvidersFailedError


@pytest.mark.asyncio
async def test_uses_primary_when_it_succeeds(working_primary, working_fallback) -> None:
    """When the primary provider works, it should be used, and fallback
    should never be called."""
    service = EmbeddingService(working_primary, working_fallback)

    result = await service.embed_texts(["hello world"])

    assert result.model_name == "fake-primary"
    assert working_primary.call_count == 1
    assert working_fallback.call_count == 0
    assert service.is_using_fallback is False


@pytest.mark.asyncio
async def test_falls_back_when_primary_fails(failing_primary, working_fallback) -> None:
    """When the primary provider fails, the fallback should be used instead."""
    service = EmbeddingService(failing_primary, working_fallback)

    result = await service.embed_texts(["hello world"])

    assert result.model_name == "fake-fallback"
    assert working_fallback.call_count == 1
    assert service.is_using_fallback is True


@pytest.mark.asyncio
async def test_primary_is_not_retried_after_first_failure(
    failing_primary, working_fallback
) -> None:
    """Once the primary has failed once, subsequent calls should skip it
    entirely and go straight to the fallback."""
    service = EmbeddingService(failing_primary, working_fallback)

    await service.embed_texts(["first call"])
    await service.embed_texts(["second call"])

    assert failing_primary.call_count == 1  # only tried once, not twice
    assert working_fallback.call_count == 2


@pytest.mark.asyncio
async def test_raises_when_both_providers_fail(failing_primary, failing_fallback) -> None:
    """If both providers fail, a clear AllEmbeddingProvidersFailedError should raise."""
    service = EmbeddingService(failing_primary, failing_fallback)

    with pytest.raises(AllEmbeddingProvidersFailedError):
        await service.embed_texts(["hello world"])


@pytest.mark.asyncio
async def test_embed_text_returns_single_vector(working_primary, working_fallback) -> None:
    """The single-text convenience method should return one vector, not a list of results."""
    service = EmbeddingService(working_primary, working_fallback)

    vector = await service.embed_text("hello world")

    assert isinstance(vector, list)
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.asyncio
async def test_empty_input_returns_empty_result(working_primary, working_fallback) -> None:
    """Embedding an empty list should return an empty result without calling either provider."""
    service = EmbeddingService(working_primary, working_fallback)

    result = await service.embed_texts([])

    assert result.vectors == []
    assert working_primary.call_count == 0