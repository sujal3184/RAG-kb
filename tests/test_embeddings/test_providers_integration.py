"""Integration tests using REAL embedding models.

These download and load actual model weights (hundreds of MB to a few
GB) and can take a while on first run. NOT part of the default fast test
suite — run explicitly with:

    uv run pytest tests/test_embeddings/test_providers_integration.py -m embedding_model -v
"""

import pytest

from app.config.settings import get_settings
from app.embeddings.bge_m3_provider import BgeM3Provider
from app.embeddings.nomic_provider import NomicProvider

pytestmark = pytest.mark.embedding_model


@pytest.mark.asyncio
async def test_bge_m3_produces_similar_vectors_for_similar_text() -> None:
    """Semantically similar sentences should have high cosine similarity."""
    settings = get_settings()
    provider = BgeM3Provider(
        settings.PRIMARY_EMBEDDING_MODEL,
        cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )

    result = await provider.embed(
        ["The cat sat on the mat.", "A kitten was resting on the rug.", "Stock markets fell today."]
    )

    assert len(result.vectors) == 3
    assert result.dimension > 0

    import numpy as np

    vectors = np.array(result.vectors)
    similar_pair_similarity = np.dot(vectors[0], vectors[1])
    dissimilar_pair_similarity = np.dot(vectors[0], vectors[2])

    assert similar_pair_similarity > dissimilar_pair_similarity


@pytest.mark.asyncio
async def test_nomic_provider_produces_valid_embeddings() -> None:
    """The fallback model should also produce well-formed embeddings independently."""
    settings = get_settings()
    provider = NomicProvider(
        settings.FALLBACK_EMBEDDING_MODEL,
        cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )

    result = await provider.embed(["This is a test sentence."])

    assert len(result.vectors) == 1
    assert result.dimension > 0