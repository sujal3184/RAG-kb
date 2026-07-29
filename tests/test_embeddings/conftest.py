"""Fake embedding providers for fast, deterministic EmbeddingService tests.

These avoid loading any real ML models — EmbeddingService's fallback
LOGIC is what we're testing here, not the models themselves (that's
covered separately in test_providers_integration.py).
"""

import pytest

from app.embeddings.base import EmbeddingProvider, EmbeddingResult
from app.embeddings.exceptions import EmbeddingError


class FakeEmbeddingProvider(EmbeddingProvider):
    """A fake provider that either succeeds predictably or always fails,
    depending on how it's configured — used to simulate primary/fallback
    scenarios without any real model inference."""

    def __init__(self, name: str, *, should_fail: bool = False, dimension: int = 4) -> None:
        self._name = name
        self._should_fail = should_fail
        self._dimension = dimension
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._name

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        self.call_count += 1
        if self._should_fail:
            raise EmbeddingError(f"{self._name} is configured to fail")

        vectors = [[float(i)] * self._dimension for i in range(len(texts))]
        return EmbeddingResult(vectors=vectors, model_name=self._name, dimension=self._dimension)


@pytest.fixture
def working_primary() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider("fake-primary", should_fail=False)


@pytest.fixture
def failing_primary() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider("fake-primary", should_fail=True)


@pytest.fixture
def working_fallback() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider("fake-fallback", should_fail=False)


@pytest.fixture
def failing_fallback() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider("fake-fallback", should_fail=True)