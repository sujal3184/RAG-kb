"""Embedding provider interface.

An `EmbeddingProvider` converts text into vector embeddings using one
specific model. `EmbeddingService` (below) composes two providers
(primary + fallback) and never needs to know which model actually
produced a given vector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """The output of embedding a batch of texts.

    Attributes:
        vectors: one embedding vector per input text, same order as input.
        model_name: which model actually produced these vectors — useful
            for logging, debugging, and downstream storage decisions
            (Module 10 may need to know which Qdrant collection/dimension
            to use).
        dimension: the length of each vector (e.g. 1024 for bge-m3).
    """

    vectors: list[list[float]]
    model_name: str
    dimension: int


class EmbeddingProvider(ABC):
    """Abstract base class for a single embedding model backend."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The identifier of the model this provider uses."""
        raise NotImplementedError

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Convert a batch of texts into embedding vectors.

        Args:
            texts: the texts to embed (e.g. chunk contents from Module 8).

        Returns:
            An EmbeddingResult containing one vector per input text.

        Raises:
            EmbeddingError: if the model fails to load or produce embeddings.
        """
        raise NotImplementedError