"""Fallback embedding provider using nomic-embed-text.

Used automatically by EmbeddingService if the primary (bge-m3) provider
fails. Structurally identical to BgeM3Provider — the only difference is
which model is loaded — demonstrating why the EmbeddingProvider interface
makes swapping/adding models straightforward.
"""

import asyncio
import logging

from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingProvider, EmbeddingResult
from app.embeddings.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class NomicProvider(EmbeddingProvider):
    """Embedding provider backed by the nomic-embed-text model."""

    def __init__(self, model_id: str, *, cache_dir: str, batch_size: int) -> None:
        """Store configuration; the actual model is NOT loaded here.

        Args:
            model_id: HuggingFace model identifier
                (e.g. "nomic-ai/nomic-embed-text-v1.5").
            cache_dir: where sentence-transformers should cache downloaded
                model weights.
            batch_size: how many texts to embed per internal model call.
        """
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        return self._model_id

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts using nomic-embed-text.

        Raises:
            EmbeddingError: if the model fails to load or run.
        """
        try:
            model = await asyncio.to_thread(self._get_or_load_model)
            vectors = await asyncio.to_thread(
                lambda: model.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
            )
        except Exception as exc:
            raise EmbeddingError(f"nomic-embed-text embedding failed: {exc}") from exc

        dimension = len(vectors[0]) if vectors else 0
        return EmbeddingResult(vectors=vectors, model_name=self._model_id, dimension=dimension)

    def _get_or_load_model(self) -> SentenceTransformer:
        """Load the model on first use, then reuse it for subsequent calls."""
        if self._model is None:
            logger.info("Loading fallback embedding model", extra={"model": self._model_id})
            self._model = SentenceTransformer(
                self._model_id, cache_folder=self._cache_dir, trust_remote_code=True
            )
            logger.info("Fallback embedding model loaded", extra={"model": self._model_id})
        return self._model