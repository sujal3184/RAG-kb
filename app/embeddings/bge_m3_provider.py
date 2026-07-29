"""Primary embedding provider using BAAI/bge-m3.

bge-m3 is a strong, multilingual, open-weight embedding model well suited
as the primary choice for a general-purpose knowledge base. The model is
loaded lazily (on first use) via sentence-transformers, since loading is
slow and shouldn't happen at app startup.
"""

import asyncio
import logging

from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingProvider, EmbeddingResult
from app.embeddings.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class BgeM3Provider(EmbeddingProvider):
    """Embedding provider backed by the BAAI/bge-m3 model."""

    def __init__(self, model_id: str, *, cache_dir: str, batch_size: int) -> None:
        """Store configuration; the actual model is NOT loaded here.

        Args:
            model_id: HuggingFace model identifier (e.g. "BAAI/bge-m3").
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
        """Embed a batch of texts using bge-m3.

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
            raise EmbeddingError(f"bge-m3 embedding failed: {exc}") from exc

        dimension = len(vectors[0]) if vectors else 0
        return EmbeddingResult(vectors=vectors, model_name=self._model_id, dimension=dimension)

    def _get_or_load_model(self) -> SentenceTransformer:
        """Load the model on first use, then reuse it for subsequent calls.

        This runs inside a thread (via asyncio.to_thread in `embed`), since
        model loading involves blocking disk/network I/O and CPU work.
        """
        if self._model is None:
            logger.info("Loading primary embedding model", extra={"model": self._model_id})
            self._model = SentenceTransformer(self._model_id, cache_folder=self._cache_dir)
            logger.info("Primary embedding model loaded", extra={"model": self._model_id})
        return self._model