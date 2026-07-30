"""Integration test using the REAL bge-reranker-v2-m3 model.

Downloads and loads real model weights — not part of the default fast
test suite. Run explicitly with:

    uv run pytest tests/test_retrieval/test_reranker_integration.py -m embedding_model -v
"""

import pytest

from app.config.settings import get_settings
from app.retrieval.reranker import BgeRerankerProvider

pytestmark = pytest.mark.embedding_model


@pytest.mark.asyncio
async def test_reranker_scores_relevant_candidate_higher() -> None:
    """A candidate genuinely relevant to the query should score higher
    than an unrelated candidate."""
    settings = get_settings()
    reranker = BgeRerankerProvider(
        settings.RERANKER_MODEL,
        cache_dir=settings.RERANKER_MODEL_CACHE_DIR,
        batch_size=settings.RERANKER_BATCH_SIZE,
    )

    scores = await reranker.score(
        query="What is the capital of France?",
        candidates=[
            "Paris is the capital and most populous city of France.",
            "Bananas are a good source of potassium.",
        ],
    )

    assert len(scores) == 2
    assert scores[0] > scores[1]