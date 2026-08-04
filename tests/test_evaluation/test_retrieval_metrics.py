"""Tests for retrieval metrics — pure computation, no infrastructure."""

from app.evaluation.retrieval_metrics import score_retrieval


def test_perfect_retrieval_scores_maximum() -> None:
    scores = score_retrieval(
        retrieved_chunk_ids=["a", "b"], relevant_chunk_ids=["a", "b"]
    )

    assert scores.hit is True
    assert scores.recall_at_k == 1.0
    assert scores.precision_at_k == 1.0
    assert scores.mrr == 1.0


def test_complete_miss_scores_zero() -> None:
    scores = score_retrieval(
        retrieved_chunk_ids=["x", "y"], relevant_chunk_ids=["a", "b"]
    )

    assert scores.hit is False
    assert scores.recall_at_k == 0.0
    assert scores.precision_at_k == 0.0
    assert scores.mrr == 0.0


def test_partial_recall() -> None:
    """One of two relevant chunks found = 50% recall."""
    scores = score_retrieval(
        retrieved_chunk_ids=["a", "x"], relevant_chunk_ids=["a", "b"]
    )

    assert scores.hit is True
    assert scores.recall_at_k == 0.5
    assert scores.precision_at_k == 0.5


def test_mrr_rewards_higher_rank() -> None:
    """A relevant chunk ranked first should score higher than ranked third."""
    first = score_retrieval(
        retrieved_chunk_ids=["a", "x", "y"], relevant_chunk_ids=["a"]
    )
    third = score_retrieval(
        retrieved_chunk_ids=["x", "y", "a"], relevant_chunk_ids=["a"]
    )

    assert first.mrr == 1.0
    assert third.mrr == pytest.approx(1 / 3)
    assert first.mrr > third.mrr


def test_empty_retrieval_is_a_miss() -> None:
    scores = score_retrieval(retrieved_chunk_ids=[], relevant_chunk_ids=["a"])

    assert scores.hit is False
    assert scores.precision_at_k == 0.0


def test_no_ground_truth_returns_zeros_not_a_crash() -> None:
    """A malformed dataset entry shouldn't divide by zero."""
    scores = score_retrieval(retrieved_chunk_ids=["a"], relevant_chunk_ids=[])

    assert scores.hit is False
    assert scores.recall_at_k == 0.0


import pytest  # noqa: E402  (used by approx above)