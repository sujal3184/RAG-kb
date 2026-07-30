"""Tests for Reciprocal Rank Fusion logic."""

from app.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_scores_higher_than_single_list_item() -> None:
    dense_ranked = ["a", "b", "c"]
    bm25_ranked = ["a", "d", "e"]

    scores = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)

    # "a" appears at rank 1 in both lists — should score highest
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["d"]


def test_higher_rank_scores_higher_within_same_list() -> None:
    scores = reciprocal_rank_fusion([["a", "b", "c"]], k=60)

    assert scores["a"] > scores["b"] > scores["c"]


def test_empty_lists_produce_empty_scores() -> None:
    assert reciprocal_rank_fusion([[], []], k=60) == {}


def test_item_only_in_one_list_still_gets_a_score() -> None:
    scores = reciprocal_rank_fusion([["a"], ["b"]], k=60)

    assert "a" in scores
    assert "b" in scores
    assert scores["a"] == scores["b"]  # both rank 1 in their own list