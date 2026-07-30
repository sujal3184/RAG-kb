"""Reciprocal Rank Fusion (RRF).

RRF combines multiple ranked result lists into one fused ranking, using
only each item's RANK POSITION in each list (not its raw score) — making
it naturally scale-independent, which matters here since BM25 scores and
cosine similarity scores are on completely different numeric scales and
can't be meaningfully averaged directly.

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods."
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], *, k: int = 60
) -> dict[str, float]:
    """Compute RRF scores for items appearing across multiple ranked lists.

    Args:
        ranked_lists: one list of item identifiers per retrieval method,
            each already sorted best-to-worst (e.g. one list of chunk_ids
            from dense search, one from BM25 search).
        k: RRF damping constant. Higher values reduce the influence of
            an item's exact rank position (60 is the standard default
            from the original RRF paper, working well across most use
            cases without tuning).

    Returns:
        A dict mapping each item identifier to its fused RRF score.
        Items appearing in multiple lists accumulate score from each list
        they appear in — this is precisely what lets hybrid search boost
        results both methods agree on.
    """
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, start=1):
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return fused_scores