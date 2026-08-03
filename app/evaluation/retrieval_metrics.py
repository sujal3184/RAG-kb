"""Retrieval quality metrics.

These are computed by comparing retrieved chunk IDs against known-relevant
chunk IDs — exact set operations, no fuzzy matching, fully deterministic.
"""

from app.evaluation.base import RetrievalScores


def score_retrieval(
    *, retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str]
) -> RetrievalScores:
    """Compute retrieval metrics for one question.

    Args:
        retrieved_chunk_ids: chunk IDs the system returned, in rank order
            (best first) — order matters for MRR.
        relevant_chunk_ids: chunk IDs that genuinely answer the question.

    Returns:
        RetrievalScores with hit, recall@k, precision@k, and MRR.
    """
    relevant = set(relevant_chunk_ids)

    if not relevant:
        # No ground truth means nothing can be scored — return zeros
        # rather than dividing by zero or silently claiming success.
        return RetrievalScores(
            hit=False,
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    retrieved_relevant = [cid for cid in retrieved_chunk_ids if cid in relevant]

    hit = len(retrieved_relevant) > 0
    recall = len(set(retrieved_relevant)) / len(relevant)
    precision = (
        len(retrieved_relevant) / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0.0
    )
    mrr = _reciprocal_rank(retrieved_chunk_ids, relevant)

    return RetrievalScores(
        hit=hit,
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )


def _reciprocal_rank(retrieved_chunk_ids: list[str], relevant: set[str]) -> float:
    """Return 1/rank of the first relevant chunk, or 0.0 if none found.

    A relevant chunk at position 1 scores 1.0; position 2 scores 0.5;
    position 10 scores 0.1. This penalizes burying the right answer deep
    in the results — which matters here because context compression
    (Module 13) may drop low-ranked chunks before they reach the LLM.
    """
    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0