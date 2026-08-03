"""Evaluation data types.

An evaluation dataset is a list of questions paired with the chunks that
should answer them (ground truth for retrieval) and optionally a
reference answer (ground truth for generation).
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class EvalExample:
    """One question with its expected retrieval and answer ground truth.

    Attributes:
        question: the query to run through the system.
        relevant_chunk_ids: chunk IDs that genuinely contain the answer.
            Retrieval is scored on whether these appear in the results.
        reference_answer: a correct answer, used by the LLM judge as a
            comparison point. Optional — the judge can also assess
            groundedness against retrieved context without one.
        metadata: anything useful for slicing results later (question
            type, difficulty, source document).
    """

    question: str
    relevant_chunk_ids: list[str]
    reference_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """A named collection of evaluation examples for one knowledge base.

    Attributes:
        name: identifies this dataset in results.
        knowledge_base_id: which KB these questions are about — the eval
            runs against this KB's actual indexed content.
        examples: the questions to evaluate.
    """

    name: str
    knowledge_base_id: uuid.UUID
    examples: list[EvalExample]


@dataclass
class RetrievalScores:
    """Retrieval quality metrics for a single question.

    Attributes:
        hit: whether ANY relevant chunk was retrieved at all. The most
            basic signal — if this is False, the answer cannot be correct.
        recall_at_k: fraction of relevant chunks that were retrieved.
        precision_at_k: fraction of retrieved chunks that were relevant.
        mrr: reciprocal of the rank of the first relevant chunk (1.0 if
            it was ranked first, 0.5 if second, 0 if absent). Rewards
            putting the right chunk near the top, which matters because
            context compression may drop lower-ranked chunks.
        retrieved_chunk_ids: what was actually retrieved, for debugging
            why a question scored poorly.
    """

    hit: bool
    recall_at_k: float
    precision_at_k: float
    mrr: float
    retrieved_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class AnswerScores:
    """LLM-judged answer quality for a single question.

    Scores are 1-5. See answer_judge.py for the rubric.

    Attributes:
        correctness: does the answer match the reference / the facts?
        groundedness: is every claim supported by the retrieved context,
            or did the model invent things? This is the key
            hallucination signal.
        completeness: does it address the full question?
        reasoning: the judge's explanation, for understanding low scores.
    """

    correctness: int
    groundedness: int
    completeness: int
    reasoning: str


@dataclass
class ExampleResult:
    """Complete evaluation result for one question."""

    question: str
    retrieval: RetrievalScores
    answer: AnswerScores | None = None
    generated_answer: str | None = None
    latency_seconds: float = 0.0
    error: str | None = None


@dataclass
class EvaluationRun:
    """Results of evaluating a whole dataset.

    Attributes:
        dataset_name: which dataset was run.
        started_at: when the run began, for ordering runs chronologically.
        config: the pipeline settings in effect (chunk size, top_k,
            models). Essential — a score is meaningless without knowing
            what configuration produced it.
        results: per-question results.
    """

    dataset_name: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    config: dict[str, Any] = field(default_factory=dict)
    results: list[ExampleResult] = field(default_factory=list)

    def aggregate_retrieval(self) -> dict[str, float]:
        """Average retrieval metrics across all successful examples."""
        scored = [r for r in self.results if r.error is None]
        if not scored:
            return {}

        count = len(scored)
        return {
            "hit_rate": sum(r.retrieval.hit for r in scored) / count,
            "recall_at_k": sum(r.retrieval.recall_at_k for r in scored) / count,
            "precision_at_k": sum(r.retrieval.precision_at_k for r in scored) / count,
            "mrr": sum(r.retrieval.mrr for r in scored) / count,
        }

    def aggregate_answers(self) -> dict[str, float]:
        """Average answer-quality scores across all judged examples."""
        judged = [r for r in self.results if r.answer is not None]
        if not judged:
            return {}

        count = len(judged)
        return {
            "correctness": sum(r.answer.correctness for r in judged) / count,
            "groundedness": sum(r.answer.groundedness for r in judged) / count,
            "completeness": sum(r.answer.completeness for r in judged) / count,
        }