"""Run an evaluation dataset and write results to a JSON file.

Usage:
    # Retrieval only — fast, no LLM calls, no cost
    uv run python -m scripts.run_evaluation evaluation/datasets/my_dataset.json --retrieval-only

    # Full evaluation including LLM-judged answer quality
    uv run python -m scripts.run_evaluation evaluation/datasets/my_dataset.json
"""

import argparse
import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.api.dependencies import (
    get_bm25_store,
    get_embedding_service,
    get_llm_service,
    get_reranker_provider,
    get_token_counter,
    get_vector_store,
)
from app.chunking.base import TokenCounter
from app.config.settings import get_settings
from app.db.session import async_session_factory
from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.base import EvalDataset, EvalExample
from app.evaluation.evaluation_runner import EvaluationRunner
from app.llm.prompt_builder import PromptBuilder
from app.repositories.chunk_repository import ChunkRepository
from app.retrieval.context_compressor import ContextCompressor
from app.retrieval.deduplication import Deduplicator
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranking_service import RerankingService

_RESULTS_DIR = Path("evaluation/results")


def load_dataset(path: Path) -> EvalDataset:
    """Load an evaluation dataset from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalDataset(
        name=data["name"],
        knowledge_base_id=uuid.UUID(data["knowledge_base_id"]),
        examples=[
            EvalExample(
                question=e["question"],
                relevant_chunk_ids=e["relevant_chunk_ids"],
                reference_answer=e.get("reference_answer"),
                metadata=e.get("metadata", {}),
            )
            for e in data["examples"]
        ],
    )


async def main(dataset_path: Path, *, retrieval_only: bool) -> None:
    """Run the evaluation and print a summary."""
    settings = get_settings()
    dataset = load_dataset(dataset_path)

    embedding_service = get_embedding_service()
    llm_service = get_llm_service()
    token_counter: TokenCounter = get_token_counter()

    async with async_session_factory() as session:
        runner = EvaluationRunner(
            ChunkRepository(session),
            HybridRetriever(
                get_vector_store(),
                get_bm25_store(),
                embedding_service,
                top_k_per_method=settings.HYBRID_TOP_K_PER_METHOD,
                rrf_k=settings.HYBRID_DENSE_WEIGHT_RRF_K,
            ),
            RerankingService(get_reranker_provider()),
            ContextCompressor(Deduplicator(embedding_service), token_counter),
            PromptBuilder(token_counter),
            llm_service,
            AnswerJudge(llm_service),
            retrieval_top_k=settings.HYBRID_TOP_K_PER_METHOD,
            rerank_top_k=5,
            similarity_threshold=settings.DEDUPLICATION_SIMILARITY_THRESHOLD,
            max_context_tokens=settings.MAX_CONTEXT_TOKENS,
        )

        run = await runner.run(
            dataset, evaluate_answers=not retrieval_only, delay_seconds=args.delay
        )
    _print_summary(run)
    _write_results(run)


def _print_summary(run) -> None:
    """Print aggregate metrics to the console."""
    print("\n" + "=" * 60)
    print(f"EVALUATION: {run.dataset_name}")
    print("=" * 60)

    retrieval = run.aggregate_retrieval()
    if retrieval:
        print("\nRETRIEVAL")
        print(f"  Hit rate      {retrieval['hit_rate']:.1%}   (any relevant chunk retrieved)")
        print(f"  Recall@k      {retrieval['recall_at_k']:.1%}   (fraction of relevant chunks found)")
        print(f"  Precision@k   {retrieval['precision_at_k']:.1%}   (fraction of retrieved that were relevant)")
        print(f"  MRR           {retrieval['mrr']:.3f}   (1.0 = relevant chunk always ranked first)")

    answers = run.aggregate_answers()
    if answers:
        print("\nANSWER QUALITY (1-5, LLM-judged — treat as relative, not absolute)")
        print(f"  Correctness   {answers['correctness']:.2f}")
        print(f"  Groundedness  {answers['groundedness']:.2f}")
        print(f"  Completeness  {answers['completeness']:.2f}")

    failed = [r for r in run.results if r.error]
    if failed:
        print(f"\n{len(failed)} example(s) errored:")
        for result in failed[:5]:
            print(f"  - {result.question[:60]}: {result.error}")

    misses = [r for r in run.results if not r.error and not r.retrieval.hit]
    if misses:
        print(f"\n{len(misses)} question(s) retrieved NO relevant chunk — start debugging here:")
        for result in misses[:5]:
            print(f"  - {result.question[:70]}")

    print()


def _write_results(run) -> None:
    """Write full results to a timestamped JSON file."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = _RESULTS_DIR / f"{run.dataset_name}_{timestamp}.json"

    payload = {
        "dataset_name": run.dataset_name,
        "started_at": run.started_at.isoformat(),
        "config": run.config,
        "aggregate_retrieval": run.aggregate_retrieval(),
        "aggregate_answers": run.aggregate_answers(),
        "results": [asdict(r) for r in run.results],
    }

    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Full results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG evaluation dataset.")
    parser.add_argument("dataset", type=Path, help="Path to the dataset JSON file")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM answer generation and judging (fast, no API cost)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds to wait between questions, to respect API rate limits (default: 4.0)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.dataset, retrieval_only=args.retrieval_only))