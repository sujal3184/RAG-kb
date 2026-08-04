"""Evaluation orchestration.

Runs a dataset through the REAL retrieval pipeline (not a
reimplementation) so measured numbers reflect what actually runs in
production.
"""

import logging
import time
import uuid
import asyncio

from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.base import (
    EvalDataset,
    EvalExample,
    EvaluationRun,
    ExampleResult,
)
from app.evaluation.retrieval_metrics import score_retrieval
from app.llm.llm_service import LLMService
from app.llm.prompt_builder import ChunkWithSource, PromptBuilder
from app.repositories.chunk_repository import ChunkRepository
from app.retrieval.bm25_store import BM25Document
from app.retrieval.context_compressor import ContextCompressor
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranking_service import RerankingService

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Executes an evaluation dataset against the live RAG pipeline."""

    def __init__(
        self,
        chunk_repo: ChunkRepository,
        hybrid_retriever: HybridRetriever,
        reranking_service: RerankingService,
        context_compressor: ContextCompressor,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
        judge: AnswerJudge,
        *,
        retrieval_top_k: int,
        rerank_top_k: int,
        similarity_threshold: float,
        max_context_tokens: int,
    ) -> None:
        """Wire the same pipeline components ConversationService uses.

        Deliberately mirrors ConversationService's dependencies so the
        evaluation measures the real pipeline rather than an
        approximation of it.
        """
        self.chunk_repo = chunk_repo
        self.hybrid_retriever = hybrid_retriever
        self.reranking_service = reranking_service
        self.context_compressor = context_compressor
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        self.judge = judge
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_tokens = max_context_tokens

    import asyncio

    async def run(
        self, dataset: EvalDataset, *, evaluate_answers: bool = True, delay_seconds: float = 4.0
    ) -> EvaluationRun:
        """Evaluate every example in a dataset.

        Args:
            dataset: the questions and ground truth to evaluate against.
            evaluate_answers: if False, only retrieval is measured — no
                LLM calls at all, so delay_seconds is irrelevant.
            delay_seconds: pause between questions to stay under Groq's
                free-tier rate limit (30 requests/minute). Each question
                makes 2 LLM calls (generation + judge) when
                evaluate_answers=True, so ~4s keeps you comfortably under
                30/min with margin. Increase this if you still see 429s;
                decrease it (or set to 0) if you're on a higher tier.

        Returns:
            An EvaluationRun containing per-question and aggregate results.
        """
        run = EvaluationRun(
            dataset_name=dataset.name,
            config={
                "retrieval_top_k": self.retrieval_top_k,
                "rerank_top_k": self.rerank_top_k,
                "similarity_threshold": self.similarity_threshold,
                "max_context_tokens": self.max_context_tokens,
                "answers_evaluated": evaluate_answers,
            },
        )

        bm25_documents = await self._load_bm25_corpus(dataset.knowledge_base_id)
        logger.info(
            "Starting evaluation",
            extra={"dataset": dataset.name, "examples": len(dataset.examples)},
        )

        for index, example in enumerate(dataset.examples, start=1):
            logger.info("Evaluating example %d/%d", index, len(dataset.examples))
            result = await self._evaluate_example(
                example,
                knowledge_base_id=dataset.knowledge_base_id,
                bm25_documents=bm25_documents,
                evaluate_answers=evaluate_answers,
            )
            run.results.append(result)

            if evaluate_answers and index < len(dataset.examples):
                await asyncio.sleep(delay_seconds)

        return run

    async def _evaluate_example(
        self,
        example: EvalExample,
        *,
        knowledge_base_id: uuid.UUID,
        bm25_documents: list[BM25Document],
        evaluate_answers: bool,
    ) -> ExampleResult:
        """Run one question through the pipeline and score the outcome."""
        started = time.perf_counter()

        try:
            candidates = await self.hybrid_retriever.retrieve(
                knowledge_base_id=knowledge_base_id,
                query=example.question,
                bm25_documents=bm25_documents,
                top_k=self.retrieval_top_k,
            )
            ranked = await self.reranking_service.rerank(
                query=example.question, candidates=candidates, top_k=self.rerank_top_k
            )
            compressed = await self.context_compressor.compress(
                ranked,
                similarity_threshold=self.similarity_threshold,
                max_context_tokens=self.max_context_tokens,
            )

            # Retrieval is scored on the COMPRESSED set — that's what
            # actually reaches the LLM, so it's what determines whether
            # the answer can be correct.
            retrieval_scores = score_retrieval(
                retrieved_chunk_ids=[c.chunk_id for c in compressed],
                relevant_chunk_ids=example.relevant_chunk_ids,
            )

            result = ExampleResult(
                question=example.question,
                retrieval=retrieval_scores,
                latency_seconds=time.perf_counter() - started,
            )

            if not evaluate_answers:
                return result

            chunks_with_source = [ChunkWithSource(c, "evaluation") for c in compressed]
            prompt = self.prompt_builder.build(
                query=example.question, chunks=chunks_with_source
            )
            llm_response = await self.llm_service.chat(prompt.messages)

            result.generated_answer = llm_response.content
            result.answer = await self.judge.judge(
                question=example.question,
                retrieved_context="\n\n".join(c.text for c in compressed),
                generated_answer=llm_response.content,
                reference_answer=example.reference_answer,
            )
            result.latency_seconds = time.perf_counter() - started
            return result

        except Exception as exc:
            logger.warning(
                "Example failed during evaluation",
                extra={"question": example.question[:80], "error": str(exc)},
            )
            return ExampleResult(
                question=example.question,
                retrieval=score_retrieval(
                    retrieved_chunk_ids=[], relevant_chunk_ids=example.relevant_chunk_ids
                ),
                latency_seconds=time.perf_counter() - started,
                error=str(exc),
            )

    async def _load_bm25_corpus(self, knowledge_base_id: uuid.UUID) -> list[BM25Document]:
        """Load all chunks in the knowledge base for BM25 search."""
        chunks = await self.chunk_repo.list_for_knowledge_base(knowledge_base_id)
        return [
            BM25Document(
                chunk_id=str(c.id),
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
            )
            for c in chunks
        ]