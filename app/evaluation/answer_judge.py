"""LLM-as-judge for answer quality.

IMPORTANT CAVEAT: the judge is an LLM and is therefore fallible. It can
be inconsistent between runs, lenient, and biased toward verbose answers.
Treat these scores as useful for RELATIVE comparison (did config B beat
config A?) rather than as absolute measures of quality. The retrieval
metrics are the more trustworthy numbers.
"""

import json
import logging

from app.evaluation.base import AnswerScores
from app.llm.base import ChatMessage, MessageRole
from app.llm.llm_service import LLMService

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluator of question-answering systems. You will be \
given a question, the context documents that were retrieved, the answer \
that was generated, and optionally a reference answer.

Score the generated answer on three criteria, each from 1 to 5:

CORRECTNESS (is the answer factually right?)
5 - Fully correct and matches the reference/facts
4 - Correct with minor imprecision
3 - Partially correct, some errors
2 - Mostly incorrect
1 - Entirely incorrect

GROUNDEDNESS (is every claim supported by the provided context?)
5 - Every claim traceable to the context
4 - Almost entirely grounded, one minor unsupported detail
3 - Mix of grounded and unsupported claims
2 - Mostly unsupported by the context
1 - Fabricated; contradicts or ignores the context
Note: correctly stating "the context does not contain this information" \
is FULLY GROUNDED and should score 5.

COMPLETENESS (does it address the whole question?)
5 - Fully addresses every part of the question
4 - Addresses the main point, minor omission
3 - Addresses roughly half
2 - Barely addresses the question
1 - Does not address the question

Respond with ONLY a JSON object, no other text:
{"correctness": <1-5>, "groundedness": <1-5>, "completeness": <1-5>, \
"reasoning": "<one or two sentences explaining the scores>"}
"""

_JUDGE_USER_TEMPLATE = """\
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}
{reference_section}
Score the generated answer now."""


class AnswerJudge:
    """Scores generated answers using an LLM against a fixed rubric."""

    def __init__(self, llm_service: LLMService) -> None:
        """Store the LLM used for judging.

        Args:
            llm_service: the LLM that will act as judge. Ideally a
                different (or stronger) model than the one being
                evaluated — a model judging its own output tends to be
                more lenient. Using the same model is acceptable for
                relative comparisons but worth knowing about.
        """
        self.llm_service = llm_service

    async def judge(
        self,
        *,
        question: str,
        retrieved_context: str,
        generated_answer: str,
        reference_answer: str | None = None,
    ) -> AnswerScores:
        """Score one generated answer.

        Args:
            question: the original question.
            retrieved_context: the chunks the answer was generated from.
            generated_answer: the system's answer.
            reference_answer: a known-good answer, if available.

        Returns:
            AnswerScores. If the judge fails or returns unparseable
            output, returns all-zero scores with the error in `reasoning`
            rather than raising — one bad judgement shouldn't abort a
            whole evaluation run.
        """
        reference_section = (
            f"\nREFERENCE ANSWER:\n{reference_answer}\n" if reference_answer else ""
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_JUDGE_SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=_JUDGE_USER_TEMPLATE.format(
                    question=question,
                    context=retrieved_context,
                    generated_answer=generated_answer,
                    reference_section=reference_section,
                ),
            ),
        ]

        try:
            response = await self.llm_service.chat(messages)
            return self._parse_scores(response.content)
        except Exception as exc:
            logger.warning("Judge failed for a question", extra={"error": str(exc)})
            return AnswerScores(
                correctness=0,
                groundedness=0,
                completeness=0,
                reasoning=f"Judging failed: {exc}",
            )

    @staticmethod
    def _parse_scores(raw_response: str) -> AnswerScores:
        """Extract scores from the judge's JSON response.

        LLMs sometimes wrap JSON in markdown fences or add commentary
        despite instructions, so we locate the JSON object rather than
        assuming the whole response is clean JSON.
        """
        cleaned = raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```")

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            return AnswerScores(
                correctness=0,
                groundedness=0,
                completeness=0,
                reasoning=f"Judge returned no JSON: {raw_response[:200]}",
            )

        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            return AnswerScores(
                correctness=0,
                groundedness=0,
                completeness=0,
                reasoning=f"Judge returned invalid JSON: {exc}",
            )

        return AnswerScores(
            correctness=int(data.get("correctness", 0)),
            groundedness=int(data.get("groundedness", 0)),
            completeness=int(data.get("completeness", 0)),
            reasoning=str(data.get("reasoning", "")),
        )