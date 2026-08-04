"""Tests for the LLM judge's response parsing, using a fake LLM."""

import pytest

from app.evaluation.answer_judge import AnswerJudge
from app.llm.base import LLMResponse


class FakeLLMService:
    """Returns a canned response, so parsing can be tested without an API."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def chat(self, messages) -> LLMResponse:
        return LLMResponse(
            content=self._response_text, model_name="fake", input_tokens=0, output_tokens=0
        )


@pytest.mark.asyncio
async def test_parses_clean_json_response() -> None:
    judge = AnswerJudge(
        FakeLLMService(
            '{"correctness": 5, "groundedness": 4, "completeness": 5, '
            '"reasoning": "Accurate and well sourced."}'
        )
    )

    scores = await judge.judge(
        question="q", retrieved_context="ctx", generated_answer="ans"
    )

    assert scores.correctness == 5
    assert scores.groundedness == 4
    assert scores.completeness == 5


@pytest.mark.asyncio
async def test_parses_json_wrapped_in_markdown_fences() -> None:
    """LLMs often wrap JSON in code fences despite instructions not to."""
    judge = AnswerJudge(
        FakeLLMService(
            '```json\n{"correctness": 3, "groundedness": 3, '
            '"completeness": 3, "reasoning": "Partial."}\n```'
        )
    )

    scores = await judge.judge(
        question="q", retrieved_context="ctx", generated_answer="ans"
    )

    assert scores.correctness == 3


@pytest.mark.asyncio
async def test_parses_json_with_surrounding_commentary() -> None:
    judge = AnswerJudge(
        FakeLLMService(
            'Here is my evaluation:\n{"correctness": 2, "groundedness": 1, '
            '"completeness": 2, "reasoning": "Hallucinated."}\nHope that helps.'
        )
    )

    scores = await judge.judge(
        question="q", retrieved_context="ctx", generated_answer="ans"
    )

    assert scores.groundedness == 1


@pytest.mark.asyncio
async def test_unparseable_response_returns_zeros_not_a_crash() -> None:
    """One malformed judgement must not abort a whole evaluation run."""
    judge = AnswerJudge(FakeLLMService("I cannot evaluate this."))

    scores = await judge.judge(
        question="q", retrieved_context="ctx", generated_answer="ans"
    )

    assert scores.correctness == 0
    assert "no JSON" in scores.reasoning