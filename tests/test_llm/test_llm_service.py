"""Fast unit tests for LLMService's fallback/retry orchestration logic."""

import pytest

from app.llm.base import ChatMessage, LLMResponse, MessageRole
from app.llm.exceptions import AllLLMProvidersFailedError, LLMError
from app.llm.llm_service import LLMService


class FakeLLMProvider:
    """A fake provider that fails a configured number of times before
    succeeding (or always fails), without any real network calls."""

    def __init__(self, name: str, *, fail_times: int = 0, always_fail: bool = False) -> None:
        self._name = name
        self._fail_times = fail_times
        self._always_fail = always_fail
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._name

    async def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        self.call_count += 1
        if self._always_fail or self.call_count <= self._fail_times:
            raise LLMError(f"{self._name} simulated failure (attempt {self.call_count})")
        return LLMResponse(content="A generated answer.", model_name=self._name, input_tokens=10, output_tokens=5)

    async def stream_chat(self, messages: list[ChatMessage]):
        self.call_count += 1
        if self._always_fail or self.call_count <= self._fail_times:
            raise LLMError(f"{self._name} simulated streaming failure")
        for token in ["A ", "generated ", "answer."]:
            yield token


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role=MessageRole.USER, content="Hello")]


@pytest.mark.asyncio
async def test_succeeds_on_first_try_with_no_retries_needed() -> None:
    primary = FakeLLMProvider("primary", fail_times=0)
    fallback = FakeLLMProvider("fallback")
    service = LLMService(primary, fallback, max_retries=2)

    response = await service.chat(_messages())

    assert response.model_name == "primary"
    assert primary.call_count == 1
    assert fallback.call_count == 0


@pytest.mark.asyncio
async def test_retries_primary_before_giving_up() -> None:
    primary = FakeLLMProvider("primary", fail_times=1)  # fails once, then succeeds
    fallback = FakeLLMProvider("fallback")
    service = LLMService(primary, fallback, max_retries=2)

    response = await service.chat(_messages())

    assert response.model_name == "primary"
    assert primary.call_count == 2  # 1 failure + 1 success
    assert fallback.call_count == 0


@pytest.mark.asyncio
async def test_falls_back_after_retries_exhausted() -> None:
    primary = FakeLLMProvider("primary", always_fail=True)
    fallback = FakeLLMProvider("fallback", fail_times=0)
    service = LLMService(primary, fallback, max_retries=2)

    response = await service.chat(_messages())

    assert response.model_name == "fallback"
    assert primary.call_count == 3  # 1 initial + 2 retries, all failed
    assert fallback.call_count == 1


@pytest.mark.asyncio
async def test_raises_when_both_providers_fail() -> None:
    primary = FakeLLMProvider("primary", always_fail=True)
    fallback = FakeLLMProvider("fallback", always_fail=True)
    service = LLMService(primary, fallback, max_retries=1)

    with pytest.raises(AllLLMProvidersFailedError):
        await service.chat(_messages())


@pytest.mark.asyncio
async def test_stream_chat_falls_back_when_primary_fails_immediately() -> None:
    primary = FakeLLMProvider("primary", always_fail=True)
    fallback = FakeLLMProvider("fallback")
    service = LLMService(primary, fallback, max_retries=0)

    chunks = [chunk async for chunk in service.stream_chat(_messages())]

    assert "".join(chunks) == "A generated answer."


@pytest.mark.asyncio
async def test_stream_chat_raises_when_both_fail() -> None:
    primary = FakeLLMProvider("primary", always_fail=True)
    fallback = FakeLLMProvider("fallback", always_fail=True)
    service = LLMService(primary, fallback, max_retries=0)

    with pytest.raises(AllLLMProvidersFailedError):
        async for _ in service.stream_chat(_messages()):
            pass