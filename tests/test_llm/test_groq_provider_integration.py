"""Integration test using the REAL Groq API.

Requires a valid GROQ_API_KEY in your .env. NOT part of the default fast
test suite — run explicitly with:

    uv run pytest tests/test_llm/test_groq_provider_integration.py -m external_api -v
"""

import pytest

from app.config.settings import get_settings
from app.llm.base import ChatMessage, MessageRole
from app.llm.groq_provider import GroqProvider

pytestmark = pytest.mark.external_api


@pytest.mark.asyncio
async def test_groq_chat_returns_a_real_response() -> None:
    settings = get_settings()
    provider = GroqProvider(
        api_key=settings.GROQ_API_KEY,
        model=settings.PRIMARY_LLM_MODEL,
        temperature=0.0,
        max_output_tokens=50,
        timeout_seconds=30.0,
    )

    response = await provider.chat(
        [ChatMessage(role=MessageRole.USER, content="Say 'hello' and nothing else.")]
    )

    assert "hello" in response.content.lower()
    assert response.model_name == settings.PRIMARY_LLM_MODEL
    assert response.output_tokens > 0


@pytest.mark.asyncio
async def test_groq_stream_chat_yields_multiple_chunks() -> None:
    settings = get_settings()
    provider = GroqProvider(
        api_key=settings.GROQ_API_KEY,
        model=settings.PRIMARY_LLM_MODEL,
        temperature=0.0,
        max_output_tokens=50,
        timeout_seconds=30.0,
    )

    chunks = [
        chunk
        async for chunk in provider.stream_chat(
            [ChatMessage(role=MessageRole.USER, content="Count from 1 to 5.")]
        )
    ]

    assert len(chunks) > 1
    assert "".join(chunks).strip() != ""