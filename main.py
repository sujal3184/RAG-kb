import asyncio
from app.api.dependencies import get_llm_service
from app.llm.base import ChatMessage, MessageRole

async def main():
    llm = get_llm_service()
    try:
        response = await llm.primary_provider.chat([
            ChatMessage(role=MessageRole.USER, content='Say hello')
        ])
        print('OK:', response)
    except Exception as exc:
        print('FAILED:', repr(exc))

asyncio.run(main())