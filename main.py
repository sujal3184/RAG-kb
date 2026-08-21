import asyncio
from app.api.dependencies import get_llm_service
from app.llm.base import ChatMessage, MessageRole

async def main():
    llm = get_llm_service()
    response = await llm.primary_provider.chat([ChatMessage(role=MessageRole.USER, content='Say hello')])
    print(response)

asyncio.run(main())