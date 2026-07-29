# def main():
#     print("Hello from rag-kb!")


# if __name__ == "__main__":
#     main()



import asyncio
from app.embeddings.bge_m3_provider import BgeM3Provider
from app.embeddings.nomic_provider import NomicProvider
from app.embeddings.embedding_service import EmbeddingService
from app.config.settings import get_settings

settings = get_settings()
broken_primary = BgeM3Provider("this-model-does-not-exist", cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR, batch_size=4)
fallback = NomicProvider(settings.FALLBACK_EMBEDDING_MODEL, cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR, batch_size=4)
service = EmbeddingService(broken_primary, fallback)

async def main():
    result = await service.embed_texts(["test fallback behavior"])
    print(f"Model used: {result.model_name}")
    print(f"Is using fallback: {service.is_using_fallback}")

asyncio.run(main())
# Expected: Model used: nomic-ai/nomic-embed-text-v1.5, Is using fallback: True