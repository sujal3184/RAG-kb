import asyncio, uuid
from app.api.dependencies import (
    get_embedding_service, get_vector_store, get_bm25_store,
    get_reranker_provider,
)
from app.retrieval.base import VectorPoint
from app.retrieval.bm25_store import BM25Document
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranking_service import RerankingService
async def main():
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()
    bm25_store = get_bm25_store()
    kb_id = uuid.uuid4()
    document_id = uuid.uuid4()
    texts = [
        "Paris is the capital city of France.",
        "The Eiffel Tower was completed in 1889.",
        "Bananas are rich in potassium and fiber.",
    ]

    embed_result = await embedding_service.embed_texts(texts)
    points = [
        VectorPoint(chunk_id=str(uuid.uuid4()), document_id=document_id, knowledge_base_id=kb_id,
                    chunk_index=i, text=texts[i], vector=embed_result.vectors[i])
        for i in range(len(texts))
    ]
    await vector_store.upsert(knowledge_base_id=kb_id, points=points, vector_dimension=embed_result.dimension)
    bm25_documents = [
        BM25Document(chunk_id=p.chunk_id, document_id=p.document_id, chunk_index=p.chunk_index, text=p.text)
        for p in points
    ]
    retriever = HybridRetriever(vector_store, bm25_store, embedding_service, top_k_per_method=10, rrf_k=60)
    candidates = await retriever.retrieve(
        knowledge_base_id=kb_id, query="What is the capital of France?",
        bm25_documents=bm25_documents, top_k=3,
    )
    reranking_service = RerankingService(get_reranker_provider())
    ranked = await reranking_service.rerank(
        query="What is the capital of France?", candidates=candidates, top_k=3
    )
    for r in ranked:
        print(f"[{r.rerank_score:.4f}] {r.text}")
    await vector_store.delete_collection(knowledge_base_id=kb_id)
asyncio.run(main())