import asyncio, uuid
from app.api.dependencies import (
    get_embedding_service, get_vector_store, get_bm25_store,
    get_reranker_provider, get_token_counter, get_llm_service,
)
from app.retrieval.base import VectorPoint
from app.retrieval.bm25_store import BM25Document
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranking_service import RerankingService
from app.retrieval.deduplication import Deduplicator
from app.retrieval.context_compressor import ContextCompressor
from app.llm.prompt_builder import ChunkWithSource, PromptBuilder

async def main():
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()
    bm25_store = get_bm25_store()

    kb_id = uuid.uuid4()
    document_id = uuid.uuid4()
    texts = [
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Bananas are a good source of potassium and fiber.",
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

    query = "What is the capital of France?"

    retriever = HybridRetriever(vector_store, bm25_store, embedding_service, top_k_per_method=10, rrf_k=60)
    candidates = await retriever.retrieve(
        knowledge_base_id=kb_id, query=query, bm25_documents=bm25_documents, top_k=3
    )

    reranking_service = RerankingService(get_reranker_provider())
    ranked = await reranking_service.rerank(query=query, candidates=candidates, top_k=3)

    compressor = ContextCompressor(Deduplicator(embedding_service), get_token_counter())
    compressed = await compressor.compress(ranked, similarity_threshold=0.9, max_context_tokens=1000)

    prompt_builder = PromptBuilder(get_token_counter())
    chunks_with_source = [ChunkWithSource(c, "geography_facts.pdf") for c in compressed]
    prompt = prompt_builder.build(query=query, chunks=chunks_with_source)

    llm_service = get_llm_service()
    response = await llm_service.chat(prompt.messages)

    print(f"Model used: {response.model_name}")
    print(f"Answer: {response.content}")
    await vector_store.delete_collection(knowledge_base_id=kb_id)

asyncio.run(main())