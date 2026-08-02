"""Conversation orchestration — the core RAG use case.

Coordinates conversation/message persistence with the full retrieval
pipeline built across Modules 9-15: embed the query, hybrid-retrieve
candidates, rerank, compress, build a prompt (including recent
conversation history), and generate a response.
"""

import logging
import uuid

from app.core.exceptions import NotFoundError
from app.llm.base import ChatMessage, MessageRole as PromptMessageRole
from app.llm.llm_service import LLMService
from app.llm.prompt_builder import ChunkWithSource, PromptBuilder
from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.message_repository import MessageRepository
from app.retrieval.bm25_store import BM25Document
from app.retrieval.context_compressor import ContextCompressor
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranking_service import RerankingService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.core.cache import CacheService
from app.observability.metrics import rag_pipeline_stage_duration_seconds, rag_queries_total
from app.observability.tracing import get_tracer
from app.guardrails.guardrail_service import GuardrailService

_RESPONSE_CACHE_NAMESPACE = "rag_response"

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class ConversationService:
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        chunk_repo: ChunkRepository,
        document_repo: DocumentRepository,
        kb_service: KnowledgeBaseService,
        hybrid_retriever: HybridRetriever,
        reranking_service: RerankingService,
        context_compressor: ContextCompressor,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
        *,
        max_history_messages: int,
        retrieval_top_k: int,
        rerank_top_k: int,
        similarity_threshold: float,
        max_context_tokens: int,
        cache: CacheService | None = None,
        response_cache_ttl_seconds: int = 300,
        guardrail_service: GuardrailService | None = None,
    ) -> None:
        """... (existing docstring, extended)

        Args:
            cache: optional CacheService for caching full RAG responses.
                If None, every message is answered fresh (no response
                caching) — keeps ConversationService usable without Redis.
            response_cache_ttl_seconds: how long a cached response for an
                identical (knowledge_base_id, query) pair remains valid.
        """
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.chunk_repo = chunk_repo
        self.document_repo = document_repo
        self.kb_service = kb_service
        self.hybrid_retriever = hybrid_retriever
        self.reranking_service = reranking_service
        self.context_compressor = context_compressor
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        self.max_history_messages = max_history_messages
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_tokens = max_context_tokens
        self._cache = cache
        self._response_cache_ttl_seconds = response_cache_ttl_seconds
        self._guardrails = guardrail_service

    async def create_conversation(
        self, *, knowledge_base_id: uuid.UUID, owner_id: uuid.UUID, title: str | None
    ) -> Conversation:
        """Start a new conversation within a knowledge base, verifying ownership."""
        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)

        conversation = Conversation(knowledge_base_id=knowledge_base_id, title=title)
        created = await self.conversation_repo.create(conversation)
        logger.info("Conversation created", extra={"conversation_id": str(created.id)})
        return created

    async def list_conversations(
        self, *, knowledge_base_id: uuid.UUID, owner_id: uuid.UUID, limit: int, offset: int
    ) -> tuple[list[Conversation], int]:
        """List conversations in a knowledge base, verifying ownership."""
        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)

        items = await self.conversation_repo.list_for_knowledge_base(
            knowledge_base_id, limit=limit, offset=offset
        )
        total = await self.conversation_repo.count_for_knowledge_base(knowledge_base_id)
        return items, total

    async def get_conversation_with_messages(
        self, *, conversation_id: uuid.UUID, knowledge_base_id: uuid.UUID, owner_id: uuid.UUID
    ) -> tuple[Conversation, list[Message]]:
        """Fetch a conversation and its full message history, verifying ownership."""
        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)
        conversation = await self._get_conversation_or_404(conversation_id, knowledge_base_id)
        messages = await self.message_repo.list_for_conversation(conversation.id)
        return conversation, messages

    async def send_message(
        self,
        *,
        conversation_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        owner_id: uuid.UUID,
        content: str,
    ) -> tuple[Message, Message, list[str]]:
        """Send a user message and generate a RAG-powered assistant reply.

        Guardrails are applied on both sides: the user's query is checked
        for prompt-injection attempts BEFORE any processing (so malicious
        input never reaches the LLM or costs us tokens), and the LLM's
        response is validated/sanitized before being persisted or returned.

        Raises:
            GuardrailViolationError: if the query fails an input guardrail
                and blocking is enabled.
            NotFoundError: if the KB or conversation doesn't exist/isn't owned.
        """
        if self._guardrails is not None:
            self._guardrails.check_input(content)

        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)
        conversation = await self._get_conversation_or_404(conversation_id, knowledge_base_id)

        history = await self._load_conversation_history(conversation.id)

        user_message = await self.message_repo.create(
            Message(conversation_id=conversation.id, role=MessageRole.USER, content=content)
        )

        cached_response = None
        cache_key = None
        if self._cache is not None and not history:
            cache_key = CacheService.build_key(
                _RESPONSE_CACHE_NAMESPACE, str(knowledge_base_id), content
            )
            cached_response = await self._cache.get(cache_key)

        if cached_response is not None:
            logger.info("RAG response cache hit", extra={"conversation_id": str(conversation.id)})
            rag_queries_total.labels(cache_status="hit").inc()
            assistant_message = await self.message_repo.create(
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.ASSISTANT,
                    content=cached_response["content"],
                )
            )
            return user_message, assistant_message, cached_response["sources"]

        rag_queries_total.labels(cache_status="miss").inc()

        chunks_with_source = await self._retrieve_context(
            knowledge_base_id=knowledge_base_id, query=content
        )

        prompt = self.prompt_builder.build(
            query=content, chunks=chunks_with_source, conversation_history=history
        )

        with tracer.start_as_current_span("rag.llm_generation"):
            with rag_pipeline_stage_duration_seconds.labels(stage="llm").time():
                llm_response = await self.llm_service.chat(prompt.messages)

        answer_text = llm_response.content
        if self._guardrails is not None:
            answer_text = self._guardrails.check_output(answer_text)

        assistant_message = await self.message_repo.create(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=answer_text,
            )
        )

        cited_filenames = sorted({item.source_filename for item in chunks_with_source})

        if self._cache is not None and cache_key is not None:
            await self._cache.set(
                cache_key,
                {"content": answer_text, "sources": cited_filenames},
                ttl_seconds=self._response_cache_ttl_seconds,
            )

        logger.info(
            "Generated conversation reply",
            extra={"conversation_id": str(conversation.id), "model": llm_response.model_name},
        )
        return user_message, assistant_message, cited_filenames
    


    async def invalidate_response_cache(self, *, knowledge_base_id: uuid.UUID) -> None:
        """Clear all cached RAG responses for a knowledge base.

        Called by DocumentService whenever a document is uploaded or
        deleted (see document_service.py changes below) — ensures users
        never get a stale cached answer that doesn't reflect their
        knowledge base's current content, without waiting for the TTL.
        """
        if self._cache is not None:
            await self._cache.delete_by_prefix(
                CacheService.build_key(_RESPONSE_CACHE_NAMESPACE, str(knowledge_base_id), "")
            )

    async def _retrieve_context(
        self, *, knowledge_base_id: uuid.UUID, query: str
    ) -> list[ChunkWithSource]:
        """Run the full retrieval pipeline (hybrid -> rerank -> compress)
        and attach source filenames for prompt building.

        Each stage is wrapped in its own OpenTelemetry span and timed as a
        Prometheus histogram, so slow stages are immediately visible in
        both traces and dashboards.
        """
        all_chunks = await self.chunk_repo.list_for_knowledge_base(knowledge_base_id)
        bm25_documents = [
            BM25Document(
                chunk_id=str(c.id),
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
            )
            for c in all_chunks
        ]

        with tracer.start_as_current_span("rag.retrieval"):
            with rag_pipeline_stage_duration_seconds.labels(stage="retrieval").time():
                candidates = await self.hybrid_retriever.retrieve(
                    knowledge_base_id=knowledge_base_id,
                    query=query,
                    bm25_documents=bm25_documents,
                    top_k=self.retrieval_top_k,
                )

        with tracer.start_as_current_span("rag.rerank"):
            with rag_pipeline_stage_duration_seconds.labels(stage="rerank").time():
                ranked = await self.reranking_service.rerank(
                    query=query, candidates=candidates, top_k=self.rerank_top_k
                )

        with tracer.start_as_current_span("rag.compress"):
            with rag_pipeline_stage_duration_seconds.labels(stage="compress").time():
                compressed = await self.context_compressor.compress(
                    ranked,
                    similarity_threshold=self.similarity_threshold,
                    max_context_tokens=self.max_context_tokens,
                )

        document_filenames = await self._resolve_document_filenames(
            {c.document_id for c in compressed}
        )
        return [
            ChunkWithSource(c, document_filenames.get(c.document_id, "unknown document"))
            for c in compressed
        ]
    

    async def _resolve_document_filenames(
        self, document_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        """Look up original filenames for a set of document ids, for citation display."""
        filenames: dict[uuid.UUID, str] = {}
        for document_id in document_ids:
            document = await self.document_repo.get_by_id(document_id)
            if document is not None:
                filenames[document_id] = document.original_filename
        return filenames

    async def _load_conversation_history(self, conversation_id: uuid.UUID) -> list[ChatMessage]:
        """Load recent conversation messages, converted into ChatMessage
        objects ready for PromptBuilder."""
        role_map = {
            MessageRole.USER: PromptMessageRole.USER,
            MessageRole.ASSISTANT: PromptMessageRole.ASSISTANT,
        }
        messages = await self.message_repo.list_for_conversation(
            conversation_id, limit=self.max_history_messages
        )
        return [ChatMessage(role=role_map[m.role], content=m.content) for m in messages]

    async def _get_conversation_or_404(
        self, conversation_id: uuid.UUID, knowledge_base_id: uuid.UUID
    ) -> Conversation:
        """Shared helper: fetch a conversation scoped to its KB, or raise 404."""
        conversation = await self.conversation_repo.get_for_knowledge_base(
            conversation_id, knowledge_base_id
        )
        if conversation is None:
            raise NotFoundError("Conversation not found")
        return conversation