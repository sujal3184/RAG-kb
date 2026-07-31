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

logger = logging.getLogger(__name__)


class ConversationService:
    """Manages conversations and generates RAG-powered replies."""

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
    ) -> None:
        """Store all dependencies needed to manage conversations and
        generate answers. This is intentionally a wide constructor — it
        reflects ConversationService's role as the orchestrator tying
        together every retrieval-pipeline module built so far."""
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

        This is the core use case tying together every retrieval-pipeline
        module: persist the user's message, run hybrid retrieval + rerank
        + compression over the knowledge base, build a prompt including
        recent conversation history, generate a response, and persist the
        assistant's reply.

        Args:
            conversation_id: which conversation this message belongs to.
            knowledge_base_id: which knowledge base to search for context.
            owner_id: the current user — verified as the KB's owner.
            content: the user's message text.

        Returns:
            A tuple of (user_message, assistant_message, cited_filenames).

        Raises:
            NotFoundError: if the KB or conversation doesn't exist / isn't owned.
        """
        await self.kb_service.get(kb_id=knowledge_base_id, owner_id=owner_id)
        conversation = await self._get_conversation_or_404(conversation_id, knowledge_base_id)

        user_message = await self.message_repo.create(
            Message(conversation_id=conversation.id, role=MessageRole.USER, content=content)
        )

        history = await self._load_conversation_history(conversation.id)

        chunks_with_source = await self._retrieve_context(
            knowledge_base_id=knowledge_base_id, query=content
        )

        prompt = self.prompt_builder.build(
            query=content, chunks=chunks_with_source, conversation_history=history
        )

        llm_response = await self.llm_service.chat(prompt.messages)

        assistant_message = await self.message_repo.create(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=llm_response.content,
            )
        )

        cited_filenames = sorted({item.source_filename for item in chunks_with_source})

        logger.info(
            "Generated conversation reply",
            extra={"conversation_id": str(conversation.id), "model": llm_response.model_name},
        )
        return user_message, assistant_message, cited_filenames

    async def _retrieve_context(
        self, *, knowledge_base_id: uuid.UUID, query: str
    ) -> list[ChunkWithSource]:
        """Run the full retrieval pipeline (hybrid -> rerank -> compress)
        and attach source filenames for prompt building."""
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

        candidates = await self.hybrid_retriever.retrieve(
            knowledge_base_id=knowledge_base_id,
            query=query,
            bm25_documents=bm25_documents,
            top_k=self.retrieval_top_k,
        )
        ranked = await self.reranking_service.rerank(
            query=query, candidates=candidates, top_k=self.rerank_top_k
        )
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