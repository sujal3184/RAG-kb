"""Scaffold an evaluation dataset from a knowledge base.

Writing ground-truth chunk IDs by hand is impractical — you'd have to
query the database for every question. This script prints all chunks in a
knowledge base with their IDs and text preview, so you can write
questions and identify which chunks answer them.

Usage:
    uv run python -m scripts.build_eval_dataset <knowledge_base_id>
"""

import asyncio
import sys
import uuid

from app.db.session import async_session_factory
from app.repositories.chunk_repository import ChunkRepository


async def list_chunks(knowledge_base_id: uuid.UUID) -> None:
    """Print every chunk in a knowledge base with its ID and a preview."""
    async with async_session_factory() as session:
        repo = ChunkRepository(session)
        chunks = await repo.list_for_knowledge_base(knowledge_base_id)

        if not chunks:
            print(f"No chunks found for knowledge base {knowledge_base_id}.")
            print("Has a document been uploaded and processed to 'ready' status?")
            return

        print(f"\n{len(chunks)} chunks in knowledge base {knowledge_base_id}:\n")
        for chunk in sorted(chunks, key=lambda c: (str(c.document_id), c.chunk_index)):
            preview = chunk.text[:150].replace("\n", " ")
            print(f"chunk_id: {chunk.id}")
            print(f"  doc: {chunk.document_id}  index: {chunk.chunk_index}")
            print(f"  text: {preview}...")
            print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python -m scripts.build_eval_dataset <knowledge_base_id>")
        sys.exit(1)

    asyncio.run(list_chunks(uuid.UUID(sys.argv[1])))