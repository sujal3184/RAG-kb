"""Tests for BM25Store — pure computation, no external services needed."""

import uuid

from app.retrieval.bm25_store import BM25Document, BM25Store


def _make_doc(text: str, chunk_index: int = 0) -> BM25Document:
    return BM25Document(
        chunk_id=str(uuid.uuid4()), document_id=uuid.uuid4(), chunk_index=chunk_index, text=text
    )


def test_finds_exact_keyword_match() -> None:
    store = BM25Store()
    documents = [
        _make_doc("The product code SKU-88213 is out of stock."),
        _make_doc("Our return policy allows refunds within 30 days."),
        _make_doc("We ship internationally to most countries."),
    ]

    results = store.search(documents=documents, query="SKU-88213", top_k=5)

    assert len(results) == 1
    assert "SKU-88213" in results[0].text

def test_ranks_more_relevant_documents_higher() -> None:
    store = BM25Store()
    documents = [
        _make_doc("Cats are popular pets known for independence."),
        _make_doc("The stock market fell sharply due to inflation fears."),
        _make_doc("Many people keep cats as pets because cats are low maintenance."),
    ]

    results = store.search(documents=documents, query="cats pets", top_k=5)

    assert len(results) >= 2
    assert "cats" in results[0].text.lower()


def test_returns_empty_list_for_no_documents() -> None:
    store = BM25Store()
    assert store.search(documents=[], query="anything", top_k=5) == []


def test_excludes_zero_relevance_documents() -> None:
    store = BM25Store()
    documents = [
        _make_doc("This document is about gardening and plants."),
    ]

    results = store.search(documents=documents, query="quantum physics", top_k=5)

    assert results == []