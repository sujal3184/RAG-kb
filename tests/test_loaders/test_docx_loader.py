"""Tests for DocxLoader."""

import pytest

from app.loaders.docx_loader import DocxLoader
from app.loaders.exceptions import LoaderError


def test_loads_paragraphs_and_tables(sample_docx_bytes: bytes) -> None:
    loader = DocxLoader()
    result = loader.load(sample_docx_bytes)

    assert "This is a test paragraph about RAG systems." in result.text
    assert "Key | Value" in result.text
    assert result.metadata["paragraph_count"] == 1
    assert result.metadata["table_count"] == 1


def test_rejects_invalid_docx() -> None:
    loader = DocxLoader()
    garbage_bytes = b"this is not a docx file"

    with pytest.raises(LoaderError):
        loader.load(garbage_bytes)