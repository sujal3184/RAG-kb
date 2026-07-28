"""Tests for PdfLoader."""

import pytest

from app.loaders.exceptions import LoaderError
from app.loaders.pdf_loader import PdfLoader


def test_loads_valid_pdf_structure(sample_pdf_bytes: bytes) -> None:
    loader = PdfLoader()
    result = loader.load(sample_pdf_bytes)

    assert result.metadata["page_count"] == 1
    assert isinstance(result.text, str)


def test_rejects_corrupted_pdf() -> None:
    loader = PdfLoader()
    garbage_bytes = b"this is not a pdf file at all"

    with pytest.raises(LoaderError):
        loader.load(garbage_bytes)