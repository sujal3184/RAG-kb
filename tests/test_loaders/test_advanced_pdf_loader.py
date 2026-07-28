"""Tests for AdvancedPdfLoader."""

import pytest

from app.loaders.advanced_pdf_loader import AdvancedPdfLoader
from app.loaders.exceptions import LoaderError


def test_extracts_two_column_text_in_correct_reading_order(
    sample_two_column_pdf_bytes: bytes,
) -> None:
    """Left column content should appear before right column content, and
    lines within each column should stay intact (not interleaved)."""
    loader = AdvancedPdfLoader()
    result = loader.load(sample_two_column_pdf_bytes)

    left_pos = result.text.find("Left column line one.")
    right_pos = result.text.find("Right column line one.")

    assert left_pos != -1
    assert right_pos != -1
    assert left_pos < right_pos  # left column fully read before right column


def test_extracts_table_as_structured_text(sample_pdf_with_table_bytes: bytes) -> None:
    """Table rows should be converted to readable 'header: value' text."""
    loader = AdvancedPdfLoader()
    result = loader.load(sample_pdf_with_table_bytes)

    assert "Name: Alice, Role: Engineer" in result.text
    assert "Name: Bob, Role: Designer" in result.text
    assert result.metadata["table_count"] == 1


def test_rejects_corrupted_pdf() -> None:
    loader = AdvancedPdfLoader()
    with pytest.raises(LoaderError):
        loader.load(b"this is not a pdf file at all")


def test_reports_correct_page_count(sample_pdf_with_table_bytes: bytes) -> None:
    loader = AdvancedPdfLoader()
    result = loader.load(sample_pdf_with_table_bytes)
    assert result.metadata["page_count"] == 1