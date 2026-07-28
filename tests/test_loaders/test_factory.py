"""Tests for LoaderFactory."""

import pytest

from app.loaders.csv_loader import CsvLoader
from app.loaders.docx_loader import DocxLoader
from app.loaders.exceptions import LoaderError
from app.loaders.factory import LoaderFactory
from app.loaders.html_loader import HtmlLoader
from app.loaders.pdf_loader import PdfLoader
from app.loaders.text_loader import TextLoader
from app.loaders.advanced_pdf_loader import AdvancedPdfLoader



@pytest.mark.parametrize(
    ("extension", "expected_type"),
    [
        ("txt", TextLoader),
        ("md", TextLoader),
        ("pdf", AdvancedPdfLoader),   # CHANGED
        ("docx", DocxLoader),
        ("html", HtmlLoader),
        ("csv", CsvLoader),
        ("PDF", AdvancedPdfLoader),   # CHANGED
    ],
)

def test_returns_correct_loader_for_extension(extension: str, expected_type: type) -> None:
    loader = LoaderFactory.get_loader(extension)
    assert isinstance(loader, expected_type)


def test_raises_for_unsupported_extension() -> None:
    with pytest.raises(LoaderError):
        LoaderFactory.get_loader("exe")