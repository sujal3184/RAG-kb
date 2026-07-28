"""Tests for CsvLoader."""

import pytest

from app.loaders.csv_loader import CsvLoader
from app.loaders.exceptions import LoaderError


def test_converts_rows_to_readable_text(sample_csv_bytes: bytes) -> None:
    loader = CsvLoader()
    result = loader.load(sample_csv_bytes)

    assert "name: Alice, role: Engineer" in result.text
    assert "name: Bob, role: Designer" in result.text
    assert result.metadata["row_count"] == 2
    assert result.metadata["column_names"] == ["name", "role"]


def test_rejects_empty_csv() -> None:
    loader = CsvLoader()
    empty_csv = "name,role\n".encode("utf-8")  # header only, no data rows

    with pytest.raises(LoaderError):
        loader.load(empty_csv)