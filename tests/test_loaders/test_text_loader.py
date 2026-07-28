"""Tests for TextLoader."""

import pytest

from app.loaders.exceptions import LoaderError
from app.loaders.text_loader import TextLoader


def test_loads_plain_text(sample_txt_bytes: bytes) -> None:
    loader = TextLoader()
    result = loader.load(sample_txt_bytes)

    assert "Hello world." in result.text
    assert result.metadata["character_count"] == len(result.text)


def test_rejects_invalid_utf8() -> None:
    loader = TextLoader()
    invalid_bytes = b"\xff\xfe\x00\x01invalid"

    with pytest.raises(LoaderError):
        loader.load(invalid_bytes)