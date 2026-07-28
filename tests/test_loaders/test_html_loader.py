"""Tests for HtmlLoader."""

from app.loaders.html_loader import HtmlLoader


def test_extracts_visible_text_and_strips_scripts(sample_html_bytes: bytes) -> None:
    loader = HtmlLoader()
    result = loader.load(sample_html_bytes)

    assert "Main Heading" in result.text
    assert "This is a paragraph about knowledge bases." in result.text
    assert "console.log" not in result.text  # script content removed
    assert "color: red" not in result.text  # style content removed
    assert result.metadata["title"] == "Test Page"