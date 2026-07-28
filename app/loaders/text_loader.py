"""Loader for plain text and Markdown files (.txt, .md).

These formats are already plain text, so "loading" is mostly just
decoding bytes into a string correctly.
"""

from app.loaders.base import DocumentLoader, LoadedDocument
from app.loaders.exceptions import LoaderError


class TextLoader(DocumentLoader):
    """Loads .txt and .md files by decoding them as UTF-8 text."""

    def load(self, content: bytes) -> LoadedDocument:
        """Decode raw bytes as UTF-8 text.

        Raises:
            LoaderError: if the content isn't valid UTF-8 text (e.g. it's
                actually a binary file mislabeled with a .txt extension).
        """
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LoaderError(
                "File is not valid UTF-8 text — it may be corrupted or "
                "not actually a text file"
            ) from exc

        return LoadedDocument(
            text=text,
            metadata={"character_count": len(text)},
        )