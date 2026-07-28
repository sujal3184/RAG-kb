"""Document loader interface.

A `DocumentLoader` takes raw file bytes and extracts plain text plus
useful metadata. This is a pure transformation — no file I/O, no database
access — which keeps loaders simple, fast, and trivially unit-testable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedDocument:
    """The result of loading a document: extracted text plus metadata.

    Attributes:
        text: the full extracted plain text content.
        metadata: format-specific extra info (e.g. {"page_count": 12} for
            a PDF, {"row_count": 500} for a CSV). Chunking (Module 8) and
            search results can use this later to show richer context.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(ABC):
    """Abstract base class for extracting text from a specific file format."""

    @abstractmethod
    def load(self, content: bytes) -> LoadedDocument:
        """Extract plain text and metadata from raw file bytes.

        Args:
            content: the raw bytes of the file (e.g. read via FileStorage).

        Returns:
            A LoadedDocument containing the extracted text and metadata.

        Raises:
            LoaderError: if the content cannot be parsed (corrupted file,
                unsupported internal format, etc.).
        """
        raise NotImplementedError