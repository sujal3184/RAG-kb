"""Loader for PDF files (.pdf).

Extracts text page-by-page using pypdf. Note: this does NOT perform OCR —
scanned/image-only PDFs (with no embedded text layer) will extract as
empty or near-empty text. OCR support is a noted future extension.
"""

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.loaders.base import DocumentLoader, LoadedDocument
from app.loaders.exceptions import LoaderError


class PdfLoader(DocumentLoader):
    """Loads .pdf files, extracting text from each page."""

    def load(self, content: bytes) -> LoadedDocument:
        """Extract text from all pages of a PDF, joined with page breaks.

        Raises:
            LoaderError: if the PDF is corrupted, encrypted without a
                password, or otherwise unreadable.
        """
        try:
            reader = PdfReader(io.BytesIO(content))
        except PdfReadError as exc:
            raise LoaderError(f"Could not open PDF: {exc}") from exc

        if reader.is_encrypted:
            raise LoaderError("PDF is password-protected and cannot be read")

        page_texts: list[str] = []
        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception as exc:  # pypdf can raise various internal errors per-page
                raise LoaderError(f"Failed to extract text from a PDF page: {exc}") from exc

        full_text = "\n\n".join(page_texts)

        return LoadedDocument(
            text=full_text,
            metadata={"page_count": len(reader.pages)},
        )