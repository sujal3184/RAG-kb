"""Loader factory — selects the right DocumentLoader for a file extension.

This is the ONE place that maps "file extension" to "which loader class
handles it". Adding support for a new file type later means writing one
new loader class and adding one line here — nothing else in the app
needs to change.
"""

from app.loaders.base import DocumentLoader
from app.loaders.csv_loader import CsvLoader
from app.loaders.docx_loader import DocxLoader
from app.loaders.exceptions import LoaderError
from app.loaders.html_loader import HtmlLoader
from app.loaders.pdf_loader import PdfLoader
from app.loaders.text_loader import TextLoader
from app.loaders.advanced_pdf_loader import AdvancedPdfLoader


class LoaderFactory:
    """Creates the appropriate DocumentLoader for a given file extension."""

    _LOADERS: dict[str, type[DocumentLoader]] = {
        "txt": TextLoader,
        "md": TextLoader,
        "pdf": AdvancedPdfLoader,
        "docx": DocxLoader,
        "html": HtmlLoader,
        "csv": CsvLoader,
    }

    @classmethod
    def get_loader(cls, extension: str) -> DocumentLoader:
        """Return a loader instance for the given file extension.

        Args:
            extension: file extension without the leading dot (e.g. "pdf"),
                case-insensitive.

        Returns:
            An instance of the loader class registered for that extension.

        Raises:
            LoaderError: if no loader is registered for this extension.
        """
        loader_class = cls._LOADERS.get(extension.lower())
        if loader_class is None:
            supported = ", ".join(sorted(cls._LOADERS.keys()))
            raise LoaderError(
                f"No loader available for file type '.{extension}'. "
                f"Supported types: {supported}"
            )
        return loader_class()