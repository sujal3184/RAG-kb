"""Loader for HTML files (.html)."""

from bs4 import BeautifulSoup

from app.loaders.base import DocumentLoader, LoadedDocument
from app.loaders.exceptions import LoaderError


class HtmlLoader(DocumentLoader):
    """Loads .html files, extracting visible text and stripping markup."""

    def load(self, content: bytes) -> LoadedDocument:
        """Parse HTML and return only the human-readable text content.

        Raises:
            LoaderError: if the content can't be decoded/parsed as HTML at all.
        """
        try:
            decoded = content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(decoded, "lxml")
        except Exception as exc:
            raise LoaderError(f"Could not parse HTML content: {exc}") from exc

        # Extract the title BEFORE removing <head> — <title> lives inside
        # <head>, so reading it after decomposing head would always be None.
        title = soup.title.string.strip() if soup.title and soup.title.string else None

        # Now remove elements that never contain meaningful readable content.
        for tag in soup(["script", "style", "noscript", "head"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        return LoadedDocument(
            text=text,
            metadata={"title": title} if title else {},
        )