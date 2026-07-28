"""Loader-specific errors.

Wraps whatever underlying library errors occur (pypdf, python-docx,
BeautifulSoup, etc.) into one consistent error type, so calling code never
needs to know or catch library-specific exceptions.
"""

from app.core.exceptions import AppException


class LoaderError(AppException):
    """Raised when a document's content cannot be parsed/extracted."""