"""File storage interface.

Defines the CONTRACT for storing uploaded files, without committing to any
specific backend. `DocumentService` depends only on this interface, never
on a concrete implementation — swapping `LocalFileStorage` for an S3/GCS
implementation later means writing one new class, and changing one line
in `api/dependencies.py`.
"""

from abc import ABC, abstractmethod


class FileStorage(ABC):
    """Abstract base class for anything that can store and retrieve files."""

    @abstractmethod
    async def save(self, *, path: str, content: bytes) -> str:
        """Save file content at the given logical path.

        Args:
            path: a logical path/key for the file (e.g.
                "owner_id/kb_id/doc_id_filename.pdf"), NOT necessarily a
                real filesystem path — implementations decide how to map
                this to their actual storage.
            content: the raw file bytes.

        Returns:
            A storage reference (e.g. a full file path or object key) that
            can later be passed to `read` or `delete`.
        """
        raise NotImplementedError

    @abstractmethod
    async def read(self, *, storage_ref: str) -> bytes:
        """Read back the full content of a previously saved file.

        Args:
            storage_ref: the reference returned by `save`.

        Returns:
            The raw file bytes.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, *, storage_ref: str) -> None:
        """Delete a previously saved file.

        Args:
            storage_ref: the reference returned by `save`.
        """
        raise NotImplementedError