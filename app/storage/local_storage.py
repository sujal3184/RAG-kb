"""Local disk file storage implementation.

Stores files on the local filesystem — suitable for development and
single-server deployments. In Docker, `LOCAL_STORAGE_PATH` should point to
a mounted volume so files survive container restarts (wired in Module 2's
docker-compose, extended below).

Swapping to a cloud provider (S3, GCS, Azure Blob) later means writing a
new class implementing `FileStorage` — `DocumentService` never changes.
"""

import asyncio
import logging
from pathlib import Path

from app.storage.base import FileStorage

logger = logging.getLogger(__name__)


class LocalFileStorage(FileStorage):
    """Stores files under a base directory on the local filesystem."""

    def __init__(self, base_path: str) -> None:
        """Set up the storage root directory, creating it if needed.

        Args:
            base_path: the root directory under which all files are stored.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save(self, *, path: str, content: bytes) -> str:
        """Write file content to disk under `base_path / path`.

        File I/O is blocking, so we run it in a thread (`asyncio.to_thread`)
        to avoid freezing the async event loop while writing large files.
        """
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(full_path.write_bytes, content)
        logger.info("File saved to local storage", extra={"path": str(full_path)})
        return str(full_path)

    async def read(self, *, storage_ref: str) -> bytes:
        """Read file content back from disk."""
        full_path = Path(storage_ref)
        return await asyncio.to_thread(full_path.read_bytes)

    async def delete(self, *, storage_ref: str) -> None:
        """Delete a file from disk, ignoring if it's already gone."""
        full_path = Path(storage_ref)
        if full_path.exists():
            await asyncio.to_thread(full_path.unlink)
            logger.info("File deleted from local storage", extra={"path": str(full_path)})