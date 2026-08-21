"""AWS S3 (or S3-compatible) object storage.

Same FileStorage interface as LocalFileStorage (Module 6), so swapping
between them is a one-line change in dependency wiring — no other code
needs to know or care which storage backend is active.

For real AWS S3, credentials come automatically from the ECS task's IAM
role — nothing to configure in application code or settings. For
S3-compatible providers other than AWS (e.g. Cloudflare R2), pass an
endpoint_url and explicit credentials instead.
"""

import asyncio
import logging

import boto3
from botocore.exceptions import ClientError

from app.storage.base import FileStorage

logger = logging.getLogger(__name__)


class S3CompatibleStorage(FileStorage):
    """Stores files in an S3 (or S3-compatible) bucket."""

    def __init__(
        self,
        *,
        bucket_name: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        """Configure the S3 client.

        Args:
            bucket_name: the bucket to store files in.
            endpoint_url: only needed for non-AWS S3-compatible providers
                (e.g. Cloudflare R2). Leave None for real AWS S3.
            access_key: only needed for non-AWS providers, or when not
                running with an IAM role attached. Leave None on ECS —
                credentials are picked up automatically from the task role.
            secret_key: paired with access_key; same rule applies.
        """
        client_kwargs: dict[str, str] = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key

        self._client = boto3.client("s3", **client_kwargs)
        self._bucket_name = bucket_name

    async def save(self, *, path: str, content: bytes) -> str:
        """Upload file content to the bucket, keyed by path."""
        await asyncio.to_thread(
            self._client.put_object, Bucket=self._bucket_name, Key=path, Body=content
        )
        logger.info("File uploaded to S3", extra={"key": path})
        return path

    async def read(self, *, storage_ref: str) -> bytes:
        """Download file content from the bucket."""
        response = await asyncio.to_thread(
            self._client.get_object, Bucket=self._bucket_name, Key=storage_ref
        )
        return response["Body"].read()

    async def delete(self, *, storage_ref: str) -> None:
        """Delete a file from the bucket, ignoring if already gone."""
        try:
            await asyncio.to_thread(
                self._client.delete_object, Bucket=self._bucket_name, Key=storage_ref
            )
            logger.info("File deleted from S3", extra={"key": storage_ref})
        except ClientError as exc:
            logger.warning("Failed to delete from S3", extra={"error": str(exc)})