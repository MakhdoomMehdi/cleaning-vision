"""Storage interface. Callers only ever hold opaque *keys*, never filesystem paths,
so the local-disk implementation can later be swapped for S3 without touching them."""

from typing import Protocol


class StorageError(Exception):
    """Storage operation failed or the key is invalid/unsafe."""


class ImageStorage(Protocol):
    def save(self, data: bytes, *, category: str, extension: str) -> str:
        """Persist bytes under a server-generated name and return its key."""
        ...

    def read(self, key: str) -> bytes:
        """Return the stored bytes. Raises StorageError if missing or the key is unsafe."""
        ...

    def delete(self, key: str) -> None:
        """Remove the stored object. Missing keys are ignored."""
        ...
