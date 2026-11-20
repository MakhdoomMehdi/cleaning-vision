"""Local-disk image storage with generated filenames and path-traversal protection."""

import logging
import re
import uuid
from pathlib import Path

from app.storage.base import StorageError

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = frozenset({".jpg", ".png", ".webp"})
_CATEGORY_RE = re.compile(r"^[a-z][a-z_]{0,31}$")
_KEY_RE = re.compile(r"^[a-z][a-z_]{0,31}/[0-9a-f]{32}\.(jpg|png|webp)$")


class LocalImageStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, *, category: str, extension: str) -> str:
        if not _CATEGORY_RE.match(category):
            raise StorageError("Invalid storage category.")
        if extension not in ALLOWED_EXTENSIONS:
            raise StorageError("Unsupported file extension.")

        # The filename is entirely server-generated; no user input reaches the path.
        key = f"{category}/{uuid.uuid4().hex}{extension}"
        path = self._resolve(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:  # "x": never overwrite an existing file
                handle.write(data)
        except OSError as exc:
            logger.exception("Failed to write image to storage")
            raise StorageError("Could not store image.") from exc
        return key

    def read(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise StorageError("Stored image not found.") from None
        except OSError as exc:
            logger.exception("Failed to read image from storage")
            raise StorageError("Could not read stored image.") from exc

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete stored image %s", key)

    def _resolve(self, key: str) -> Path:
        """Map a key to a path, refusing anything that is not a well-formed key under the root."""
        if not _KEY_RE.match(key):
            raise StorageError("Invalid storage key.")
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise StorageError("Invalid storage key.")
        return path
