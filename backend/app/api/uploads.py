"""Helpers for reading uploads with a hard size cap."""

from fastapi import UploadFile

from app.services.image_service import ImageValidationError


def _too_large(max_bytes: int) -> ImageValidationError:
    return ImageValidationError(
        f"File is too large (maximum is {max_bytes / (1024 * 1024):g} MB).", status_code=413
    )


async def read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read at most `max_bytes` (+1 to detect overflow) without buffering oversized bodies."""
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise _too_large(max_bytes)
    return data


def read_limited_sync(file: UploadFile, max_bytes: int) -> bytes:
    """Same as `read_limited` for synchronous endpoints (runs in FastAPI's threadpool)."""
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise _too_large(max_bytes)
    return data
