"""Shared handling of user uploads: sanitised display names and all-or-nothing validation."""

import os
import re
from dataclasses import dataclass

from app.services.image_service import FORMAT_EXTENSIONS, ImageService, ImageValidationError

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    data: bytes


@dataclass(frozen=True)
class PreparedUpload:
    file: UploadedFile
    display_name: str
    extension: str
    width: int
    height: int
    mime_type: str


def display_filename(raw: str | None) -> str:
    """Sanitised, display-only filename (basename, no control chars, max 255 chars)."""
    name = os.path.basename((raw or "").replace("\\", "/"))
    name = _CONTROL_CHARS.sub("", name).strip()
    return (name or "image")[:255]


def prepare_uploads(images: ImageService, files: list[UploadedFile]) -> list[PreparedUpload]:
    """Validate every file before anything is written; the first failure aborts the batch."""
    prepared: list[PreparedUpload] = []
    for item in files:
        name = display_filename(item.filename)
        try:
            validated = images.validate(item.data)
        except ImageValidationError as exc:
            raise ImageValidationError(f"{name}: {exc.message}", exc.status_code) from None
        # Keep only metadata; drop decoded pixels so large batches stay light.
        prepared.append(
            PreparedUpload(
                file=item,
                display_name=name,
                extension=FORMAT_EXTENSIONS[validated.format],
                width=validated.width,
                height=validated.height,
                mime_type=validated.mime_type,
            )
        )
        validated.image.close()
    return prepared
