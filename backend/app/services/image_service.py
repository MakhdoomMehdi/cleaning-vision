"""Upload validation based on actually decoded image content (never the file extension)."""

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Pillow format name -> MIME type
ALLOWED_FORMATS: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


FORMAT_EXTENSIONS: dict[str, str] = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class ImageValidationError(Exception):
    """Raised for rejected uploads. `status_code` is the suggested HTTP status."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ValidatedImage:
    """Decoded RGB image with EXIF orientation applied.

    `width`/`height` are the dimensions every bounding box refers to. Because the
    orientation is applied here, they match what a browser displays for the file.
    """

    image: Image.Image
    width: int
    height: int
    format: str
    mime_type: str
    size_bytes: int


class ImageService:
    def __init__(
        self,
        max_upload_bytes: int,
        min_dimension: int,
        max_dimension: int,
    ) -> None:
        self._max_upload_bytes = max_upload_bytes
        self._min_dimension = min_dimension
        self._max_dimension = max_dimension

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    def validate(self, data: bytes) -> ValidatedImage:
        size = len(data)
        if size == 0:
            raise ImageValidationError("Uploaded file is empty.")
        if size > self._max_upload_bytes:
            limit_mb = self._max_upload_bytes / (1024 * 1024)
            raise ImageValidationError(
                f"File is too large (maximum is {limit_mb:g} MB).", status_code=413
            )

        try:
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format or ""
                width, height = probe.size  # header only; no full decode yet
        except (UnidentifiedImageError, OSError):
            raise ImageValidationError(
                "File is not a valid image. Supported types: JPEG, PNG, WebP.",
                status_code=415,
            ) from None

        if fmt not in ALLOWED_FORMATS:
            raise ImageValidationError(
                f"Unsupported image type '{fmt or 'unknown'}'. Supported types: JPEG, PNG, WebP.",
                status_code=415,
            )

        # Check dimensions from the header before decoding pixels (decompression-bomb guard).
        self._check_dimensions(width, height)

        try:
            with Image.open(io.BytesIO(data)) as decoded:
                decoded.load()  # full decode: catches truncated/corrupt files
                oriented = ImageOps.exif_transpose(decoded)
                rgb = oriented.convert("RGB")
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
            logger.info("Rejected upload: image failed to decode (format=%s)", fmt)
            raise ImageValidationError("Image is corrupt or could not be decoded.") from None

        out_width, out_height = rgb.size
        # EXIF rotation can swap the sides, so check again on the final image.
        self._check_dimensions(out_width, out_height)

        return ValidatedImage(
            image=rgb,
            width=out_width,
            height=out_height,
            format=fmt,
            mime_type=ALLOWED_FORMATS[fmt],
            size_bytes=size,
        )

    def _check_dimensions(self, width: int, height: int) -> None:
        if min(width, height) < self._min_dimension:
            raise ImageValidationError(
                f"Image is too small ({width}x{height}); each side must be at least "
                f"{self._min_dimension}px."
            )
        if max(width, height) > self._max_dimension:
            raise ImageValidationError(
                f"Image is too large ({width}x{height}); each side must be at most "
                f"{self._max_dimension}px."
            )
