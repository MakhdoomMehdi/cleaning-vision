import pytest

from app.services.image_service import ImageService, ImageValidationError
from tests.conftest import make_image_bytes


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_valid_formats_accepted(image_service: ImageService, fmt: str) -> None:
    result = image_service.validate(make_image_bytes(fmt, (640, 480)))
    assert (result.width, result.height) == (640, 480)
    assert result.format == fmt
    assert result.image.mode == "RGB"


def test_extension_is_irrelevant_content_decides(image_service: ImageService) -> None:
    with pytest.raises(ImageValidationError) as exc:
        image_service.validate(b"%PDF-1.4 definitely not an image")
    assert exc.value.status_code == 415


def test_unsupported_but_decodable_format_rejected(image_service: ImageService) -> None:
    with pytest.raises(ImageValidationError) as exc:
        image_service.validate(make_image_bytes("GIF", (640, 480)))
    assert exc.value.status_code == 415


def test_empty_file_rejected(image_service: ImageService) -> None:
    with pytest.raises(ImageValidationError):
        image_service.validate(b"")


def test_oversized_upload_rejected(image_service: ImageService) -> None:
    data = make_image_bytes("PNG", (800, 800), noise=True)
    assert len(data) > image_service.max_upload_bytes
    with pytest.raises(ImageValidationError) as exc:
        image_service.validate(data)
    assert exc.value.status_code == 413


def test_too_small_dimensions_rejected(image_service: ImageService) -> None:
    with pytest.raises(ImageValidationError, match="too small"):
        image_service.validate(make_image_bytes("PNG", (200, 200)))


def test_too_large_dimensions_rejected(image_service: ImageService) -> None:
    with pytest.raises(ImageValidationError, match="too large"):
        image_service.validate(make_image_bytes("PNG", (4100, 400)))


def test_truncated_image_rejected(image_service: ImageService) -> None:
    data = make_image_bytes("PNG", (640, 480), noise=True)
    with pytest.raises(ImageValidationError):
        image_service.validate(data[: len(data) // 2])


def test_exif_orientation_applied_so_boxes_match_browser_display(
    image_service: ImageService,
) -> None:
    # Stored as 640x480 landscape with EXIF "rotate 90 CW" -> displayed as 480x640 portrait.
    # Dimensions (min side must be >= 320) are fine either way.
    data = make_image_bytes("JPEG", (640, 480), exif_orientation=6)
    result = image_service.validate(data)
    assert (result.width, result.height) == (480, 640)
