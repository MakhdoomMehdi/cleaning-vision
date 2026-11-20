import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import get_image_service, get_storage, get_vision_provider
from app.core import config
from app.db.session import reset_db_caches
from app.main import create_app
from app.services.image_service import ImageService


def make_image_bytes(
    fmt: str = "JPEG",
    size: tuple[int, int] = (640, 480),
    exif_orientation: int | None = None,
    noise: bool = False,
) -> bytes:
    """Build an in-memory image. `noise=True` makes it hard to compress (for size tests)."""
    if noise:
        array = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        image = Image.fromarray(array)
    else:
        image = Image.new("RGB", size, (120, 160, 200))
    buffer = io.BytesIO()
    kwargs: dict[str, object] = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def _clear_caches() -> None:
    config.get_settings.cache_clear()
    reset_db_caches()
    get_storage.cache_clear()
    get_vision_provider.cache_clear()


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Every test gets its own SQLite file and upload dir; real code paths, no shared state."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    _clear_caches()
    yield tmp_path / "uploads"
    _clear_caches()


@pytest.fixture
def image_service() -> ImageService:
    return ImageService(max_upload_bytes=1024 * 1024, min_dimension=320, max_dimension=4000)


@pytest.fixture
def client(image_service: ImageService) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_image_service] = lambda: image_service
    with TestClient(app) as test_client:
        yield test_client
