"""Opt-in test that runs the real YOLO model (downloads weights on first use).

    $env:RUN_YOLO_INTEGRATION = "1"; pytest tests/test_yolo_integration.py
"""

import os

import pytest

from app.providers.yolo_provider import YoloProvider
from app.services.detection_service import DetectionService
from app.services.image_service import ImageService
from tests.conftest import make_image_bytes

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_YOLO_INTEGRATION") != "1",
    reason="set RUN_YOLO_INTEGRATION=1 to run the real YOLO model",
)


def test_real_model_runs_on_cpu_and_handles_blank_image() -> None:
    provider = YoloProvider("yolo11n.pt", device="cpu")
    service = DetectionService(provider, 0.25, 0.45)
    image = ImageService(10 * 1024 * 1024, 320, 8000).validate(make_image_bytes("PNG", (640, 480)))

    result = service.detect(image, "img_001")

    assert (result.width, result.height) == (640, 480)
    assert result.detections == []  # a flat-colour image has nothing to detect
    assert "chair" in provider.class_names and len(provider.class_names) == 80
