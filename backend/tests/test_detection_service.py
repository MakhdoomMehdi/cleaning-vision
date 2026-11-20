import pytest
from pydantic import ValidationError

from app.providers.yolo_provider import DetectionProviderError, RawDetection
from app.schemas.detection import Detection, ImageDetections
from app.services.detection_service import DetectionService
from app.services.image_service import ImageService
from tests.conftest import FakeProvider, make_image_bytes


@pytest.fixture
def validated(image_service: ImageService):
    return image_service.validate(make_image_bytes("JPEG", (640, 480)))


def make_service(provider: FakeProvider) -> DetectionService:
    return DetectionService(provider, default_conf_threshold=0.25, default_iou_threshold=0.45)


def test_empty_detections(validated) -> None:
    result = make_service(FakeProvider([])).detect(validated, "img_001")
    assert result.detections == []
    assert (result.width, result.height) == (640, 480)


def test_valid_boxes_get_ids_and_are_sorted_by_confidence(validated) -> None:
    provider = FakeProvider(
        [
            RawDetection("chair", 0.60, (10, 20, 110, 220)),
            RawDetection("couch", 0.90, (200, 100, 600, 400)),
        ]
    )
    result = make_service(provider).detect(validated, "img_001")

    assert [d.id for d in result.detections] == ["det_001", "det_002"]
    assert [d.label for d in result.detections] == ["couch", "chair"]
    assert result.detections[0].bbox == (200.0, 100.0, 600.0, 400.0)
    assert result.detections[0].confidence == 0.9  # raw value, not altered


def test_boxes_slightly_outside_image_are_clamped(validated) -> None:
    provider = FakeProvider([RawDetection("tv", 0.8, (-5.0, -3.0, 645.0, 490.0))])
    box = make_service(provider).detect(validated, "img_001").detections[0].bbox
    assert box == (0.0, 0.0, 640.0, 480.0)


def test_degenerate_and_invalid_detections_are_dropped(validated) -> None:
    provider = FakeProvider(
        [
            RawDetection("chair", 0.8, (100, 100, 100, 200)),  # zero width
            RawDetection("chair", 0.8, (700, 10, 800, 90)),  # fully outside -> zero area after clamp
            RawDetection("chair", 1.5, (10, 10, 90, 90)),  # impossible confidence
            RawDetection("chair", 0.1, (10, 10, 90, 90)),  # below threshold
            RawDetection("chair", 0.7, (10, 10, 90, 90)),  # the only valid one
        ]
    )
    result = make_service(provider).detect(validated, "img_001")
    assert len(result.detections) == 1
    assert result.detections[0].confidence == 0.7


def test_swapped_corners_are_normalized(validated) -> None:
    provider = FakeProvider([RawDetection("chair", 0.8, (200, 300, 100, 100))])
    assert make_service(provider).detect(validated, "img_001").detections[0].bbox == (
        100.0,
        100.0,
        200.0,
        300.0,
    )


def test_thresholds_are_passed_to_provider(validated) -> None:
    provider = FakeProvider([])
    make_service(provider).detect(validated, "img_001", conf_threshold=0.5, iou_threshold=0.3)
    make_service(provider).detect(validated, "img_001")
    assert provider.calls == [(0.5, 0.3), (0.25, 0.45)]


def test_provider_failure_is_not_swallowed(validated) -> None:
    with pytest.raises(DetectionProviderError):
        make_service(FakeProvider(fail=True)).detect(validated, "img_001")


@pytest.mark.parametrize(
    "bbox",
    [(10, 10, 10, 50), (50, 10, 20, 50), (-1, 0, 10, 10)],
)
def test_schema_rejects_invalid_bounding_boxes(bbox) -> None:
    with pytest.raises(ValidationError):
        Detection(id="det_001", label="chair", confidence=0.9, bbox=bbox)


def test_schema_rejects_boxes_outside_image() -> None:
    det = Detection(id="det_001", label="chair", confidence=0.9, bbox=(0, 0, 700, 100))
    with pytest.raises(ValidationError):
        ImageDetections(image_id="img_001", width=640, height=480, detections=[det])
