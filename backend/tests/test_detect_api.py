from fastapi.testclient import TestClient

from app.providers.yolo_provider import RawDetection
from tests.conftest import FakeProvider, make_image_bytes


def post_image(client: TestClient, data: bytes, filename: str = "room.jpg", **params):
    return client.post(
        "/api/v1/detect",
        files={"file": (filename, data, "image/jpeg")},
        params=params,
    )


def test_detect_returns_boxes_and_dimensions(client: TestClient, fake_provider: FakeProvider) -> None:
    fake_provider.detections = [RawDetection("chair", 0.94, (120, 150, 350, 450))]
    response = post_image(client, make_image_bytes("JPEG", (1280, 720)))

    assert response.status_code == 200
    body = response.json()
    assert body["image_id"] == "img_001"
    assert (body["width"], body["height"]) == (1280, 720)
    assert body["detections"] == [
        {"id": "det_001", "label": "chair", "confidence": 0.94, "bbox": [120.0, 150.0, 350.0, 450.0]}
    ]
    assert body["model"]["name"] == "fake-yolo.pt"
    assert "not evidence that the object is absent" in body["note"]


def test_detect_with_no_detections_is_success(client: TestClient) -> None:
    response = post_image(client, make_image_bytes("PNG"), filename="x.png")
    assert response.status_code == 200
    assert response.json()["detections"] == []


def test_detect_threshold_overrides(client: TestClient, fake_provider: FakeProvider) -> None:
    response = post_image(client, make_image_bytes(), conf_threshold=0.6, iou_threshold=0.5)
    assert response.status_code == 200
    assert fake_provider.calls == [(0.6, 0.5)]
    assert response.json()["model"]["conf_threshold"] == 0.6


def test_detect_rejects_out_of_range_threshold(client: TestClient) -> None:
    assert post_image(client, make_image_bytes(), conf_threshold=1.5).status_code == 422


def test_invalid_file_type_rejected_even_with_image_extension(client: TestClient) -> None:
    response = post_image(client, b"not an image at all", filename="photo.jpg")
    assert response.status_code == 415
    assert "not a valid image" in response.json()["detail"]


def test_oversized_upload_rejected(client: TestClient) -> None:
    response = post_image(client, make_image_bytes("PNG", (800, 800), noise=True))
    assert response.status_code == 413


def test_tiny_image_rejected(client: TestClient) -> None:
    response = post_image(client, make_image_bytes("PNG", (100, 100)))
    assert response.status_code == 422


def test_missing_file_field(client: TestClient) -> None:
    assert client.post("/api/v1/detect").status_code == 422


def test_detector_failure_returns_503_not_empty_result(
    client: TestClient, fake_provider: FakeProvider
) -> None:
    fake_provider.fail = True
    response = post_image(client, make_image_bytes())
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_supported_classes_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/detect/classes")
    assert response.status_code == 200
    body = response.json()
    assert body["classes"] == ["chair", "couch", "tv"]
    assert "unsupported" in body["note"]
