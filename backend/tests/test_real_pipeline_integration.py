"""Opt-in end-to-end check with the REAL YOLO model on a real photo with controlled edits.

    $env:RUN_YOLO_INTEGRATION = "1"; pytest tests/test_real_pipeline_integration.py

Expectations mirror what was observed on the sample photo (two cats, a remote, a couch). They
document behaviour of this specific image; they are a regression check, not an accuracy claim.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_YOLO_INTEGRATION") != "1",
    reason="set RUN_YOLO_INTEGRATION=1 to run the real YOLO model",
)

SAMPLE = Path(__file__).parent / "sample_images" / "cats_couch.jpg"


def _camera(width: int, height: int, scale=1.0, rot=0.0, tx=0.0, ty=0.0) -> np.ndarray:
    m = cv2.getRotationMatrix2D((width / 2, height / 2), rot, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return np.vstack([m, [0, 0, 1]])


def _encode(array: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", array)[1].tobytes()


def _inspect(client: TestClient, after: np.ndarray) -> dict:
    response = client.post(
        "/api/v1/inspections",
        data={"property_id": "real", "location": "bedroom"},
        files=[("files", ("after.jpg", _encode(after), "image/jpeg"))],
    )
    assert response.status_code == 201, response.text
    detail = response.json()
    assert detail["status"] == "completed", detail
    return client.get(f"/api/v1/inspections/{detail['id']}/results").json()


@pytest.fixture
def client_with_reference():
    original = cv2.imread(str(SAMPLE))
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/references",
            data={"property_id": "real", "location": "bedroom"},
            files=[("files", ("ref.jpg", _encode(original), "image/jpeg"))],
        )
        assert response.status_code == 201
        yield client, original


def _kinds(results: dict) -> list[tuple[str, str]]:
    return [(c["kind"], c["label"]) for c in results["candidates"]]


def test_camera_change_only_produces_no_candidates(client_with_reference) -> None:
    client, original = client_with_reference
    h, w = original.shape[:2]
    moved = cv2.warpPerspective(
        original, _camera(w, h, 1.1, 4, 14, -10), (w, h), borderMode=cv2.BORDER_REFLECT
    )
    results = _inspect(client, moved)

    assert results["comparison"]["coverage"] == "full"
    assert _kinds(results) == []


def test_removed_cat_is_a_single_reviewable_missing_candidate(client_with_reference) -> None:
    client, original = client_with_reference
    h, w = original.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[20:380, 330:640] = 255  # the right-hand cat
    results = _inspect(client, cv2.inpaint(original, mask, 9, cv2.INPAINT_TELEA))

    assert _kinds(results) == [("possible_missing", "cat")]
    candidate = results["candidates"][0]
    assert candidate["after_detection_id"] is None and candidate["requires_review"] is True


def test_cats_leaving_the_frame_are_not_visible_and_another_image_is_requested(
    client_with_reference,
) -> None:
    client, original = client_with_reference
    h, w = original.shape[:2]
    shifted = cv2.warpPerspective(original, _camera(w, h, tx=-330), (w, h))
    results = _inspect(client, shifted)

    assert all(kind != "possible_missing" for kind, _ in _kinds(results))
    assert ("not_visible", "cat") in _kinds(results)
    assert results["comparison"]["coverage"] == "partial"
    assert results["comparison"]["needs_additional_image"] is True
