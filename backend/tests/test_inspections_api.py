import io
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import func, select

from app.api.deps import get_detection_service, get_image_service
from app.core import config
from app.db.models import DetectionRecord, Inspection, InspectionImage
from app.db.session import get_session_factory, reset_db_caches
from app.main import create_app
from app.providers.yolo_provider import DetectionProviderError, RawDetection
from app.services.detection_service import DetectionService
from app.services.image_service import ImageService
from app.services.matching_service import ComparisonImage, MatchingService
from tests.conftest import make_image_bytes
from tests.synthetic import Scenario, WorldObject, camera

CHAIR = WorldObject("chair", seed=1, xy=(150, 200))
TV = WorldObject("tv", seed=2, xy=(600, 380))


class ScriptedProvider:
    """Returns one pre-set detection list per call, in call order (references first)."""

    model_name = "fake-yolo.pt"
    device = "cpu"
    class_names = ["chair", "tv"]

    def __init__(self) -> None:
        self.script: list[list[RawDetection]] = []
        self.fail = False
        self.calls = 0

    def detect(self, image, conf_threshold, iou_threshold):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.fail:
            raise DetectionProviderError("YOLO inference failed.")
        return self.script.pop(0) if self.script else []


def jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def raw(ci: ComparisonImage) -> list[RawDetection]:
    return [RawDetection(d.label, d.confidence, d.bbox) for d in ci.detections.detections]


@pytest.fixture
def provider() -> ScriptedProvider:
    return ScriptedProvider()


@pytest.fixture
def api(provider: ScriptedProvider) -> Iterator[TestClient]:
    app = create_app()
    detector = DetectionService(provider, 0.25, 0.45)  # type: ignore[arg-type]
    app.dependency_overrides[get_detection_service] = lambda: detector
    app.dependency_overrides[get_image_service] = lambda: ImageService(2 * 1024 * 1024, 320, 4000)
    with TestClient(app) as client:
        yield client


def upload_reference(api: TestClient, image: Image.Image, property_id: str = "p1", location: str = "garage"):
    return api.post(
        "/api/v1/references",
        data={"property_id": property_id, "location": location},
        files=[("files", ("ref.jpg", jpeg_bytes(image), "image/jpeg"))],
    )


def start_inspection(
    api: TestClient, images: list[bytes], property_id: str = "p1", location: str = "garage"
):
    return api.post(
        "/api/v1/inspections",
        data={"property_id": property_id, "location": location},
        files=[("files", (f"after{i}.jpg", data, "image/jpeg")) for i, data in enumerate(images)],
    )


def run_scenario(
    api: TestClient, provider: ScriptedProvider, scenario: Scenario
) -> tuple[dict, bytes, bytes]:
    ref, new = scenario.build()
    ref_bytes, new_bytes = jpeg_bytes(ref.image), jpeg_bytes(new.image)
    assert upload_reference(api, ref.image).status_code == 201
    provider.script = [raw(ref), raw(new)]
    response = start_inspection(api, [new_bytes])
    assert response.status_code == 201, response.text
    return response.json(), ref_bytes, new_bytes


def stored(upload_dir: Path, category: str) -> list[Path]:
    folder = upload_dir / category
    return [p for p in folder.rglob("*") if p.is_file()] if folder.exists() else []


def count(model: type) -> int:
    with get_session_factory()() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


# --------------------------------------------------------------------------- happy paths


def test_missing_object_flows_through_to_a_reviewable_candidate(
    api: TestClient, provider: ScriptedProvider
) -> None:
    detail, _, new_bytes = run_scenario(
        api, provider, Scenario(ref_objects=[CHAIR, TV], new_objects=[TV])
    )

    assert detail["status"] == "completed"
    assert detail["failed_stage"] is None and detail["error_message"] is None
    assert detail["property_id"] == "p1" and detail["location"] == "garage"
    assert detail["reference_image_count"] == 1 and detail["inspection_image_count"] == 1
    assert detail["models"]["detector"]["name"] == "fake-yolo.pt"
    assert detail["models"]["matching_version"] == "matching-v1"
    assert detail["completed_at"] is not None
    assert [i["image_id"] for i in detail["images"]] == ["ref_1", "new_1"]

    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["status"] == "completed"
    by_image = {i["image_id"]: i for i in results["images"]}
    assert [d["id"] for d in by_image["ref_1"]["detections"]] == ["ref_1_det_001", "ref_1_det_002"]
    assert [d["id"] for d in by_image["new_1"]["detections"]] == ["new_1_det_001"]
    assert by_image["ref_1"]["width"] == 960

    assert results["comparison"]["valid"] is True
    assert results["comparison"]["coverage"] in ("full", "partial")
    missing = [c for c in results["candidates"] if c["kind"] == "possible_missing"]
    assert [c["label"] for c in missing] == ["chair"]
    assert missing[0]["before_detection_id"] == "ref_1_det_001"
    assert missing[0]["after_detection_id"] is None
    assert missing[0]["requires_review"] is True
    assert "decision" not in results  # decision comes from the validation stage (later phase)

    # Uploaded inspection image is served back unchanged.
    image = api.get(by_image["new_1"]["image_url"])
    assert image.status_code == 200 and image.content == new_bytes
    assert api.get(by_image["ref_1"]["image_url"]).status_code == 200


def test_camera_angle_change_alone_creates_no_missing_candidate_end_to_end(
    api: TestClient, provider: ScriptedProvider
) -> None:
    scenario = Scenario(
        ref_objects=[CHAIR, TV],
        new_cam=camera(scale=1.1, rotation_deg=5, tx=22, ty=-14, perspective=(3e-5, -2e-5)),
    )
    detail, _, _ = run_scenario(api, provider, scenario)

    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["status"] == "completed"
    assert all(c["kind"] != "possible_missing" for c in results["candidates"])
    assert len(results["matches"]) == 2


def test_object_leaving_the_frame_is_not_visible_end_to_end(
    api: TestClient, provider: ScriptedProvider
) -> None:
    detail, _, _ = run_scenario(
        api, provider, Scenario(ref_objects=[CHAIR, TV], new_cam=camera(tx=430))
    )
    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    kinds = {c["label"]: c["kind"] for c in results["candidates"]}
    assert kinds == {"tv": "not_visible"}


def test_unrelated_images_report_insufficient_coverage_and_request_another_image(
    api: TestClient, provider: ScriptedProvider
) -> None:
    from tests.synthetic import textured_world

    scenario = Scenario(
        ref_objects=[CHAIR], new_objects=[], new_background=textured_world(seed=91)
    )
    detail, _, _ = run_scenario(api, provider, scenario)

    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["status"] == "completed"
    assert results["comparison"]["coverage"] == "insufficient"
    assert results["comparison"]["valid"] is False
    assert results["comparison"]["needs_additional_image"] is True
    assert results["comparison"]["recommendation"]
    assert {c["kind"] for c in results["candidates"]} == {"inconclusive"}


def test_results_survive_restart(api: TestClient, provider: ScriptedProvider) -> None:
    detail, _, _ = run_scenario(api, provider, Scenario(ref_objects=[CHAIR, TV], new_objects=[TV]))
    before = api.get(f"/api/v1/inspections/{detail['id']}/results").json()

    reset_db_caches()
    with TestClient(create_app()) as fresh:
        after = fresh.get(f"/api/v1/inspections/{detail['id']}/results").json()
        assert fresh.get(f"/api/v1/inspections/{detail['id']}").json()["status"] == "completed"
    assert after == before


def test_database_rows_for_inspection_images_and_detections(
    api: TestClient, provider: ScriptedProvider
) -> None:
    detail, _, _ = run_scenario(api, provider, Scenario(ref_objects=[CHAIR, TV], new_objects=[TV]))

    with get_session_factory()() as session:
        inspection = session.get_one(Inspection, detail["id"])
        assert len(inspection.id) == 36
        assert inspection.yolo_model == "fake-yolo.pt" and inspection.created_at.tzinfo is not None
        images = session.scalars(select(InspectionImage)).all()
        assert sorted(i.image_ref for i in images) == ["new_1", "ref_1"]
        ref = next(i for i in images if i.role == "reference")
        assert ref.storage_key is None and ref.reference_image_id  # reuses the reference file
        assert count(DetectionRecord) == 3


# --------------------------------------------------------------------------- input rejection


def test_inspection_without_references_is_rejected_and_stores_nothing(
    api: TestClient, isolated_storage: Path
) -> None:
    response = start_inspection(api, [make_image_bytes("JPEG", (640, 480))])
    assert response.status_code == 409
    assert "reference" in response.json()["detail"].lower()
    assert count(Inspection) == 0 and stored(isolated_storage, "inspections") == []


def test_references_of_another_location_are_not_used(api: TestClient, provider: ScriptedProvider) -> None:
    ref, _ = Scenario(ref_objects=[CHAIR]).build()
    upload_reference(api, ref.image, location="kitchen")
    response = start_inspection(api, [make_image_bytes()], location="garage")
    assert response.status_code == 409


def test_invalid_image_rejected_atomically(
    api: TestClient, provider: ScriptedProvider, isolated_storage: Path
) -> None:
    ref, _ = Scenario(ref_objects=[CHAIR]).build()
    upload_reference(api, ref.image)
    response = start_inspection(api, [make_image_bytes(), b"not an image"])

    assert response.status_code == 415
    assert "after1.jpg" in response.json()["detail"]
    assert count(Inspection) == 0
    assert stored(isolated_storage, "inspections") == []
    assert provider.calls == 0  # no model work for rejected input


def test_oversized_and_too_many_and_bad_fields(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    ref, _ = Scenario(ref_objects=[CHAIR]).build()
    upload_reference(api, ref.image)

    big = make_image_bytes("PNG", (1600, 1600), noise=True)  # > 2 MB test limit
    assert start_inspection(api, [big]).status_code == 413

    monkeypatch.setenv("MAX_IMAGES_PER_REQUEST", "1")
    config.get_settings.cache_clear()
    small = make_image_bytes()
    assert start_inspection(api, [small, small]).status_code == 422
    config.get_settings.cache_clear()

    assert start_inspection(api, [small], location="attic").status_code == 422
    assert start_inspection(api, [small], property_id="../x").status_code == 422
    assert api.post("/api/v1/inspections", data={"property_id": "p1", "location": "garage"}).status_code == 422


def test_too_many_reference_images_is_rejected_not_silently_truncated(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref, _ = Scenario(ref_objects=[CHAIR]).build()
    upload_reference(api, ref.image)
    upload_reference(api, ref.image)
    monkeypatch.setenv("MAX_REFERENCE_IMAGES_PER_INSPECTION", "1")
    config.get_settings.cache_clear()

    response = start_inspection(api, [make_image_bytes()])
    assert response.status_code == 422
    assert "at most 1" in response.json()["detail"]
    assert count(Inspection) == 0


def test_storage_failure_while_saving_inputs_leaves_no_orphans(
    api: TestClient, isolated_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref, _ = Scenario(ref_objects=[CHAIR]).build()
    upload_reference(api, ref.image)

    from sqlalchemy.orm import Session

    def boom(self: Session) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(Session, "commit", boom)
    with TestClient(create_app(), raise_server_exceptions=False) as failing:
        response = start_inspection(failing, [make_image_bytes(), make_image_bytes()])
    assert response.status_code == 500
    assert stored(isolated_storage, "inspections") == []


# --------------------------------------------------------------------------- failed stages


def test_detector_failure_is_recorded_not_swallowed(
    api: TestClient, provider: ScriptedProvider
) -> None:
    ref, new = Scenario(ref_objects=[CHAIR]).build()
    upload_reference(api, ref.image)
    provider.fail = True

    response = start_inspection(api, [jpeg_bytes(new.image)])
    assert response.status_code == 201
    detail = response.json()
    assert detail["status"] == "failed"
    assert detail["failed_stage"] == "detect_objects"
    assert "detection failed" in detail["error_message"].lower()
    assert "YOLO inference failed" not in detail["error_message"]  # no internals leaked

    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["status"] == "failed"
    assert results["comparison"] is None and results["candidates"] == []
    assert api.get("/api/v1/inspections").json()["items"][0]["status"] == "failed"


def test_comparison_failure_keeps_detections_and_names_the_stage(
    api: TestClient, provider: ScriptedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(self: MatchingService, *args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(MatchingService, "compare", explode)
    ref, new = Scenario(ref_objects=[CHAIR]).build()
    upload_reference(api, ref.image)
    provider.script = [raw(ref), raw(new)]

    detail = start_inspection(api, [jpeg_bytes(new.image)]).json()
    assert detail["status"] == "failed" and detail["failed_stage"] == "compare_images"
    assert "secret internal detail" not in detail["error_message"]

    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["images"][0]["detections"]  # stage-3 output was preserved
    assert results["candidates"] == []


def test_inspection_is_never_left_processing_or_queued(
    api: TestClient, provider: ScriptedProvider
) -> None:
    run_scenario(api, provider, Scenario(ref_objects=[CHAIR, TV], new_objects=[TV]))
    statuses = {i["status"] for i in api.get("/api/v1/inspections").json()["items"]}
    assert statuses <= {"completed", "failed"}


# --------------------------------------------------------------------------- listing / lookup


def test_list_filters_ordering_and_pagination(api: TestClient, provider: ScriptedProvider) -> None:
    ref, new = Scenario(ref_objects=[CHAIR]).build()
    upload_reference(api, ref.image, "p1", "garage")
    upload_reference(api, ref.image, "p2", "garage")
    new_bytes = jpeg_bytes(new.image)
    ids = []
    for prop in ("p1", "p2", "p1"):
        provider.script = [raw(ref), raw(new)]
        ids.append(start_inspection(api, [new_bytes], property_id=prop).json()["id"])

    everything = api.get("/api/v1/inspections").json()
    assert everything["total"] == 3
    assert [i["id"] for i in everything["items"]] == list(reversed(ids))  # newest first
    assert api.get("/api/v1/inspections", params={"property_id": "p1"}).json()["total"] == 2
    assert api.get("/api/v1/inspections", params={"location": "kitchen"}).json()["total"] == 0
    page = api.get("/api/v1/inspections", params={"limit": 1, "offset": 1}).json()
    assert page["total"] == 3 and [i["id"] for i in page["items"]] == [ids[1]]
    first = everything["items"][0]
    assert first["coverage"] in ("full", "partial") and first["candidate_count"] is not None
    assert api.get("/api/v1/inspections", params={"limit": 0}).status_code == 422


def test_unknown_and_malformed_ids(api: TestClient) -> None:
    missing = "00000000-0000-4000-8000-000000000000"
    assert api.get(f"/api/v1/inspections/{missing}").status_code == 404
    assert api.get(f"/api/v1/inspections/{missing}/results").status_code == 404
    assert api.get(f"/api/v1/inspections/{missing}/images/new_1").status_code == 404
    assert api.get("/api/v1/inspections/not-a-uuid").status_code == 422


def test_image_ref_is_validated_and_unknown_refs_404(
    api: TestClient, provider: ScriptedProvider
) -> None:
    detail, _, _ = run_scenario(api, provider, Scenario(ref_objects=[CHAIR]))
    base = f"/api/v1/inspections/{detail['id']}/images"
    assert api.get(f"{base}/new_9").status_code == 404
    assert api.get(f"{base}/..%2F..%2Fetc").status_code in (404, 422)
    assert api.get(f"{base}/evil_1").status_code == 422


def test_files_deleted_from_disk_give_404_not_500(
    api: TestClient, provider: ScriptedProvider, isolated_storage: Path
) -> None:
    detail, _, _ = run_scenario(api, provider, Scenario(ref_objects=[CHAIR]))
    for path in stored(isolated_storage, "inspections"):
        path.unlink()
    assert api.get(f"/api/v1/inspections/{detail['id']}/images/new_1").status_code == 404
