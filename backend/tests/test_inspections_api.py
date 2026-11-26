import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import func, select

from app.api.deps import get_image_service, get_vision_provider
from app.core import config
from app.db.models import Inspection, InspectionImage
from app.db.session import get_session_factory, reset_db_caches
from app.main import create_app
from app.providers.openai_provider import ReviewImage, VisionProviderError
from app.schemas.review import ChangeMark, GuestIssue, ReviewMode, RoomReview
from app.services.image_service import ImageService
from tests.conftest import make_image_bytes


class FakeVision:
    model_name = "fake-vision"

    def __init__(self) -> None:
        self.fail = False
        self.calls = 0
        self.ready_calls = 0
        self.ready_report = RoomReview(
            readiness=True,
            guest_ready=False,
            overall_score=35,
            summary="Bed is not made and the sheet is stained.",
            issues=[
                GuestIssue(
                    category="bedding",
                    severity="critical",
                    photo_number=1,
                    comment="Bedsheet is wrinkled and untucked; remake the bed.",
                    x1=0.12,
                    y1=0.4,
                    x2=0.78,
                    y2=0.82,
                )
            ],
        )
        self.report = RoomReview(
            mode=ReviewMode.BOXES,
            matched_photo=2,
            summary="One scratch and one missing chair.",
            views_compared="The bed wall.",
            marks=[
                ChangeMark(
                    kind="damage",
                    description="scratch on the door",
                    side="after",
                    photo_number=1,
                    x1=0.1,
                    y1=0.2,
                    x2=0.4,
                    y2=0.55,
                ),
                ChangeMark(
                    kind="missing",
                    description="chair by the window",
                    side="before",
                    photo_number=1,
                    x1=0.5,
                    y1=0.1,
                    x2=0.8,
                    y2=0.6,
                ),
            ],
        )

    def review(self, before: list[ReviewImage], after: list[ReviewImage]) -> RoomReview:
        self.calls += 1
        assert before and after
        if self.fail:
            raise VisionProviderError("secret internal detail")
        return self.report

    def check_ready(self, photos: list[ReviewImage]) -> RoomReview:
        self.ready_calls += 1
        assert photos
        if self.fail:
            raise VisionProviderError("secret internal detail")
        return self.ready_report


def jpeg_bytes(size: tuple[int, int] = (640, 480)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (140, 170, 190)).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


@pytest.fixture
def vision() -> FakeVision:
    return FakeVision()


@pytest.fixture
def api(vision: FakeVision) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_vision_provider] = lambda: vision
    app.dependency_overrides[get_image_service] = lambda: ImageService(2 * 1024 * 1024, 320, 4000)
    with TestClient(app) as client:
        yield client


def upload_reference(api: TestClient, property_id: str = "p1", location: str = "garage") -> None:
    response = api.post(
        "/api/v1/references",
        data={"property_id": property_id, "location": location},
        files=[("files", ("ref.jpg", jpeg_bytes(), "image/jpeg"))],
    )
    assert response.status_code == 201, response.text


def start_inspection(
    api: TestClient,
    images: list[bytes],
    property_id: str = "p1",
    location: str = "garage",
):
    return api.post(
        "/api/v1/inspections",
        data={"property_id": property_id, "location": location},
        files=[("files", (f"after{i}.jpg", data, "image/jpeg")) for i, data in enumerate(images)],
    )


def stored(upload_dir: Path, category: str) -> list[Path]:
    folder = upload_dir / category
    return [p for p in folder.rglob("*") if p.is_file()] if folder.exists() else []


def count(model: type) -> int:
    with get_session_factory()() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def test_review_returns_damage_missing_and_new(api: TestClient, vision: FakeVision) -> None:
    upload_reference(api)
    upload_reference(api)
    after = jpeg_bytes()
    response = start_inspection(api, [after])
    assert response.status_code == 201, response.text
    detail = response.json()

    assert detail["status"] == "completed"
    assert detail["failed_stage"] is None and detail["error_message"] is None
    assert detail["reference_image_count"] == 2 and detail["inspection_image_count"] == 1
    assert detail["change_count"] == 2
    assert detail["decision"] == "potential_issues"
    assert detail["vision_model"] == "fake-vision"
    assert [i["image_id"] for i in detail["images"]] == ["ref_1", "ref_2", "new_1"]
    assert vision.calls == 1

    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["matched_image_id"] == "ref_2"
    assert [image["image_id"] for image in results["images"]] == ["ref_2", "new_1"]
    assert results["damage"] == ["scratch on the door"]
    assert results["missing"] == ["chair by the window"]
    assert results["new"] == []
    assert {box["kind"] for box in results["boxes"]} == {"damage", "missing"}
    assert {box["image_id"] for box in results["boxes"]} == {"new_1"}

    image = api.get(f"/api/v1/inspections/{detail['id']}/images/new_1")
    assert image.status_code == 200 and image.content == after
    assert api.get(f"/api/v1/inspections/{detail['id']}/images/ref_1").status_code == 200


def test_moved_item_is_one_box_and_a_pin(api: TestClient, vision: FakeVision) -> None:
    vision.report = RoomReview(
        mode=ReviewMode.BOXES,
        matched_photo=1,
        marks=[
            ChangeMark(
                kind="moved",
                description="lamp moved onto the dresser",
                side="after",
                photo_number=1,
                x1=0.6,
                y1=0.2,
                x2=0.75,
                y2=0.45,
                origin_x=0.2,
                origin_y=0.4,
            )
        ],
    )
    upload_reference(api)
    detail = start_inspection(api, [jpeg_bytes()]).json()
    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert len(results["boxes"]) == 1
    box = results["boxes"][0]
    assert box["kind"] == "moved"
    assert box["image_id"] == "new_1"
    assert box["origin_x"] == 0.2 and box["origin_y"] == 0.4
    assert results["new"] == []


def test_inconclusive_when_the_model_cannot_match_a_view(api: TestClient, vision: FakeVision) -> None:
    vision.report = RoomReview(
        could_compare=False,
        summary="The after photo shows a corner that was not in the before photos.",
        damage=["should be dropped"],
    )
    upload_reference(api)
    detail = start_inspection(api, [jpeg_bytes()]).json()
    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["decision"] == "inconclusive"
    assert results["damage"] == [] and results["missing"] == [] and results["new"] == []
    assert "corner" in results["summary"]


def test_results_survive_restart(api: TestClient) -> None:
    upload_reference(api)
    detail = start_inspection(api, [jpeg_bytes()]).json()
    before = api.get(f"/api/v1/inspections/{detail['id']}/results").json()

    reset_db_caches()
    with TestClient(create_app()) as fresh:
        after = fresh.get(f"/api/v1/inspections/{detail['id']}/results").json()
        assert fresh.get(f"/api/v1/inspections/{detail['id']}").json()["status"] == "completed"
    assert after == before


def test_database_rows_link_reference_files(api: TestClient) -> None:
    upload_reference(api)
    detail = start_inspection(api, [jpeg_bytes()]).json()
    with get_session_factory()() as session:
        inspection = session.get_one(Inspection, detail["id"])
        assert len(inspection.id) == 36 and inspection.created_at.tzinfo is not None
        assert inspection.comparison["missing"] == ["chair by the window"]
        images = session.scalars(select(InspectionImage)).all()
        assert sorted(i.image_ref for i in images) == ["new_1", "ref_1"]
        ref = next(i for i in images if i.role == "reference")
        assert ref.storage_key is None and ref.reference_image_id


def test_inspection_without_references_checks_guest_readiness(
    api: TestClient, vision: FakeVision, isolated_storage: Path
) -> None:
    response = start_inspection(api, [jpeg_bytes()])
    assert response.status_code == 201
    detail = response.json()
    assert detail["status"] == "completed"
    assert detail["reference_image_count"] == 0 and detail["inspection_image_count"] == 1
    assert detail["decision"] == "potential_issues"
    assert vision.calls == 0 and vision.ready_calls == 1
    assert stored(isolated_storage, "inspections")

    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["readiness"] is True and results["guest_ready"] is False
    assert results["overall_score"] == 35
    assert results["summary"] == "Bed is not made and the sheet is stained."
    assert [image["image_id"] for image in results["images"]] == ["new_1"]
    assert results["boxes"] == [
        {
            "image_id": "new_1",
            "kind": "issue",
            "description": "Critical: Bedsheet is wrinkled and untucked; remake the bed.",
            "x1": 0.12,
            "y1": 0.4,
            "x2": 0.78,
            "y2": 0.82,
            "origin_x": None,
            "origin_y": None,
        }
    ]


def test_references_of_another_location_are_not_used(api: TestClient, vision: FakeVision) -> None:
    upload_reference(api, location="kitchen")
    response = start_inspection(api, [jpeg_bytes()], location="garage")
    assert response.status_code == 201
    assert response.json()["reference_image_count"] == 0
    assert vision.calls == 0 and vision.ready_calls == 1


def test_invalid_image_rejected_atomically(
    api: TestClient, vision: FakeVision, isolated_storage: Path
) -> None:
    upload_reference(api)
    response = start_inspection(api, [jpeg_bytes(), b"not an image"])
    assert response.status_code == 415
    assert "after1.jpg" in response.json()["detail"]
    assert count(Inspection) == 0
    assert stored(isolated_storage, "inspections") == []
    assert vision.calls == 0


def test_oversized_and_too_many_and_bad_fields(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    upload_reference(api)
    big = make_image_bytes("PNG", (1600, 1600), noise=True)
    assert start_inspection(api, [big]).status_code == 413

    monkeypatch.setenv("MAX_IMAGES_PER_REQUEST", "1")
    config.get_settings.cache_clear()
    small = jpeg_bytes()
    assert start_inspection(api, [small, small]).status_code == 422
    config.get_settings.cache_clear()

    assert start_inspection(api, [small], location="attic").status_code == 422
    assert start_inspection(api, [small], property_id="../x").status_code == 422
    assert api.post("/api/v1/inspections", data={"property_id": "p1", "location": "garage"}).status_code == 422


def test_too_many_reference_images_is_rejected_not_silently_truncated(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_reference(api)
    upload_reference(api)
    monkeypatch.setenv("MAX_REFERENCE_IMAGES_PER_INSPECTION", "1")
    config.get_settings.cache_clear()
    response = start_inspection(api, [jpeg_bytes()])
    assert response.status_code == 422
    assert "at most 1" in response.json()["detail"]
    assert count(Inspection) == 0


def test_storage_failure_while_saving_inputs_leaves_no_orphans(
    api: TestClient, isolated_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_reference(api)

    from sqlalchemy.orm import Session

    def boom(self: Session) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(Session, "commit", boom)
    with TestClient(create_app(), raise_server_exceptions=False) as failing:
        response = start_inspection(failing, [jpeg_bytes()])
    assert response.status_code == 500
    assert stored(isolated_storage, "inspections") == []


def test_vision_failure_is_recorded_without_leaking_internals(api: TestClient, vision: FakeVision) -> None:
    upload_reference(api)
    vision.fail = True
    detail = start_inspection(api, [jpeg_bytes()]).json()
    assert detail["status"] == "failed"
    assert detail["failed_stage"] == "review"
    assert "could not be compared" in detail["error_message"].lower()
    assert "secret internal detail" not in detail["error_message"]

    results = api.get(f"/api/v1/inspections/{detail['id']}/results").json()
    assert results["damage"] == [] and results["missing"] == [] and results["new"] == []
    assert api.get("/api/v1/inspections").json()["items"][0]["status"] == "failed"


def test_missing_api_key_fails_the_review(vision: FakeVision) -> None:
    app = create_app()
    app.dependency_overrides[get_image_service] = lambda: ImageService(2 * 1024 * 1024, 320, 4000)
    with TestClient(app) as client:
        upload_reference(client)
        detail = start_inspection(client, [jpeg_bytes()]).json()
    assert detail["status"] == "failed" and detail["failed_stage"] == "review"
    assert "OPENAI_API_KEY" in detail["error_message"]
    assert vision.calls == 0


def test_inspection_is_never_left_processing(api: TestClient) -> None:
    upload_reference(api)
    start_inspection(api, [jpeg_bytes()])
    statuses = {i["status"] for i in api.get("/api/v1/inspections").json()["items"]}
    assert statuses <= {"completed", "failed"}


def test_list_filters_ordering_and_pagination(api: TestClient) -> None:
    upload_reference(api, "p1", "garage")
    upload_reference(api, "p2", "garage")
    photo = jpeg_bytes()
    ids = [start_inspection(api, [photo], property_id=prop).json()["id"] for prop in ("p1", "p2", "p1")]

    everything = api.get("/api/v1/inspections").json()
    assert everything["total"] == 3
    assert [i["id"] for i in everything["items"]] == list(reversed(ids))
    assert api.get("/api/v1/inspections", params={"property_id": "p1"}).json()["total"] == 2
    assert api.get("/api/v1/inspections", params={"location": "kitchen"}).json()["total"] == 0
    page = api.get("/api/v1/inspections", params={"limit": 1, "offset": 1}).json()
    assert page["total"] == 3 and [i["id"] for i in page["items"]] == [ids[1]]
    assert everything["items"][0]["change_count"] == 2
    assert api.get("/api/v1/inspections", params={"limit": 0}).status_code == 422


def test_unknown_and_malformed_ids(api: TestClient) -> None:
    missing = "00000000-0000-4000-8000-000000000000"
    assert api.get(f"/api/v1/inspections/{missing}").status_code == 404
    assert api.get(f"/api/v1/inspections/{missing}/results").status_code == 404
    assert api.get(f"/api/v1/inspections/{missing}/images/new_1").status_code == 404
    assert api.get("/api/v1/inspections/not-a-uuid").status_code == 422


def test_image_ref_is_validated_and_unknown_refs_404(api: TestClient) -> None:
    upload_reference(api)
    detail = start_inspection(api, [jpeg_bytes()]).json()
    base = f"/api/v1/inspections/{detail['id']}/images"
    assert api.get(f"{base}/new_9").status_code == 404
    assert api.get(f"{base}/..%2F..%2Fetc").status_code in (404, 422)
    assert api.get(f"{base}/evil_1").status_code == 422


def test_files_deleted_from_disk_give_404_not_500(api: TestClient, isolated_storage: Path) -> None:
    upload_reference(api)
    detail = start_inspection(api, [jpeg_bytes()]).json()
    for path in stored(isolated_storage, "inspections"):
        path.unlink()
    assert api.get(f"/api/v1/inspections/{detail['id']}/images/new_1").status_code == 404
