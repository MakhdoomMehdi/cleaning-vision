from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core import config
from app.db.models import ReferenceImage
from app.db.session import get_session_factory, reset_db_caches
from app.main import create_app
from tests.conftest import make_image_bytes


def upload(client: TestClient, files, property_id: str = "prop-1", location: str = "garage"):
    return client.post(
        "/api/v1/references",
        data={"property_id": property_id, "location": location},
        files=[("files", (name, data, mime)) for name, data, mime in files],
    )


def jpeg(name: str = "a.jpg", size: tuple[int, int] = (640, 480)):
    return (name, make_image_bytes("JPEG", size), "image/jpeg")


def stored_files(upload_dir: Path) -> list[Path]:
    return [p for p in upload_dir.rglob("*") if p.is_file()]


def test_locations_endpoint(client: TestClient) -> None:
    body = client.get("/api/v1/locations").json()
    values = [item["value"] for item in body["items"]]
    assert values == ["garage", "bedroom", "kitchen", "bathroom", "ceiling", "room_corner"]
    assert {"value": "room_corner", "label": "Room Corner"} in body["items"]


def test_upload_valid_image_and_retrieve_bytes(client: TestClient, isolated_storage: Path) -> None:
    data = make_image_bytes("JPEG", (800, 600))
    response = upload(client, [("kitchen.jpg", data, "image/jpeg")])

    assert response.status_code == 201
    body = response.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["property_id"] == "prop-1"
    assert item["location"] == "garage"
    assert (item["width"], item["height"]) == (800, 600)
    assert item["mime_type"] == "image/jpeg"
    assert item["size_bytes"] == len(data)
    assert item["original_filename"] == "kitchen.jpg"
    assert item["image_url"] == f"/api/v1/references/{item['id']}/image"

    image = client.get(item["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.content == data  # original bytes preserved

    # Stored under a generated name, never the user's filename.
    (path,) = stored_files(isolated_storage)
    assert path.name != "kitchen.jpg"
    assert path.parent.name == "references"


def test_hostile_filename_is_sanitised_and_never_used_as_a_path(
    client: TestClient, isolated_storage: Path
) -> None:
    response = upload(client, [("../../evil.jpg", make_image_bytes(), "image/jpeg")])
    assert response.status_code == 201
    assert response.json()["items"][0]["original_filename"] == "evil.jpg"
    (path,) = stored_files(isolated_storage)
    assert path.is_relative_to(isolated_storage)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../evil.jpg", "evil.jpg"),
        ("C:\\Users\\x\\room.png", "room.png"),
        ("evil\x00\x1f.jpg", "evil.jpg"),
        ("", "image"),
        (None, "image"),
        ("a" * 400 + ".jpg", ("a" * 400 + ".jpg")[:255]),
    ],
)
def test_display_filename_sanitiser(raw: str | None, expected: str) -> None:
    from app.services.reference_service import display_filename

    assert display_filename(raw) == expected


def test_multiple_references_per_location_in_one_request(client: TestClient) -> None:
    response = upload(
        client,
        [
            jpeg("a.jpg"),
            ("b.png", make_image_bytes("PNG"), "image/png"),
            ("c.webp", make_image_bytes("WEBP"), "image/webp"),
        ],
    )
    assert response.status_code == 201
    items = response.json()["items"]
    assert [i["mime_type"] for i in items] == ["image/jpeg", "image/png", "image/webp"]
    assert len({i["id"] for i in items}) == 3

    listed = client.get("/api/v1/references", params={"property_id": "prop-1", "location": "garage"})
    assert listed.json()["total"] == 3


def test_list_filters_by_property_and_location(client: TestClient) -> None:
    upload(client, [jpeg()], "prop-1", "garage")
    upload(client, [jpeg()], "prop-1", "kitchen")
    upload(client, [jpeg()], "prop-2", "garage")

    def count(**params: str) -> int:
        return client.get("/api/v1/references", params=params).json()["total"]

    assert count() == 3
    assert count(property_id="prop-1") == 2
    assert count(location="garage") == 2
    assert count(property_id="prop-1", location="garage") == 1
    assert count(property_id="nope") == 0


def test_references_survive_restart(client: TestClient) -> None:
    reference_id = upload(client, [jpeg()]).json()["items"][0]["id"]

    # Simulate a process restart: drop cached engine/sessions and build a fresh app.
    reset_db_caches()
    with TestClient(create_app()) as fresh:
        items = fresh.get("/api/v1/references").json()["items"]
        assert [i["id"] for i in items] == [reference_id]
        assert fresh.get(f"/api/v1/references/{reference_id}/image").status_code == 200


def test_database_row_has_uuid_utc_timestamp_and_checksum(client: TestClient) -> None:
    upload(client, [jpeg()])

    with get_session_factory()() as session:
        row = session.scalars(select(ReferenceImage)).one()
        assert len(row.id) == 36 and row.id.count("-") == 4
        assert row.created_at.tzinfo is not None
        assert len(row.sha256) == 64
        assert row.storage_key.startswith("references/")


def test_invalid_file_type_rejected(client: TestClient, isolated_storage: Path) -> None:
    response = upload(client, [("notes.jpg", b"definitely not an image", "image/jpeg")])
    assert response.status_code == 415
    assert "notes.jpg" in response.json()["detail"]
    assert stored_files(isolated_storage) == []


def test_oversized_file_rejected(client: TestClient, isolated_storage: Path) -> None:
    big = make_image_bytes("PNG", (800, 800), noise=True)  # larger than the 1 MB test limit
    response = upload(client, [("big.png", big, "image/png")])
    assert response.status_code == 413
    assert stored_files(isolated_storage) == []


def test_undersized_image_rejected(client: TestClient) -> None:
    assert upload(client, [jpeg("tiny.jpg", (100, 100))]).status_code == 422


def test_batch_is_atomic_one_bad_file_stores_nothing(
    client: TestClient, isolated_storage: Path
) -> None:
    response = upload(client, [jpeg("good.jpg"), ("bad.jpg", b"garbage", "image/jpeg")])
    assert response.status_code == 415
    assert stored_files(isolated_storage) == []
    assert client.get("/api/v1/references").json()["total"] == 0


def test_database_failure_removes_files_already_written(
    client: TestClient, isolated_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.orm import Session

    def boom(self: Session) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(Session, "commit", boom)
    with TestClient(create_app(), raise_server_exceptions=False) as failing:
        response = upload(failing, [jpeg("a.jpg"), jpeg("b.jpg")])
    assert response.status_code == 500
    assert stored_files(isolated_storage) == []  # no orphaned files


def test_too_many_files_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_IMAGES_PER_REQUEST", "2")
    config.get_settings.cache_clear()
    response = upload(client, [jpeg("1.jpg"), jpeg("2.jpg"), jpeg("3.jpg")])
    assert response.status_code == 422
    assert "At most 2" in response.json()["detail"]


@pytest.mark.parametrize("location", ["attic", "", "GARAGE"])
def test_invalid_location_rejected(client: TestClient, location: str) -> None:
    assert upload(client, [jpeg()], location=location).status_code == 422


@pytest.mark.parametrize("property_id", ["", " ", "../etc", "a b", "-lead", "x" * 65, "a/b"])
def test_invalid_property_id_rejected(client: TestClient, property_id: str) -> None:
    assert upload(client, [jpeg()], property_id=property_id).status_code == 422


def test_missing_files_field_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/references", data={"property_id": "p", "location": "garage"})
    assert response.status_code == 422


def test_unknown_reference_image_404(client: TestClient) -> None:
    assert client.get("/api/v1/references/00000000-0000-4000-8000-000000000000/image").status_code == 404


def test_non_uuid_reference_id_rejected(client: TestClient) -> None:
    assert client.get("/api/v1/references/not-a-uuid/image").status_code == 422
    assert client.get("/api/v1/references/..%2F..%2Fetc%2Fpasswd/image").status_code in (404, 422)


def test_missing_file_on_disk_returns_404_not_500(
    client: TestClient, isolated_storage: Path
) -> None:
    item = upload(client, [jpeg()]).json()["items"][0]
    for path in stored_files(isolated_storage):
        path.unlink()
    assert client.get(item["image_url"]).status_code == 404


def test_list_rejects_invalid_filters(client: TestClient) -> None:
    assert client.get("/api/v1/references", params={"location": "attic"}).status_code == 422
    assert client.get("/api/v1/references", params={"property_id": "../x"}).status_code == 422
