from pathlib import Path

import pytest

from app.storage.base import StorageError
from app.storage.local_storage import LocalImageStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalImageStorage:
    return LocalImageStorage(tmp_path / "store")


def test_save_read_delete_roundtrip(storage: LocalImageStorage, tmp_path: Path) -> None:
    key = storage.save(b"hello", category="references", extension=".jpg")

    assert key.startswith("references/") and key.endswith(".jpg")
    assert storage.read(key) == b"hello"
    assert (tmp_path / "store" / key).is_file()

    storage.delete(key)
    with pytest.raises(StorageError):
        storage.read(key)
    storage.delete(key)  # deleting a missing key is a no-op


def test_generated_names_are_unique(storage: LocalImageStorage) -> None:
    keys = {storage.save(b"x", category="references", extension=".png") for _ in range(20)}
    assert len(keys) == 20


@pytest.mark.parametrize(
    "key",
    [
        "../secret.jpg",
        "references/../../secret.jpg",
        "..\\..\\secret.jpg",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "references/not-a-generated-name.jpg",
        "references/" + "a" * 32 + ".exe",
        "",
    ],
)
def test_path_traversal_and_malformed_keys_rejected(storage: LocalImageStorage, key: str) -> None:
    with pytest.raises(StorageError):
        storage.read(key)
    with pytest.raises(StorageError):
        storage.delete(key)


@pytest.mark.parametrize("category", ["../x", "Refs", "a/b", "", "1abc"])
def test_invalid_category_rejected(storage: LocalImageStorage, category: str) -> None:
    with pytest.raises(StorageError):
        storage.save(b"x", category=category, extension=".jpg")


@pytest.mark.parametrize("extension", [".exe", "jpg", ".jpg/../x", ".svg"])
def test_invalid_extension_rejected(storage: LocalImageStorage, extension: str) -> None:
    with pytest.raises(StorageError):
        storage.save(b"x", category="references", extension=extension)
