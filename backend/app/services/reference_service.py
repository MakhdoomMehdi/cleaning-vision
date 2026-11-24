"""Reference-image management: validate, store on disk, persist metadata."""

import hashlib
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import InspectionImage, ReferenceImage
from app.schemas.reference import LocationType
from app.services.image_service import ImageService
from app.services.uploads import UploadedFile, display_filename, prepare_uploads
from app.storage.base import ImageStorage, StorageError

logger = logging.getLogger(__name__)

REFERENCE_CATEGORY = "references"

__all__ = [
    "ReferenceNotFoundError",
    "ReferenceService",
    "UploadedFile",
    "display_filename",
]


class ReferenceNotFoundError(Exception):
    pass


class ReferenceService:
    def __init__(self, db: Session, storage: ImageStorage, images: ImageService) -> None:
        self._db = db
        self._storage = storage
        self._images = images

    @property
    def max_upload_bytes(self) -> int:
        return self._images.max_upload_bytes

    def create_references(
        self, property_id: str, location: LocationType, files: list[UploadedFile]
    ) -> list[ReferenceImage]:
        """Store all files or none: every file is validated before anything is written."""
        prepared = prepare_uploads(self._images, files)

        saved_keys: list[str] = []
        rows: list[ReferenceImage] = []
        try:
            for item in prepared:
                key = self._storage.save(
                    item.file.data, category=REFERENCE_CATEGORY, extension=item.extension
                )
                saved_keys.append(key)
                row = ReferenceImage(
                    property_id=property_id,
                    location=location.value,
                    original_filename=item.display_name,
                    storage_key=key,
                    mime_type=item.mime_type,
                    width=item.width,
                    height=item.height,
                    size_bytes=len(item.file.data),
                    sha256=hashlib.sha256(item.file.data).hexdigest(),
                )
                self._db.add(row)
                rows.append(row)
            self._db.commit()
        except Exception:
            self._db.rollback()
            for key in saved_keys:  # don't leave orphaned files behind
                self._storage.delete(key)
            raise

        logger.info(
            "Stored %d reference image(s) for property=%s location=%s",
            len(rows),
            property_id,
            location.value,
        )
        return rows

    def list_references(
        self, property_id: str | None = None, location: LocationType | str | None = None
    ) -> list[ReferenceImage]:
        stmt = select(ReferenceImage).order_by(ReferenceImage.created_at, ReferenceImage.id)
        if property_id is not None:
            stmt = stmt.where(ReferenceImage.property_id == property_id)
        if location is not None:
            value = location.value if isinstance(location, LocationType) else location
            stmt = stmt.where(ReferenceImage.location == value)
        return list(self._db.scalars(stmt))

    def delete_reference(self, reference_id: str) -> None:
        """Remove the metadata row and the file on disk."""
        row = self._db.get(ReferenceImage, reference_id)
        if row is None:
            raise ReferenceNotFoundError(reference_id)
        key = row.storage_key
        linked = self._db.scalars(
            select(InspectionImage).where(InspectionImage.reference_image_id == row.id)
        )
        for image in linked:
            image.reference_image_id = None
        self._db.delete(row)
        self._db.commit()
        self._storage.delete(key)
        logger.info("Deleted reference image %s", reference_id)

    def get_image(self, reference_id: str) -> tuple[ReferenceImage, bytes]:
        row = self._db.get(ReferenceImage, reference_id)
        if row is None:
            raise ReferenceNotFoundError(reference_id)
        try:
            return row, self._storage.read(row.storage_key)
        except StorageError:
            logger.error("Reference %s has no readable file on disk", reference_id)
            raise ReferenceNotFoundError(reference_id) from None
