"""Shared FastAPI dependencies. Override these in tests via `app.dependency_overrides`."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.providers.openai_provider import OpenAIVisionProvider, VisionReviewer
from app.services.image_service import ImageService
from app.services.inspection_service import InspectionService
from app.services.reference_service import ReferenceService
from app.storage.base import ImageStorage
from app.storage.local_storage import LocalImageStorage


@lru_cache
def get_image_service() -> ImageService:
    settings = get_settings()
    return ImageService(
        max_upload_bytes=settings.max_upload_bytes,
        min_dimension=settings.min_image_dimension,
        max_dimension=settings.max_image_dimension,
    )


@lru_cache
def get_storage() -> ImageStorage:
    """Local disk today; return an S3-backed ImageStorage here later."""
    return LocalImageStorage(get_settings().upload_root)


@lru_cache
def get_vision_provider() -> VisionReviewer | None:
    settings = get_settings()
    if not settings.openai_configured:
        return None
    assert settings.openai_api_key is not None
    return OpenAIVisionProvider(
        api_key=settings.openai_api_key.get_secret_value().strip(),
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


def get_reference_service(
    db: Annotated[Session, Depends(get_db)],
    storage: Annotated[ImageStorage, Depends(get_storage)],
    images: Annotated[ImageService, Depends(get_image_service)],
) -> ReferenceService:
    return ReferenceService(db=db, storage=storage, images=images)


def get_inspection_service(
    db: Annotated[Session, Depends(get_db)],
    storage: Annotated[ImageStorage, Depends(get_storage)],
    images: Annotated[ImageService, Depends(get_image_service)],
    references: Annotated[ReferenceService, Depends(get_reference_service)],
    vision: Annotated[VisionReviewer | None, Depends(get_vision_provider)],
) -> InspectionService:
    return InspectionService(
        db=db,
        storage=storage,
        images=images,
        references=references,
        max_reference_images=get_settings().max_reference_images_per_inspection,
        vision=vision,
    )
