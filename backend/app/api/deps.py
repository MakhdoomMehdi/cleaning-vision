"""Shared FastAPI dependencies. Override these in tests via `app.dependency_overrides`."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.providers.openai_provider import OpenAIVisionProvider
from app.providers.yolo_provider import YoloProvider
from app.services.detection_service import DetectionService
from app.services.image_service import ImageService
from app.services.inspection_service import InspectionService
from app.services.matching_service import MatchingService
from app.services.reference_service import ReferenceService
from app.services.validation_service import ValidationService
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
def get_detection_service() -> DetectionService:
    """Single shared service so the YOLO model is loaded once and reused."""
    settings = get_settings()
    provider = YoloProvider(model_path=settings.yolo_model_path, device=settings.yolo_device)
    return DetectionService(
        provider=provider,
        default_conf_threshold=settings.yolo_conf_threshold,
        default_iou_threshold=settings.yolo_iou_threshold,
    )


@lru_cache
def get_storage() -> ImageStorage:
    """Local disk today; return an S3-backed ImageStorage here later."""
    return LocalImageStorage(get_settings().upload_root)


@lru_cache
def get_matching_service() -> MatchingService:
    return MatchingService()


@lru_cache
def get_vision_provider() -> OpenAIVisionProvider | None:
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


@lru_cache
def get_validation_service() -> ValidationService:
    return ValidationService()


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
    detector: Annotated[DetectionService, Depends(get_detection_service)],
    matcher: Annotated[MatchingService, Depends(get_matching_service)],
    vision: Annotated[OpenAIVisionProvider | None, Depends(get_vision_provider)],
    validator: Annotated[ValidationService, Depends(get_validation_service)],
) -> InspectionService:
    return InspectionService(
        db=db,
        storage=storage,
        images=images,
        references=references,
        detector=detector,
        matcher=matcher,
        max_reference_images=get_settings().max_reference_images_per_inspection,
        vision=vision,
        validator=validator,
    )
