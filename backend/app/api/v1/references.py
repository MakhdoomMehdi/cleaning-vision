import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile

from app.api.deps import get_reference_service
from app.api.uploads import read_limited_sync
from app.core.config import get_settings
from app.db.models import ReferenceImage
from app.schemas.reference import (
    PROPERTY_ID_PATTERN,
    LocationListResponse,
    LocationOption,
    LocationType,
    ReferenceImageOut,
    ReferenceListResponse,
)
from app.services.image_service import ImageValidationError
from app.services.reference_service import (
    ReferenceNotFoundError,
    ReferenceService,
    UploadedFile,
)
from app.storage.base import StorageError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["references"])

ReferenceServiceDep = Annotated[ReferenceService, Depends(get_reference_service)]


def to_out(row: ReferenceImage) -> ReferenceImageOut:
    return ReferenceImageOut(
        id=row.id,
        property_id=row.property_id,
        location=LocationType(row.location),
        original_filename=row.original_filename,
        mime_type=row.mime_type,
        width=row.width,
        height=row.height,
        size_bytes=row.size_bytes,
        created_at=row.created_at,
        image_url=f"/api/v1/references/{row.id}/image",
    )


@router.get("/locations", response_model=LocationListResponse)
def list_locations() -> LocationListResponse:
    """Supported location types (single source of truth for the UI dropdown)."""
    return LocationListResponse(
        items=[
            LocationOption(value=loc, label=loc.value.replace("_", " ").title()) for loc in LocationType
        ]
    )


@router.post("/references", response_model=ReferenceListResponse, status_code=201)
def create_references(
    service: ReferenceServiceDep,
    property_id: Annotated[str, Form(pattern=PROPERTY_ID_PATTERN)],
    location: Annotated[LocationType, Form()],
    files: Annotated[list[UploadFile], File(description="One or more JPEG/PNG/WebP images")],
) -> ReferenceListResponse:
    """Register reference images for a property location. All files are stored, or none."""
    settings = get_settings()
    if len(files) > settings.max_images_per_request:
        raise HTTPException(
            status_code=422,
            detail=f"At most {settings.max_images_per_request} images per request.",
        )

    try:
        uploads = [
            UploadedFile(
                filename=f.filename or "image",
                data=read_limited_sync(f, service.max_upload_bytes),
            )
            for f in files
        ]
        rows = service.create_references(property_id, location, uploads)
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None
    except StorageError:
        raise HTTPException(status_code=500, detail="Could not store the uploaded images.") from None

    items = [to_out(row) for row in rows]
    return ReferenceListResponse(items=items, total=len(items))


@router.get("/references", response_model=ReferenceListResponse)
def list_references(
    service: ReferenceServiceDep,
    property_id: Annotated[str | None, Query(pattern=PROPERTY_ID_PATTERN)] = None,
    location: LocationType | None = None,
) -> ReferenceListResponse:
    items = [to_out(row) for row in service.list_references(property_id, location)]
    return ReferenceListResponse(items=items, total=len(items))


@router.delete("/references/{reference_id}", status_code=204)
def delete_reference(reference_id: UUID, service: ReferenceServiceDep) -> Response:
    """Delete one saved reference image and its file on disk."""
    try:
        service.delete_reference(str(reference_id))
    except ReferenceNotFoundError:
        raise HTTPException(status_code=404, detail="Reference image not found.") from None
    return Response(status_code=204)


@router.get("/references/{reference_id}/image")
def get_reference_image(reference_id: UUID, service: ReferenceServiceDep) -> Response:
    """Serve a stored reference image by ID (the client never supplies a file path)."""
    try:
        row, data = service.get_image(str(reference_id))
    except ReferenceNotFoundError:
        raise HTTPException(status_code=404, detail="Reference image not found.") from None
    return Response(
        content=data,
        media_type=row.mime_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )
