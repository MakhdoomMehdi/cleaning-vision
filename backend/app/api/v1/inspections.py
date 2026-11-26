import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, Response, UploadFile

from app.api.deps import get_inspection_service
from app.api.uploads import read_limited_sync
from app.core.config import get_settings
from app.schemas.inspection import (
    InspectionDetail,
    InspectionListResponse,
    InspectionResults,
)
from app.schemas.reference import PROPERTY_ID_PATTERN, LocationType
from app.services.image_service import ImageValidationError
from app.services.inspection_service import (
    InspectionNotFoundError,
    InspectionService,
    NoReferencesError,
    TooManyReferencesError,
)
from app.services.uploads import UploadedFile
from app.storage.base import StorageError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inspections", tags=["inspections"])

InspectionServiceDep = Annotated[InspectionService, Depends(get_inspection_service)]


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Inspection not found.")


@router.post("", response_model=InspectionDetail, status_code=201)
def create_inspection(
    service: InspectionServiceDep,
    property_id: Annotated[str, Form(pattern=PROPERTY_ID_PATTERN)],
    location: Annotated[LocationType, Form()],
    files: Annotated[list[UploadFile], File(description="One or more inspection images")],
) -> InspectionDetail:
    """Run an inspection synchronously and return it.

    Input problems (invalid images) are rejected with a 4xx and nothing is
    stored. A location with no before photos is checked for guest readiness instead of compared.
    If a pipeline stage fails after the inspection was created, the inspection is kept
    with `status: "failed"`, `failed_stage` and `error_message` (HTTP 201): the request created
    a resource whose processing failed. The status is never left as "queued" or "processing".
    """
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
        inspection = service.create(property_id, location, uploads)
    except NoReferencesError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except TooManyReferencesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None
    except StorageError:
        raise HTTPException(status_code=500, detail="Could not store the uploaded images.") from None
    return service.detail(inspection)


@router.get("", response_model=InspectionListResponse)
def list_inspections(
    service: InspectionServiceDep,
    property_id: Annotated[str | None, Query(pattern=PROPERTY_ID_PATTERN)] = None,
    location: LocationType | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> InspectionListResponse:
    rows, total = service.list_inspections(property_id, location, limit, offset)
    return InspectionListResponse(items=service.summaries(rows), total=total)


@router.get("/{inspection_id}", response_model=InspectionDetail)
def get_inspection(inspection_id: UUID, service: InspectionServiceDep) -> InspectionDetail:
    try:
        return service.detail(service.get(str(inspection_id)))
    except InspectionNotFoundError:
        raise _not_found() from None


@router.get("/{inspection_id}/results", response_model=InspectionResults)
def get_inspection_results(inspection_id: UUID, service: InspectionServiceDep) -> InspectionResults:
    """Text report: damage, missing items, and new items."""
    try:
        return service.results(service.get(str(inspection_id)))
    except InspectionNotFoundError:
        raise _not_found() from None


@router.get("/{inspection_id}/images/{image_ref}")
def get_inspection_image(
    inspection_id: UUID,
    image_ref: Annotated[str, Path(pattern=r"^(ref|new)_[0-9]{1,3}$")],
    service: InspectionServiceDep,
) -> Response:
    try:
        row, data = service.get_image(str(inspection_id), image_ref)
    except InspectionNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found.") from None
    return Response(
        content=data,
        media_type=row.mime_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )
