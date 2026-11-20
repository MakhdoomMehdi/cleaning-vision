import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_detection_service, get_image_service
from app.api.uploads import read_limited
from app.providers.yolo_provider import DetectionProviderError
from app.schemas.detection import DetectResponse, SupportedClassesResponse
from app.services.detection_service import DetectionService
from app.services.image_service import ImageService, ImageValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("", response_model=DetectResponse)
async def detect(
    file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image")],
    images: Annotated[ImageService, Depends(get_image_service)],
    detector: Annotated[DetectionService, Depends(get_detection_service)],
    conf_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    iou_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> DetectResponse:
    """Run YOLO on a single image. Boxes are [x1, y1, x2, y2] in original-image pixels."""
    try:
        data = await read_limited(file, images.max_upload_bytes)
        validated = images.validate(data)
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None

    try:
        # Inference is blocking CPU work: keep it off the event loop.
        result = await run_in_threadpool(
            detector.detect,
            validated,
            image_id="img_001",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
    except DetectionProviderError as exc:
        # Detail is intentionally generic; the cause is already logged server-side.
        raise HTTPException(status_code=503, detail=f"Object detection unavailable: {exc}") from None

    return DetectResponse(
        **result.model_dump(),
        model=detector.model_info(conf_threshold, iou_threshold),
    )


@router.get("/classes", response_model=SupportedClassesResponse)
def supported_classes(
    detector: Annotated[DetectionService, Depends(get_detection_service)],
) -> SupportedClassesResponse:
    """Classes the detector can recognise. Anything else is unsupported, not 'absent'."""
    try:
        classes = detector.provider.class_names
    except DetectionProviderError as exc:
        raise HTTPException(status_code=503, detail=f"Object detection unavailable: {exc}") from None
    return SupportedClassesResponse(model=detector.provider.model_name, classes=classes)
