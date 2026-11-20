"""Normalized detection schemas shared between pipeline stages and the API."""

from pydantic import BaseModel, Field, model_validator


class Detection(BaseModel):
    """A single detected object. `bbox` is [x1, y1, x2, y2] in original-image pixels."""

    id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def _check_bbox(self) -> "Detection":
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box {self.bbox}")
        return self


class ImageDetections(BaseModel):
    """Detections for one image. `width`/`height` are the original (EXIF-oriented) pixel size."""

    image_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[Detection]

    @model_validator(mode="after")
    def _check_within_image(self) -> "ImageDetections":
        for det in self.detections:
            _, _, x2, y2 = det.bbox
            if x2 > self.width or y2 > self.height:
                raise ValueError(f"Detection {det.id} lies outside the image bounds")
        return self


class DetectionModelInfo(BaseModel):
    name: str
    conf_threshold: float
    iou_threshold: float
    device: str


class DetectResponse(ImageDetections):
    model: DetectionModelInfo
    # An empty list means "nothing of a supported class was found", NOT "the scene is empty".
    # See GET /api/v1/detect/classes for what the detector can recognise at all.
    note: str = (
        "Detections only cover the object classes the model was trained on. "
        "An object that is not detected may be of an unsupported class, occluded, "
        "or out of view; it is not evidence that the object is absent."
    )


class SupportedClassesResponse(BaseModel):
    model: str
    classes: list[str]
    note: str = (
        "The pretrained detector only knows these classes. Property-specific objects "
        "(e.g. toolboxes, wall marks) are unsupported, not absent."
    )
