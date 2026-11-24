from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.reference import LocationType


class InspectionStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InspectionDecision(StrEnum):
    NO_ISSUE_DETECTED = "no_issue_detected"
    POTENTIAL_ISSUES = "potential_issues"
    INCONCLUSIVE = "inconclusive"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class VisionStatus(StrEnum):
    SKIPPED = "skipped"
    COMPLETED = "completed"
    FAILED = "failed"


class InspectionSummary(BaseModel):
    id: str
    property_id: str
    location: LocationType
    status: InspectionStatus
    created_at: datetime
    completed_at: datetime | None = None
    reference_image_count: int
    inspection_image_count: int
    change_count: int | None = None
    decision: InspectionDecision | None = None


class ChangeBox(BaseModel):
    """A described change, placed on one photo. Coordinates are fractions of that photo."""

    image_id: str
    kind: Literal["damage", "missing", "new", "moved"]
    description: str
    x1: float
    y1: float
    x2: float
    y2: float
    origin_x: float | None = None
    origin_y: float | None = None


class InspectionImageOut(BaseModel):
    """`image_id` is only a URL key: ref_1.. for saved photos, new_1.. for the upload."""

    image_id: str
    role: Literal["reference", "inspection"]
    original_filename: str
    width: int
    height: int
    image_url: str


class InspectionDetail(InspectionSummary):
    failed_stage: str | None = None
    error_message: str | None = None
    vision_model: str | None = None
    images: list[InspectionImageOut]


class InspectionResults(BaseModel):
    """Damage, missing items, and new items. Boxes are included only for box mode."""

    inspection_id: str
    property_id: str
    location: LocationType
    status: InspectionStatus
    failed_stage: str | None = None
    error_message: str | None = None
    mode: Literal["boxes"] = "boxes"
    images: list[InspectionImageOut]
    damage: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    boxes: list[ChangeBox] = Field(default_factory=list)
    # The one before photo chosen for this comparison. Other saved angles are omitted.
    matched_image_id: str | None = None
    summary: str | None = None
    views_compared: str | None = None
    decision: InspectionDecision | None = None
    vision_status: VisionStatus | None = None
    vision_error: str | None = None
    note: str | None = None


class InspectionListResponse(BaseModel):
    items: list[InspectionSummary]
    total: int
