from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.comparison import ComparisonSummary, ObjectMatch
from app.schemas.detection import Detection
from app.schemas.finding import CandidateDifference, Finding, InspectionDecision, VisionStatus
from app.schemas.reference import LocationType


class InspectionStatus(StrEnum):
    PROCESSING = "processing"
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
    # None until the comparison has completed.
    coverage: Literal["full", "partial", "insufficient"] | None = None
    candidate_count: int | None = None
    finding_count: int | None = None
    decision: InspectionDecision | None = None


class InspectionImageOut(BaseModel):
    """`image_id` is stable within the inspection: ref_1.. for references, new_1.. for uploads."""

    image_id: str
    role: Literal["reference", "inspection"]
    original_filename: str
    width: int
    height: int
    image_url: str


class InspectionImageResult(InspectionImageOut):
    detections: list[Detection] = Field(default_factory=list)


class DetectorRunInfo(BaseModel):
    """The detector configuration that produced this inspection's detections."""

    name: str
    conf_threshold: float
    iou_threshold: float


class InspectionModels(BaseModel):
    detector: DetectorRunInfo
    matching_version: str | None = None
    vision_model: str | None = None


class InspectionDetail(InspectionSummary):
    failed_stage: str | None = None
    error_message: str | None = None
    models: InspectionModels
    images: list[InspectionImageOut]


class InspectionResults(BaseModel):
    """Everything produced so far for an inspection.

    `candidates` are leads for review from the deterministic matching stage, not verified
    findings. A final decision is produced by later stages (OpenAI comparison + validation).
    """

    inspection_id: str
    property_id: str
    location: LocationType
    status: InspectionStatus
    failed_stage: str | None = None
    error_message: str | None = None
    models: InspectionModels
    images: list[InspectionImageResult]
    # None unless the inspection completed.
    comparison: ComparisonSummary | None = None
    matches: list[ObjectMatch] = Field(default_factory=list)
    candidates: list[CandidateDifference] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    decision: InspectionDecision | None = None
    vision_status: VisionStatus | None = None
    vision_error: str | None = None
    note: str | None = None


class InspectionListResponse(BaseModel):
    items: list[InspectionSummary]
    total: int
