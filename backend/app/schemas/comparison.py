"""Normalized outputs of the viewpoint-assessment and matching stages."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.finding import CandidateDifference


class AlignmentStatus(StrEnum):
    ALIGNED = "aligned"  # reliable alignment, similar viewpoint
    DIFFERENT_VIEWPOINT = "different_viewpoint"  # reliable alignment, noticeably different camera
    INSUFFICIENT_OVERLAP = "insufficient_overlap"  # could not establish a reliable correspondence


class ImageQuality(BaseModel):
    image_id: str
    brightness: float  # mean gray level, 0-255
    sharpness: float  # variance of the Laplacian (higher = sharper)
    issues: list[str] = Field(default_factory=list)


class PairAssessment(BaseModel):
    """Whether (and how well) one inspection image can be related to one reference image.

    The homography behind this is only a weak, planar approximation of the camera change; a
    "reliable" pair means the images demonstrably overlap, not that every object can be located.
    """

    reference_image_id: str
    inspection_image_id: str
    status: AlignmentStatus
    reliable: bool
    good_matches: int
    inliers: int
    inlier_ratio: float
    # Fraction of the reference view / inspection view covered by the other image (None if unknown).
    reference_overlap: float | None = None
    inspection_overlap: float | None = None
    scale: float | None = None
    rotation_deg: float | None = None
    notes: list[str] = Field(default_factory=list)


class ReferenceCoverage(BaseModel):
    reference_image_id: str
    best_overlap: float | None
    covered: Literal["full", "partial", "none"]
    best_inspection_image_id: str | None = None


class ObjectMatch(BaseModel):
    before_detection_id: str | None
    after_detection_id: str | None
    method: Literal["spatial", "features"]
    detail: str


class ComparisonSummary(BaseModel):
    valid: bool
    coverage: Literal["full", "partial", "insufficient"]
    needs_additional_image: bool
    recommendation: str | None = None
    reference_coverage: list[ReferenceCoverage]
    pairs: list[PairAssessment]
    quality: list[ImageQuality]
    quality_issues: list[str] = Field(default_factory=list)
    unaligned_inspection_images: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    summary: ComparisonSummary
    matches: list[ObjectMatch]
    candidates: list[CandidateDifference]
    matching_version: str
