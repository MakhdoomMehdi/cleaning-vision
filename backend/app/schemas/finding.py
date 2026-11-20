"""Candidate differences (matcher) and validated findings (matcher + vision).

Candidates are *leads*. Findings are what the UI reports. Detector boxes stay authoritative:
a missing object never gets a fabricated after-box. Vision regions are approximate hints only.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class CandidateKind(StrEnum):
    POSSIBLE_MISSING = "possible_missing"
    POSSIBLE_MOVED = "possible_moved"
    POSSIBLE_NEW = "possible_new"
    NOT_VISIBLE = "not_visible"
    POSSIBLY_OCCLUDED = "possibly_occluded"
    INCONCLUSIVE = "inconclusive"


class FindingKind(StrEnum):
    POSSIBLE_MISSING = "possible_missing"
    POSSIBLE_MOVED = "possible_moved"
    POSSIBLE_NEW = "possible_new"
    POSSIBLE_DAMAGE = "possible_damage"
    POSSIBLE_DEBRIS = "possible_debris"
    POSSIBLE_STAIN = "possible_stain"
    NOT_VISIBLE = "not_visible"
    POSSIBLY_OCCLUDED = "possibly_occluded"
    INCONCLUSIVE = "inconclusive"


class InspectionDecision(StrEnum):
    NO_ISSUE_DETECTED = "no_issue_detected"
    POTENTIAL_ISSUES = "potential_issues"
    INCONCLUSIVE = "inconclusive"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class CandidateDifference(BaseModel):
    id: str
    kind: CandidateKind
    label: str
    before_image_id: str | None = None
    after_image_id: str | None = None
    # A missing object has a before detection but NO after detection (never fabricated).
    before_detection_id: str | None = None
    after_detection_id: str | None = None
    # Rule-derived, NOT a probability. Never "high": an unmatched detection is not proof.
    evidence_strength: Literal["low", "medium"]
    requires_review: bool = True
    explanation: str
    evidence: list[str] = Field(default_factory=list)


class ApproximateRegion(BaseModel):
    """Pixel box on `image_id`. Always a hint from vision, never a detector box."""

    image_id: str
    bbox: tuple[float, float, float, float]
    approximate: Literal[True] = True


class Finding(BaseModel):
    id: str
    kind: FindingKind
    label: str
    source: Literal["matcher", "vision", "both"]
    before_image_id: str | None = None
    after_image_id: str | None = None
    before_detection_id: str | None = None
    after_detection_id: str | None = None
    region: ApproximateRegion | None = None
    evidence_strength: Literal["low", "medium"]
    requires_review: bool = True
    explanation: str
    evidence: list[str] = Field(default_factory=list)


class VisionStatus(StrEnum):
    SKIPPED = "skipped"
    COMPLETED = "completed"
    FAILED = "failed"
