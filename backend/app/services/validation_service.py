"""Turn matcher candidates + optional vision output into findings and a decision."""

from __future__ import annotations

from app.providers.openai_provider import VisionComparisonOutput, VisionObservation
from app.schemas.comparison import ComparisonResult
from app.schemas.finding import (
    ApproximateRegion,
    CandidateDifference,
    Finding,
    FindingKind,
    InspectionDecision,
    VisionStatus,
)

_ISSUE_KINDS = {
    FindingKind.POSSIBLE_MISSING,
    FindingKind.POSSIBLE_NEW,
    FindingKind.POSSIBLE_MOVED,
    FindingKind.POSSIBLE_DAMAGE,
    FindingKind.POSSIBLE_DEBRIS,
    FindingKind.POSSIBLE_STAIN,
}

_NOT_MISSING = {FindingKind.NOT_VISIBLE, FindingKind.POSSIBLY_OCCLUDED, FindingKind.INCONCLUSIVE}


class ValidationService:
    def validate(
        self,
        comparison: ComparisonResult,
        vision: VisionComparisonOutput | None,
        vision_status: VisionStatus,
        image_sizes: dict[str, tuple[int, int]],
        known_detection_ids: set[str],
    ) -> tuple[list[Finding], InspectionDecision, str]:
        findings = [self._from_candidate(c) for c in comparison.candidates]
        not_missing_ids = {
            f.before_detection_id
            for f in findings
            if f.kind in _NOT_MISSING and f.before_detection_id
        }

        if vision is not None and vision_status == VisionStatus.COMPLETED:
            extra: list[Finding] = []
            for i, obs in enumerate(vision.observations, start=1):
                finding = self._from_observation(
                    obs, i, image_sizes, known_detection_ids, not_missing_ids, findings
                )
                if finding is not None:
                    extra.append(finding)
            findings.extend(extra)

        decision = self._decide(comparison, findings, vision, vision_status)
        note = self._note(decision, vision_status)
        return findings, decision, note

    def _from_candidate(self, candidate: CandidateDifference) -> Finding:
        return Finding(
            id=f"find_{candidate.id}",
            kind=FindingKind(candidate.kind.value),
            label=candidate.label,
            source="matcher",
            before_image_id=candidate.before_image_id,
            after_image_id=candidate.after_image_id,
            before_detection_id=candidate.before_detection_id,
            after_detection_id=candidate.after_detection_id,
            region=None,
            evidence_strength=candidate.evidence_strength,
            requires_review=candidate.requires_review,
            explanation=candidate.explanation,
            evidence=list(candidate.evidence),
        )

    def _from_observation(
        self,
        obs: VisionObservation,
        index: int,
        image_sizes: dict[str, tuple[int, int]],
        known_detection_ids: set[str],
        not_missing_ids: set[str],
        existing: list[Finding],
    ) -> Finding | None:
        kind = FindingKind(obs.kind)
        detection_id = obs.related_detection_id
        if detection_id and detection_id not in known_detection_ids:
            detection_id = None

        if kind == FindingKind.POSSIBLE_MISSING and detection_id in not_missing_ids:
            kind = FindingKind.NOT_VISIBLE

        if kind == FindingKind.POSSIBLE_MISSING:
            after_detection_id = None  # never fabricate
            before_detection_id = detection_id if detection_id and detection_id.startswith("ref_") else None
        elif kind == FindingKind.POSSIBLE_NEW:
            before_detection_id = None
            after_detection_id = detection_id if detection_id and detection_id.startswith("new_") else None
        else:
            before_detection_id = detection_id if detection_id and detection_id.startswith("ref_") else None
            after_detection_id = detection_id if detection_id and detection_id.startswith("new_") else None

        if obs.related_candidate_id:
            for finding in existing:
                if finding.id == f"find_{obs.related_candidate_id}" or finding.id.endswith(
                    obs.related_candidate_id
                ):
                    finding.source = "both"
                    if obs.description and obs.description not in finding.explanation:
                        finding.explanation = f"{finding.explanation} Vision: {obs.description}"
                    return None

        region = self._region(obs, image_sizes)
        return Finding(
            id=f"find_vis_{index:03d}",
            kind=kind,
            label=obs.label[:80] or "visual difference",
            source="vision",
            before_image_id=obs.before_image_id,
            after_image_id=obs.after_image_id,
            before_detection_id=before_detection_id,
            after_detection_id=after_detection_id,
            region=region,
            evidence_strength=obs.confidence if obs.confidence in ("low", "medium") else "low",
            requires_review=True,
            explanation=obs.description,
            evidence=["openai_vision"],
        )

    @staticmethod
    def _region(
        obs: VisionObservation, image_sizes: dict[str, tuple[int, int]]
    ) -> ApproximateRegion | None:
        if obs.region is None:
            return None
        size = image_sizes.get(obs.region.image_id)
        if size is None:
            return None
        width, height = size
        x1, x2 = sorted((obs.region.x1, obs.region.x2))
        y1, y2 = sorted((obs.region.y1, obs.region.y2))
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            return None
        return ApproximateRegion(
            image_id=obs.region.image_id,
            bbox=(x1 * width, y1 * height, x2 * width, y2 * height),
            approximate=True,
        )

    @staticmethod
    def _decide(
        comparison: ComparisonResult,
        findings: list[Finding],
        vision: VisionComparisonOutput | None,
        vision_status: VisionStatus,
    ) -> InspectionDecision:
        if comparison.summary.coverage == "insufficient":
            return InspectionDecision.INCONCLUSIVE
        if vision_status == VisionStatus.FAILED:
            return InspectionDecision.MANUAL_REVIEW_REQUIRED
        issues = [f for f in findings if f.kind in _ISSUE_KINDS]
        if issues:
            return InspectionDecision.POTENTIAL_ISSUES
        if vision_status == VisionStatus.SKIPPED:
            return InspectionDecision.MANUAL_REVIEW_REQUIRED
        if vision is not None and (vision.cannot_assess or vision.viewpoint_changed):
            return InspectionDecision.INCONCLUSIVE
        return InspectionDecision.NO_ISSUE_DETECTED

    @staticmethod
    def _note(decision: InspectionDecision, vision_status: VisionStatus) -> str:
        if decision == InspectionDecision.NO_ISSUE_DETECTED:
            return (
                "No object-level or reported visual issues in the images that could be compared. "
                "This is not a certification that the property is clean or undamaged."
            )
        if decision == InspectionDecision.INCONCLUSIVE:
            return "The images do not support a confident comparison. Upload another photo of the same view."
        if vision_status == VisionStatus.SKIPPED:
            return (
                "Object matching finished. Vision analysis was skipped (no API key), so stains, "
                "debris and damage were not assessed."
            )
        if vision_status == VisionStatus.FAILED:
            return "Object matching finished, but vision analysis failed. Review the images manually."
        return "Potential issues were flagged for review. They are candidates, not confirmed faults."
