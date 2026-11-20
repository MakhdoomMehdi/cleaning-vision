"""Conservative object matching between reference and inspection images.

Principles (the rules this stage must never break):
  * An unmatched detection is NEVER proof that an object is missing.
  * "possible_missing" is only emitted when ALL of these hold: the two images are reliably
    aligned, the object's projected region lies (almost) fully inside the inspection view,
    nothing else covers that region, and no same-class detection or consistent image features
    support the object's continued presence. Even then it is a low/medium-strength lead.
  * Otherwise the outcome is not_visible / possibly_occluded / inconclusive.
  * Spatial reasoning uses a homography, which is only a weak planar approximation, so every
    candidate says so and requires review.
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from app.schemas.comparison import (
    AlignmentStatus,
    ComparisonResult,
    ComparisonSummary,
    ImageQuality,
    ObjectMatch,
    PairAssessment,
    ReferenceCoverage,
)
from app.schemas.detection import Detection, ImageDetections
from app.schemas.finding import CandidateDifference, CandidateKind
from app.services.viewpoint_service import (
    ImageFeatures,
    PairAlignment,
    ViewpointService,
    transform_points,
)

logger = logging.getLogger(__name__)

MATCHING_VERSION = "matching-v1"

Box = tuple[float, float, float, float]

SPATIAL_IOU_MIN = 0.3
APPEARANCE_SIMILARITY_MIN = 0.7
FEATURE_SUPPORT_MIN = 8  # consistent keypoints inside an object's inner region
INNER_REGION_FRACTION = 0.7
IN_VIEW_MIN = 0.8  # share of an object's projected box that must be inside the other view
OCCLUSION_COVER_MIN = 0.5
COVERAGE_FULL = 0.7
COVERAGE_PARTIAL = 0.25
MEDIUM_STRENGTH_MIN_INLIERS = 30
MEDIUM_STRENGTH_MIN_DET_CONF = 0.5


@dataclass(frozen=True)
class ComparisonImage:
    """An image plus its (already identifier-qualified) detections."""

    image_id: str
    image: Image.Image
    detections: ImageDetections


@dataclass
class _Outcome:
    kind: str  # matched | moved | occluded | missing | new | not_visible
    pair: PairAssessment
    partner_id: str | None = None
    method: str = ""
    detail: str = ""
    quality_notes: list[str] = field(default_factory=list)
    other_image_id: str = ""

    def evidence(self) -> list[str]:
        return [self.detail, *self.quality_notes, *self.pair.notes]


# Lower rank wins when several image pairs disagree. Ordering is deliberately conservative:
# evidence that an object is still there beats evidence that it may be gone, and
# occlusion/outside-view beats "missing".
_BEFORE_RANK = {"matched": 0, "moved": 1, "occluded": 2, "missing": 3, "not_visible": 4}
_AFTER_RANK = {"matched": 0, "moved": 1, "new": 2, "not_visible": 3}


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: Box, b: Box) -> float:
    return box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def project_bounds(box: Box, matrix: np.ndarray) -> Box:
    """Axis-aligned bounds of the box corners after a homography."""
    corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    pts = transform_points(matrix, np.array(corners))
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def visible_fraction(box: Box, width: int, height: int) -> float:
    area = box_area(box)
    return intersection_area(box, (0.0, 0.0, float(width), float(height))) / area if area > 0 else 0.0


def inner_region(box: Box) -> Box:
    margin = (1.0 - INNER_REGION_FRACTION) / 2.0
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] + w * margin, box[1] + h * margin, box[2] - w * margin, box[3] - h * margin)


def count_points_in(points: np.ndarray, box: Box) -> int:
    if len(points) == 0:
        return 0
    inside = (points[:, 0] >= box[0]) & (points[:, 0] <= box[2]) & (points[:, 1] >= box[1]) & (points[:, 1] <= box[3])
    return int(inside.sum())


def appearance_similarity(img_a: np.ndarray, box_a: Box, img_b: np.ndarray, box_b: Box) -> float:
    """HSV colour-histogram correlation of two crops (weak evidence; lighting can shift it)."""

    def histogram(img: np.ndarray, box: Box) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        crop = img[max(y1, 0) : y2, max(x1, 0) : x2]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

    ha, hb = histogram(img_a, box_a), histogram(img_b, box_b)
    if ha is None or hb is None:
        return 0.0
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def _fmt(box: Box) -> str:
    return "[" + ", ".join(f"{v:.0f}" for v in box) + "]"


class MatchingService:
    def __init__(self, viewpoint: ViewpointService | None = None) -> None:
        self._viewpoint = viewpoint or ViewpointService()

    def compare(
        self, references: list[ComparisonImage], inspections: list[ComparisonImage]
    ) -> ComparisonResult:
        feats: dict[str, ImageFeatures] = {
            img.image_id: self._viewpoint.extract(img.image_id, img.image)
            for img in [*references, *inspections]
        }
        pixels = {img.image_id: np.asarray(img.image.convert("RGB")) for img in [*references, *inspections]}

        alignments: list[PairAlignment] = [
            self._viewpoint.assess(feats[ref.image_id], feats[new.image_id])
            for ref in references
            for new in inspections
        ]
        align_by_pair = {
            (a.assessment.reference_image_id, a.assessment.inspection_image_id): a for a in alignments
        }

        before_outcomes: dict[str, list[_Outcome]] = {}
        after_outcomes: dict[str, list[_Outcome]] = {}
        for ref in references:
            for new in inspections:
                align = align_by_pair[(ref.image_id, new.image_id)]
                if not align.assessment.reliable:
                    continue
                self._assess_pair(ref, new, align, pixels, feats, before_outcomes, after_outcomes)

        qualities = [feats[img.image_id].quality for img in [*references, *inspections]]
        summary = self._summarize(references, inspections, alignments, qualities)
        matches, candidates = self._build_candidates(
            references, inspections, before_outcomes, after_outcomes, summary
        )
        return ComparisonResult(
            summary=summary, matches=matches, candidates=candidates, matching_version=MATCHING_VERSION
        )

    # ------------------------------------------------------------------ per image pair

    def _assess_pair(
        self,
        ref: ComparisonImage,
        new: ComparisonImage,
        align: PairAlignment,
        pixels: dict[str, np.ndarray],
        feats: dict[str, ImageFeatures],
        before_outcomes: dict[str, list[_Outcome]],
        after_outcomes: dict[str, list[_Outcome]],
    ) -> None:
        assert align.after_to_before is not None and align.before_to_after is not None
        pair = align.assessment
        h_after_to_before, h_before_to_after = align.after_to_before, align.before_to_after
        before = list(ref.detections.detections)
        after = list(new.detections.detections)
        new_w, new_h = feats[new.image_id].width, feats[new.image_id].height
        ref_w, ref_h = feats[ref.image_id].width, feats[ref.image_id].height
        quality_notes = [
            f"{q.image_id}: {issue}"
            for q in (feats[ref.image_id].quality, feats[new.image_id].quality)
            for issue in q.issues
        ]

        def record(store: dict[str, list[_Outcome]], det_id: str, outcome: _Outcome) -> None:
            outcome.quality_notes = quality_notes
            store.setdefault(det_id, []).append(outcome)

        projected = {b.id: project_bounds(b.bbox, h_before_to_after) for b in before}
        unmatched_before = {b.id: b for b in before}
        unmatched_after = {a.id: a for a in after}

        # 1) Spatial match: same class and overlapping after projecting into the other view.
        spatial = sorted(
            (
                (iou(projected[b.id], a.bbox), b, a)
                for b in before
                for a in after
                if a.label == b.label
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, b, a in spatial:
            if score < SPATIAL_IOU_MIN:
                break
            if b.id in unmatched_before and a.id in unmatched_after:
                detail = f"same class '{b.label}', overlap {score:.2f} after viewpoint compensation"
                record(before_outcomes, b.id, _Outcome("matched", pair, a.id, "spatial", detail, other_image_id=new.image_id))
                record(after_outcomes, a.id, _Outcome("matched", pair, b.id, "spatial", detail, other_image_id=ref.image_id))
                del unmatched_before[b.id], unmatched_after[a.id]

        # 2) Feature support: consistent image features inside the object's region mean the
        #    same content is visible in both images, even if the detector missed it in one.
        for b in list(unmatched_before.values()):
            support = count_points_in(align.inlier_points_before, inner_region(b.bbox))
            if support >= FEATURE_SUPPORT_MIN:
                detail = f"{support} consistent image features inside the object's region in both images"
                record(before_outcomes, b.id, _Outcome("matched", pair, None, "features", detail, other_image_id=new.image_id))
                del unmatched_before[b.id]

        # 3) Possible move: same class, no spatial match, but similar appearance.
        appearance = sorted(
            (
                (appearance_similarity(pixels[ref.image_id], b.bbox, pixels[new.image_id], a.bbox), b, a)
                for b in unmatched_before.values()
                for a in unmatched_after.values()
                if a.label == b.label
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, b, a in appearance:
            if score < APPEARANCE_SIMILARITY_MIN:
                break
            if b.id in unmatched_before and a.id in unmatched_after:
                detail = f"same class '{b.label}' with similar appearance (colour similarity {score:.2f}) at a different position"
                record(before_outcomes, b.id, _Outcome("moved", pair, a.id, "appearance", detail, other_image_id=new.image_id))
                record(after_outcomes, a.id, _Outcome("moved", pair, b.id, "appearance", detail, other_image_id=ref.image_id))
                del unmatched_before[b.id], unmatched_after[a.id]

        # 4) Remaining reference objects: where would they be in the inspection image?
        for b in unmatched_before.values():
            proj = projected[b.id]
            visible = visible_fraction(proj, new_w, new_h)
            if visible < IN_VIEW_MIN:
                detail = (
                    f"estimated position {_fmt(proj)} is only {visible:.0%} inside the inspection image; "
                    "the object is probably outside the camera's field of view"
                )
                record(before_outcomes, b.id, _Outcome("not_visible", pair, detail=detail, other_image_id=new.image_id))
                continue
            cover = max(
                (
                    (intersection_area(proj, a.bbox) / box_area(proj), a)
                    for a in after
                    if a.label != b.label and box_area(proj) > 0
                ),
                key=lambda item: item[0],
                default=(0.0, None),
            )
            if cover[0] >= OCCLUSION_COVER_MIN and cover[1] is not None:
                detail = (
                    f"the expected region is {cover[0]:.0%} covered by a detected '{cover[1].label}' "
                    f"({cover[1].id}); the object may be hidden behind it"
                )
                record(before_outcomes, b.id, _Outcome("occluded", pair, cover[1].id, detail=detail, other_image_id=new.image_id))
                continue
            detail = (
                f"expected region {_fmt(proj)} is inside the inspection image, no '{b.label}' detected "
                "there and no consistent image features support its presence"
            )
            record(before_outcomes, b.id, _Outcome("missing", pair, detail=detail, other_image_id=new.image_id))

        # 5) Remaining inspection objects: were they in the reference view?
        for a in unmatched_after.values():
            proj = project_bounds(a.bbox, h_after_to_before)
            visible = visible_fraction(proj, ref_w, ref_h)
            if visible < IN_VIEW_MIN:
                detail = (
                    f"estimated position {_fmt(proj)} is only {visible:.0%} inside the reference image; "
                    "that area may not have been photographed before"
                )
                record(after_outcomes, a.id, _Outcome("not_visible", pair, detail=detail, other_image_id=ref.image_id))
                continue
            support = count_points_in(align.inlier_points_before, inner_region(proj))
            if support >= FEATURE_SUPPORT_MIN:
                detail = f"{support} consistent image features show this region was already present in the reference"
                record(after_outcomes, a.id, _Outcome("matched", pair, None, "features", detail, other_image_id=ref.image_id))
                continue
            detail = (
                f"'{a.label}' is in a region visible in the reference image (estimated {_fmt(proj)}) "
                "but nothing comparable was detected or supported by image features there"
            )
            record(after_outcomes, a.id, _Outcome("new", pair, detail=detail, other_image_id=ref.image_id))

    # ------------------------------------------------------------------ summary

    def _summarize(
        self,
        references: list[ComparisonImage],
        inspections: list[ComparisonImage],
        alignments: list[PairAlignment],
        qualities: list[ImageQuality],
    ) -> ComparisonSummary:
        pairs = [a.assessment for a in alignments]
        reliable = [p for p in pairs if p.reliable]

        coverage_rows: list[ReferenceCoverage] = []
        for ref in references:
            candidates = [p for p in reliable if p.reference_image_id == ref.image_id]
            best = max(candidates, key=lambda p: p.reference_overlap or 0.0, default=None)
            overlap = best.reference_overlap if best else None
            level = "none"
            if overlap is not None and overlap >= COVERAGE_FULL:
                level = "full"
            elif overlap is not None and overlap >= COVERAGE_PARTIAL:
                level = "partial"
            coverage_rows.append(
                ReferenceCoverage(
                    reference_image_id=ref.image_id,
                    best_overlap=overlap,
                    covered=level,  # type: ignore[arg-type]
                    best_inspection_image_id=best.inspection_image_id if best else None,
                )
            )

        usable = [row for row in coverage_rows if row.covered != "none"]
        if not usable:
            coverage = "insufficient"
        elif all(row.covered == "full" for row in coverage_rows):
            coverage = "full"
        else:
            coverage = "partial"

        aligned_inspections = {p.inspection_image_id for p in reliable}
        unaligned = [i.image_id for i in inspections if i.image_id not in aligned_inspections]

        recommendation: str | None = None
        if coverage == "insufficient":
            recommendation = (
                "The inspection images could not be reliably related to the reference images. "
                "Take another image from a similar position and angle, showing the same area, "
                "with good lighting."
            )
        elif coverage == "partial":
            weak = [row.reference_image_id for row in coverage_rows if row.covered != "full"]
            recommendation = (
                "Coverage of the reference area is incomplete. Capture another inspection image "
                f"showing the area seen in: {', '.join(weak)}."
            )
        if unaligned and coverage != "insufficient":
            recommendation = (recommendation or "") + (
                f" Inspection image(s) not alignable with any reference: {', '.join(unaligned)}."
            )

        quality_issues = [f"{q.image_id}: {issue}" for q in qualities for issue in q.issues]
        return ComparisonSummary(
            valid=coverage != "insufficient",
            coverage=coverage,  # type: ignore[arg-type]
            needs_additional_image=coverage != "full",
            recommendation=recommendation.strip() if recommendation else None,
            reference_coverage=coverage_rows,
            pairs=pairs,
            quality=qualities,
            quality_issues=quality_issues,
            unaligned_inspection_images=unaligned,
        )

    # ------------------------------------------------------------------ candidates

    def _build_candidates(
        self,
        references: list[ComparisonImage],
        inspections: list[ComparisonImage],
        before_outcomes: dict[str, list[_Outcome]],
        after_outcomes: dict[str, list[_Outcome]],
        summary: ComparisonSummary,
    ) -> tuple[list[ObjectMatch], list[CandidateDifference]]:
        matches: list[ObjectMatch] = []
        candidates: list[CandidateDifference] = []
        emitted_moves: set[tuple[str, str]] = set()
        reliable_refs = {p.reference_image_id for p in summary.pairs if p.reliable}
        reliable_news = {p.inspection_image_id for p in summary.pairs if p.reliable}

        def add(**kwargs: object) -> None:
            candidates.append(CandidateDifference(id=f"cand_{len(candidates) + 1:03d}", **kwargs))  # type: ignore[arg-type]

        for ref in references:
            for det in ref.detections.detections:
                outcomes = before_outcomes.get(det.id, [])
                if not outcomes:
                    reason = (
                        "no inspection image could be reliably aligned with this reference image"
                        if ref.image_id not in reliable_refs
                        else "no usable comparison was possible"
                    )
                    add(
                        kind=CandidateKind.INCONCLUSIVE,
                        label=det.label,
                        before_image_id=ref.image_id,
                        before_detection_id=det.id,
                        evidence_strength="low",
                        explanation=(
                            f"The '{det.label}' seen in the reference image could not be verified: {reason}. "
                            "This is not evidence that it is missing."
                        ),
                    )
                    continue

                best = min(outcomes, key=lambda o: _BEFORE_RANK[o.kind])
                common = dict(
                    label=det.label,
                    before_image_id=ref.image_id,
                    before_detection_id=det.id,
                    evidence=best.evidence(),
                )
                if best.kind == "matched":
                    matches.append(
                        ObjectMatch(
                            before_detection_id=det.id,
                            after_detection_id=best.partner_id,
                            method=best.method,  # type: ignore[arg-type]
                            detail=best.detail,
                        )
                    )
                elif best.kind == "moved":
                    assert best.partner_id is not None
                    emitted_moves.add((det.id, best.partner_id))
                    add(
                        kind=CandidateKind.POSSIBLE_MOVED,
                        after_image_id=best.other_image_id,
                        after_detection_id=best.partner_id,
                        evidence_strength="low",
                        explanation=(
                            f"A '{det.label}' that looks similar to the reference one appears at a different "
                            "position. It may have been moved (or this may be a different object of the same type)."
                        ),
                        **common,  # type: ignore[arg-type]
                    )
                elif best.kind == "occluded":
                    add(
                        kind=CandidateKind.POSSIBLY_OCCLUDED,
                        after_image_id=best.other_image_id,
                        after_detection_id=best.partner_id,
                        evidence_strength="low",
                        explanation=(
                            f"The '{det.label}' from the reference image was not detected, but its expected "
                            "location is covered by another object, so it cannot be judged missing."
                        ),
                        **common,  # type: ignore[arg-type]
                    )
                elif best.kind == "missing":
                    medium = (
                        best.pair.status is AlignmentStatus.ALIGNED
                        and best.pair.inliers >= MEDIUM_STRENGTH_MIN_INLIERS
                        and not best.quality_notes
                        and det.confidence >= MEDIUM_STRENGTH_MIN_DET_CONF
                    )
                    add(
                        kind=CandidateKind.POSSIBLE_MISSING,
                        after_image_id=best.other_image_id,
                        after_detection_id=None,  # never fabricate an "after" box
                        evidence_strength="medium" if medium else "low",
                        explanation=(
                            f"The '{det.label}' seen in the reference image was not found where expected in the "
                            "inspection image. This is a lead for human review, not confirmation: the detector "
                            "may have missed it, the viewpoint estimate may be off, or it may be hidden."
                        ),
                        **common,  # type: ignore[arg-type]
                    )
                else:  # not_visible
                    add(
                        kind=CandidateKind.NOT_VISIBLE,
                        after_image_id=best.other_image_id,
                        evidence_strength="low",
                        explanation=(
                            f"The '{det.label}' from the reference image appears to be outside the area shown in "
                            "the inspection image, so its presence cannot be assessed. Another image is needed."
                        ),
                        **common,  # type: ignore[arg-type]
                    )

        for new in inspections:
            for det in new.detections.detections:
                outcomes = after_outcomes.get(det.id, [])
                if not outcomes:
                    add(
                        kind=CandidateKind.INCONCLUSIVE,
                        label=det.label,
                        after_image_id=new.image_id,
                        after_detection_id=det.id,
                        evidence_strength="low",
                        explanation=(
                            f"The '{det.label}' in the inspection image could not be compared with the reference"
                            + (
                                ": this image could not be reliably aligned with any reference image."
                                if new.image_id not in reliable_news
                                else "."
                            )
                        ),
                    )
                    continue
                best = min(outcomes, key=lambda o: _AFTER_RANK[o.kind])
                common = dict(
                    label=det.label,
                    after_image_id=new.image_id,
                    after_detection_id=det.id,
                    evidence=best.evidence(),
                )
                if best.kind == "matched":
                    if best.partner_id is None:  # feature-only match, not already reported from the before side
                        matches.append(
                            ObjectMatch(
                                before_detection_id=None,
                                after_detection_id=det.id,
                                method="features",
                                detail=best.detail,
                            )
                        )
                elif best.kind == "moved":
                    assert best.partner_id is not None
                    if (best.partner_id, det.id) in emitted_moves:
                        continue
                    emitted_moves.add((best.partner_id, det.id))
                    add(
                        kind=CandidateKind.POSSIBLE_MOVED,
                        before_image_id=best.other_image_id,
                        before_detection_id=best.partner_id,
                        evidence_strength="low",
                        explanation=(
                            f"This '{det.label}' resembles one in the reference image but is at a different position; "
                            "it may have been moved."
                        ),
                        **common,  # type: ignore[arg-type]
                    )
                elif best.kind == "new":
                    add(
                        kind=CandidateKind.POSSIBLE_NEW,
                        before_image_id=best.other_image_id,
                        evidence_strength="low",
                        explanation=(
                            f"A '{det.label}' appears in the inspection image in an area that was visible in the "
                            "reference image, where nothing comparable was found. It may be new (or earlier hidden)."
                        ),
                        **common,  # type: ignore[arg-type]
                    )
                else:  # not_visible
                    add(
                        kind=CandidateKind.NOT_VISIBLE,
                        before_image_id=best.other_image_id,
                        evidence_strength="low",
                        explanation=(
                            f"The '{det.label}' in the inspection image is in an area the reference image does not "
                            "clearly show, so it cannot be called new."
                        ),
                        **common,  # type: ignore[arg-type]
                    )

        return matches, candidates
