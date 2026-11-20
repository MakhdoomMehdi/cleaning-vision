"""Matching-stage tests on synthetic scenes with known ground truth (detector replaced by boxes)."""

import cv2
import numpy as np
import pytest

from app.schemas.comparison import ComparisonResult
from app.schemas.finding import CandidateKind
from app.services.matching_service import (
    MatchingService,
    inner_region,
    iou,
    project_bounds,
    visible_fraction,
)
from tests.synthetic import (
    Scenario,
    WorldObject,
    camera,
    make_image,
    object_patch,
    project_box,
    render_world,
    textured_world,
    view,
)

CHAIR = WorldObject("chair", seed=1, xy=(150, 200))
TV = WorldObject("tv", seed=2, xy=(600, 380))


@pytest.fixture(scope="module")
def service() -> MatchingService:
    return MatchingService()


def run(service: MatchingService, scenario: Scenario) -> ComparisonResult:
    ref, new = scenario.build()
    return service.compare([ref], [new])


def kinds(result: ComparisonResult) -> list[CandidateKind]:
    return [c.kind for c in result.candidates]


def matched_before(result: ComparisonResult) -> int:
    return sum(1 for m in result.matches if m.before_detection_id is not None)


# --------------------------------------------------------------------------- geometry helpers


def test_iou_and_visible_fraction() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert visible_fraction((-5, 0, 5, 10), 100, 100) == pytest.approx(0.5)
    assert visible_fraction((200, 0, 210, 10), 100, 100) == 0.0
    assert inner_region((0, 0, 100, 100)) == pytest.approx((15, 15, 85, 85))


def test_project_bounds_with_translation() -> None:
    shift = np.array([[1, 0, 30], [0, 1, -10], [0, 0, 1]], dtype=np.float64)
    assert project_bounds((0, 0, 50, 50), shift) == pytest.approx((30, -10, 80, 40))


# --------------------------------------------------------------------------- viewpoint changes


def test_camera_angle_change_alone_does_not_trigger_missing(service: MatchingService) -> None:
    """Same scene, different camera (zoom, rotation, shift, mild perspective): nothing changed."""
    scenario = Scenario(
        ref_objects=[CHAIR, TV],
        new_cam=camera(scale=1.12, rotation_deg=6, tx=25, ty=-18, perspective=(4e-5, -3e-5)),
    )
    result = run(service, scenario)

    assert result.summary.valid
    assert result.summary.pairs[0].reliable
    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)
    assert CandidateKind.POSSIBLE_NEW not in kinds(result)
    assert CandidateKind.POSSIBLE_MOVED not in kinds(result)
    assert matched_before(result) == 2


def test_lighting_change_and_noise_do_not_trigger_missing(service: MatchingService) -> None:
    rng = np.random.default_rng(5)

    def dim_and_noisy(pixels: np.ndarray) -> np.ndarray:
        dim = cv2.convertScaleAbs(pixels, alpha=0.7, beta=-15).astype(np.int16)
        return np.clip(dim + rng.normal(0, 4, dim.shape), 0, 255).astype(np.uint8)

    result = run(service, Scenario(ref_objects=[CHAIR, TV], new_pixel_transform=dim_and_noisy))

    assert result.summary.valid
    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)
    assert matched_before(result) == 2


def test_object_outside_new_field_of_view_is_not_visible_not_missing(
    service: MatchingService,
) -> None:
    """The camera moved so the TV left the frame: that must NOT become a missing-object claim."""
    result = run(service, Scenario(ref_objects=[CHAIR, TV], new_cam=camera(tx=430)))

    assert result.summary.pairs[0].reliable
    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)
    tv = [c for c in result.candidates if c.label == "tv"]
    assert [c.kind for c in tv] == [CandidateKind.NOT_VISIBLE]
    assert tv[0].requires_review
    assert "outside" in " ".join(tv[0].evidence) or "field of view" in " ".join(tv[0].evidence)
    assert matched_before(result) == 1  # the chair is still matched


# --------------------------------------------------------------------------- true differences


def test_removed_object_becomes_a_reviewable_missing_candidate_only(
    service: MatchingService,
) -> None:
    result = run(service, Scenario(ref_objects=[CHAIR, TV], new_objects=[TV]))

    missing = [c for c in result.candidates if c.kind is CandidateKind.POSSIBLE_MISSING]
    assert [c.label for c in missing] == ["chair"]
    candidate = missing[0]
    assert candidate.before_detection_id is not None
    assert candidate.after_detection_id is None  # no fabricated "after" box
    assert candidate.requires_review is True
    assert candidate.evidence_strength in ("low", "medium")  # never "high"
    assert "not confirmation" in candidate.explanation
    assert matched_before(result) == 1  # the TV is still matched


def test_occluded_object_is_not_called_missing(service: MatchingService) -> None:
    blocker = WorldObject("person", seed=99, xy=(105, 160), size=(240, 200))
    result = run(service, Scenario(ref_objects=[CHAIR, TV], new_objects=[blocker, TV]))

    chair = [c for c in result.candidates if c.label == "chair"]
    assert [c.kind for c in chair] == [CandidateKind.POSSIBLY_OCCLUDED]
    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)
    assert chair[0].after_detection_id is not None  # points at the covering detection


def test_detector_miss_is_not_reported_as_missing_when_image_features_agree(
    service: MatchingService,
) -> None:
    """The chair is still there but the detector found nothing in the new image."""
    result = run(
        service,
        Scenario(ref_objects=[CHAIR, TV], detect_in_new=set(), new_cam=camera(scale=1.05, tx=12)),
    )

    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)
    feature_matches = [m for m in result.matches if m.method == "features"]
    assert len([m for m in feature_matches if m.before_detection_id]) == 2
    assert all(m.after_detection_id is None for m in feature_matches if m.before_detection_id)


def test_new_object_is_reported_as_possible_new(service: MatchingService) -> None:
    extra = WorldObject("bottle", seed=7, xy=(400, 80), size=(90, 130))
    result = run(service, Scenario(ref_objects=[CHAIR], new_objects=[CHAIR, extra]))

    new = [c for c in result.candidates if c.kind is CandidateKind.POSSIBLE_NEW]
    assert [c.label for c in new] == ["bottle"]
    assert new[0].before_detection_id is None and new[0].after_detection_id is not None
    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)


def test_moved_object_links_both_detections(service: MatchingService) -> None:
    moved = WorldObject("chair", seed=1, xy=(620, 90))  # same object (same appearance), new place
    result = run(service, Scenario(ref_objects=[CHAIR, TV], new_objects=[moved, TV]))

    moves = [c for c in result.candidates if c.kind is CandidateKind.POSSIBLE_MOVED]
    assert len(moves) == 1
    assert moves[0].before_detection_id and moves[0].after_detection_id
    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)
    assert CandidateKind.POSSIBLE_NEW not in kinds(result)


def test_new_object_outside_reference_view_is_not_called_new(service: MatchingService) -> None:
    """A detection in an area the reference never photographed cannot be called 'new'."""
    chair = WorldObject("chair", seed=1, xy=(300, 250))
    far = WorldObject("bottle", seed=8, xy=(800, 560), size=(100, 130))
    # The reference camera is zoomed in (sees roughly x 160-800, y 120-600 of the world);
    # the inspection camera sees the whole world, including the bottle outside that area.
    result = run(
        service,
        Scenario(
            ref_objects=[chair],
            new_objects=[chair, far],
            ref_cam=camera(scale=1.5),
            new_cam=camera(),
        ),
    )

    assert result.summary.pairs[0].reliable
    bottle = [c for c in result.candidates if c.label == "bottle"]
    assert [c.kind for c in bottle] == [CandidateKind.NOT_VISIBLE]
    assert CandidateKind.POSSIBLE_NEW not in kinds(result)
    assert matched_before(result) == 1


# --------------------------------------------------------------------------- coverage / quality


def test_insufficient_overlap_yields_only_inconclusive_and_requests_another_image(
    service: MatchingService,
) -> None:
    other_room = textured_world(seed=77)
    result = run(
        service,
        Scenario(ref_objects=[CHAIR, TV], new_objects=[], new_background=other_room),
    )

    assert result.summary.valid is False
    assert result.summary.coverage == "insufficient"
    assert result.summary.needs_additional_image is True
    assert result.summary.recommendation and "another image" in result.summary.recommendation
    assert result.summary.unaligned_inspection_images == ["new_1"]
    assert not result.summary.pairs[0].reliable
    assert result.candidates and set(kinds(result)) == {CandidateKind.INCONCLUSIVE}
    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)


def test_blank_images_do_not_crash_and_are_inconclusive(service: MatchingService) -> None:
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    ref = make_image("ref_1", blank, [("chair", (100, 100, 200, 250), 0.9)])
    new = make_image("new_1", blank, [])
    result = service.compare([ref], [new])

    assert result.summary.coverage == "insufficient"
    assert kinds(result) == [CandidateKind.INCONCLUSIVE]
    assert any("low-texture" in note for note in result.summary.pairs[0].notes)


def test_partial_coverage_when_one_reference_view_is_not_covered(service: MatchingService) -> None:
    world = render_world(textured_world(1), [CHAIR])
    ref_a = make_image("ref_1", view(world, camera()), [("chair", project_box(CHAIR.box, camera()), 0.9)])
    other = render_world(textured_world(55), [WorldObject("sofa", seed=5, xy=(300, 300))])
    ref_b = make_image("ref_2", view(other, camera()), [("sofa", (300, 300, 450, 420), 0.9)])
    new_cam = camera(scale=1.05, tx=10)
    new = make_image("new_1", view(world, new_cam), [("chair", project_box(CHAIR.box, new_cam), 0.9)])

    result = service.compare([ref_a, ref_b], [new])

    assert result.summary.coverage == "partial"
    assert result.summary.valid and result.summary.needs_additional_image
    by_ref = {row.reference_image_id: row.covered for row in result.summary.reference_coverage}
    assert by_ref["ref_1"] in ("full", "partial") and by_ref["ref_2"] == "none"
    assert "ref_2" in (result.summary.recommendation or "")
    sofa = [c for c in result.candidates if c.label == "sofa"]
    assert [c.kind for c in sofa] == [CandidateKind.INCONCLUSIVE]
    assert CandidateKind.POSSIBLE_MISSING not in kinds(result)


def test_poor_lighting_is_reported(service: MatchingService) -> None:
    def very_dark(pixels: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(pixels, alpha=0.15, beta=0)

    result = run(service, Scenario(ref_objects=[CHAIR], new_pixel_transform=very_dark))

    assert any(issue.startswith("new_1") and "dark" in issue for issue in result.summary.quality_issues)
    dark = next(q for q in result.summary.quality if q.image_id == "new_1")
    assert dark.brightness < 45


def test_missing_candidate_is_low_strength_when_a_quality_issue_exists(
    service: MatchingService,
) -> None:
    def slightly_dark(pixels: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(pixels, alpha=0.3, beta=0)

    result = run(
        service,
        Scenario(ref_objects=[CHAIR, TV], new_objects=[TV], new_pixel_transform=slightly_dark),
    )
    for candidate in result.candidates:
        if candidate.kind is CandidateKind.POSSIBLE_MISSING:
            assert candidate.evidence_strength == "low"
            assert any("dark" in e for e in candidate.evidence)


def test_object_patch_is_deterministic() -> None:
    assert np.array_equal(object_patch(3), object_patch(3))
    assert not np.array_equal(object_patch(3), object_patch(4))
