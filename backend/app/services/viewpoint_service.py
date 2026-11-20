"""Viewpoint & scene-overlap assessment between a reference and an inspection image.

Method: SIFT keypoints -> ratio-test matches -> RANSAC homography. The homography is a *weak,
planar approximation* of the camera change. It is only trusted when many mutually consistent
matches support it and the implied transform is physically sensible. It is never treated as proof
of reliable per-object correspondence across substantially different viewpoints.
"""

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.schemas.comparison import AlignmentStatus, ImageQuality, PairAssessment

logger = logging.getLogger(__name__)

FEATURE_MAX_SIDE = 1000
MAX_FEATURES = 2500
RATIO_TEST = 0.75
RANSAC_REPROJ_PX = 4.0  # in feature-image pixels

MIN_GOOD_MATCHES = 12
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.25
SCALE_RANGE = (0.25, 4.0)

ALIGNED_SCALE_RANGE = (0.8, 1.25)
ALIGNED_MAX_ROTATION_DEG = 10.0
ALIGNED_MIN_OVERLAP = 0.7

DARK_BRIGHTNESS = 45.0
BRIGHT_BRIGHTNESS = 215.0
BLURRY_SHARPNESS = 40.0


@dataclass(frozen=True)
class ImageFeatures:
    image_id: str
    width: int
    height: int
    scale: float  # feature-image size / original size (<= 1)
    points: np.ndarray  # (N, 2) float32, feature-image coordinates
    descriptors: np.ndarray | None
    quality: ImageQuality


@dataclass(frozen=True)
class PairAlignment:
    assessment: PairAssessment
    # 3x3 matrices in ORIGINAL pixel coordinates; None unless the pair is reliable.
    after_to_before: np.ndarray | None
    before_to_after: np.ndarray | None
    # RANSAC-consistent keypoints, in original-pixel coordinates of the REFERENCE image.
    inlier_points_before: np.ndarray


def assess_quality(image_id: str, gray: np.ndarray) -> ImageQuality:
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    issues: list[str] = []
    if brightness < DARK_BRIGHTNESS:
        issues.append("image is very dark (poor lighting)")
    elif brightness > BRIGHT_BRIGHTNESS:
        issues.append("image is overexposed")
    if sharpness < BLURRY_SHARPNESS:
        issues.append("image is blurry or has very little detail")
    return ImageQuality(
        image_id=image_id,
        brightness=round(brightness, 1),
        sharpness=round(sharpness, 1),
        issues=issues,
    )


def _rect_polygon(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def _overlap_fraction(quad: np.ndarray, width: int, height: int) -> float:
    """Fraction of the (width x height) rectangle covered by the convex `quad`."""
    area, _ = cv2.intersectConvexConvex(_rect_polygon(width, height), quad.astype(np.float32))
    return float(min(max(area / (width * height), 0.0), 1.0))


class ViewpointService:
    def extract(self, image_id: str, image: Image.Image) -> ImageFeatures:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        scale = min(1.0, FEATURE_MAX_SIDE / max(width, height))
        if scale < 1.0:
            gray = cv2.resize(
                gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
            )

        quality = assess_quality(image_id, gray)
        sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        points = (
            np.array([kp.pt for kp in keypoints], dtype=np.float32)
            if keypoints
            else np.zeros((0, 2), dtype=np.float32)
        )
        return ImageFeatures(
            image_id=image_id,
            width=width,
            height=height,
            scale=scale,
            points=points,
            descriptors=descriptors,
            quality=quality,
        )

    def assess(self, reference: ImageFeatures, inspection: ImageFeatures) -> PairAlignment:
        """Relate `inspection` ("after") to `reference` ("before")."""
        notes: list[str] = []

        def unreliable(good: int = 0, inliers: int = 0, ratio: float = 0.0) -> PairAlignment:
            return PairAlignment(
                assessment=PairAssessment(
                    reference_image_id=reference.image_id,
                    inspection_image_id=inspection.image_id,
                    status=AlignmentStatus.INSUFFICIENT_OVERLAP,
                    reliable=False,
                    good_matches=good,
                    inliers=inliers,
                    inlier_ratio=round(ratio, 3),
                    notes=notes,
                ),
                after_to_before=None,
                before_to_after=None,
                inlier_points_before=np.zeros((0, 2), dtype=np.float64),
            )

        if (
            reference.descriptors is None
            or inspection.descriptors is None
            or len(reference.points) < 2
            or len(inspection.points) < 2
        ):
            notes.append("too few visible features in one of the images (blank or low-texture)")
            return unreliable()

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(inspection.descriptors, reference.descriptors, k=2)
        good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]
        if len(good) < MIN_GOOD_MATCHES:
            notes.append(f"only {len(good)} feature matches; the images share too little visible content")
            return unreliable(good=len(good))

        src = np.float32([inspection.points[m.queryIdx] for m in good])  # after (feature px)
        dst = np.float32([reference.points[m.trainIdx] for m in good])  # before (feature px)
        h_feat, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_PX, maxIters=3000)
        if h_feat is None or mask is None:
            notes.append("no consistent geometric relation between the images could be found")
            return unreliable(good=len(good))

        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        ratio = inliers / len(good)
        if inliers < MIN_INLIERS or ratio < MIN_INLIER_RATIO:
            notes.append(
                f"weak geometric support ({inliers} consistent matches, ratio {ratio:.2f}); "
                "alignment not trusted"
            )
            return unreliable(good=len(good), inliers=inliers, ratio=ratio)

        # Feature-pixel homography -> original-pixel homography.
        s_after = np.diag([inspection.scale, inspection.scale, 1.0])
        s_before_inv = np.diag([1.0 / reference.scale, 1.0 / reference.scale, 1.0])
        h_orig = s_before_inv @ h_feat @ s_after

        sane, scale, rotation = self._sanity_check(h_orig, inspection.width, inspection.height, notes)
        if not sane:
            return unreliable(good=len(good), inliers=inliers, ratio=ratio)

        quad_after_in_before = transform_points(h_orig, _rect_polygon(inspection.width, inspection.height))
        h_inv = np.linalg.inv(h_orig)
        quad_before_in_after = transform_points(h_inv, _rect_polygon(reference.width, reference.height))
        reference_overlap = _overlap_fraction(quad_after_in_before, reference.width, reference.height)
        inspection_overlap = _overlap_fraction(quad_before_in_after, inspection.width, inspection.height)

        near_identity = (
            ALIGNED_SCALE_RANGE[0] <= scale <= ALIGNED_SCALE_RANGE[1]
            and abs(rotation) <= ALIGNED_MAX_ROTATION_DEG
            and reference_overlap >= ALIGNED_MIN_OVERLAP
        )
        status = AlignmentStatus.ALIGNED if near_identity else AlignmentStatus.DIFFERENT_VIEWPOINT
        if status is AlignmentStatus.DIFFERENT_VIEWPOINT:
            notes.append(
                "camera position/angle differs noticeably; object locations are only approximate"
            )

        inlier_pts_before = dst[inlier_mask] / reference.scale
        return PairAlignment(
            assessment=PairAssessment(
                reference_image_id=reference.image_id,
                inspection_image_id=inspection.image_id,
                status=status,
                reliable=True,
                good_matches=len(good),
                inliers=inliers,
                inlier_ratio=round(ratio, 3),
                reference_overlap=round(reference_overlap, 3),
                inspection_overlap=round(inspection_overlap, 3),
                scale=round(scale, 3),
                rotation_deg=round(rotation, 1),
                notes=notes,
            ),
            after_to_before=h_orig,
            before_to_after=h_inv,
            inlier_points_before=inlier_pts_before.astype(np.float64),
        )

    @staticmethod
    def _sanity_check(
        h: np.ndarray, width: int, height: int, notes: list[str]
    ) -> tuple[bool, float, float]:
        """Reject degenerate homographies (flips, extreme zoom, non-convex warps)."""
        if not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
            notes.append("degenerate geometric estimate rejected")
            return False, 0.0, 0.0
        h = h / h[2, 2]
        det = float(np.linalg.det(h[:2, :2]))
        if det <= 0:
            notes.append("implausible (mirrored) geometric estimate rejected")
            return False, 0.0, 0.0
        scale = math.sqrt(det)
        rotation = math.degrees(math.atan2(h[1, 0], h[0, 0]))
        if not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
            notes.append(f"implausible zoom factor {scale:.2f} rejected")
            return False, scale, rotation
        quad = transform_points(h, _rect_polygon(width, height)).astype(np.float32)
        if not cv2.isContourConvex(quad) or cv2.contourArea(quad) <= 1.0:
            notes.append("implausible (distorted) geometric estimate rejected")
            return False, scale, rotation
        return True, scale, rotation
