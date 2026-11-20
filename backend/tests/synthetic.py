"""Synthetic before/after scenes with known ground truth, for testing the matching stage.

A "world" is a richly textured image with textured rectangular "objects" pasted onto it. A camera
is a homography; the reference and inspection images are the world seen through two cameras.
Detections are computed from ground-truth boxes (standing in for YOLO) so the tests exercise the
matching rules rather than the detector.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from app.schemas.detection import Detection, ImageDetections
from app.services.matching_service import ComparisonImage

W, H = 960, 720


def textured_world(seed: int = 1, size: tuple[int, int] = (W, H)) -> np.ndarray:
    """Multi-scale smooth noise plus random shapes: plenty of distinctive corners for SIFT."""
    rng = np.random.default_rng(seed)
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    for cell, weight in ((64, 0.5), (24, 0.35), (8, 0.25)):
        small = rng.integers(0, 255, (max(height // cell, 2), max(width // cell, 2), 3)).astype(np.float32)
        canvas += weight * cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    image = np.clip(canvas / 1.1, 0, 255).astype(np.uint8)
    for _ in range(140):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        if rng.random() < 0.5:
            cv2.circle(image, (x, y), int(rng.integers(4, 22)), color, -1)
        else:
            cv2.rectangle(image, (x, y), (x + int(rng.integers(6, 40)), y + int(rng.integers(6, 40))), color, -1)
    return image


def object_patch(seed: int, size: tuple[int, int] = (150, 120)) -> np.ndarray:
    """A strongly textured, distinctly coloured patch standing in for a household object."""
    rng = np.random.default_rng(1000 + seed)
    width, height = size
    base = np.array([rng.integers(40, 215) for _ in range(3)], dtype=np.float32)
    noise = rng.integers(0, 255, (height // 6, width // 6, 3)).astype(np.float32)
    patch = 0.45 * base + 0.55 * cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    cv2.rectangle(patch, (0, 0), (width - 1, height - 1), (10, 10, 10), 4)
    cv2.line(patch, (0, 0), (width - 1, height - 1), (250, 250, 250), 3)
    return patch


@dataclass
class WorldObject:
    label: str
    seed: int
    xy: tuple[int, int]
    size: tuple[int, int] = (150, 120)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.xy[0], self.xy[1], self.xy[0] + self.size[0], self.xy[1] + self.size[1])


def render_world(background: np.ndarray, objects: list[WorldObject]) -> np.ndarray:
    world = background.copy()
    for obj in objects:
        x, y = obj.xy
        patch = object_patch(obj.seed, obj.size)
        world[y : y + obj.size[1], x : x + obj.size[0]] = patch
    return world


def camera(
    scale: float = 1.0,
    rotation_deg: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    perspective: tuple[float, float] = (0.0, 0.0),
    size: tuple[int, int] = (W, H),
) -> np.ndarray:
    cx, cy = size[0] / 2, size[1] / 2
    theta = np.deg2rad(rotation_deg)
    cos, sin = np.cos(theta) * scale, np.sin(theta) * scale
    a = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)
    to_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    back = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)
    persp = np.array([[1, 0, 0], [0, 1, 0], [perspective[0], perspective[1], 1]], dtype=np.float64)
    return back @ persp @ a @ to_center


def view(world: np.ndarray, cam: np.ndarray, size: tuple[int, int] = (W, H)) -> np.ndarray:
    return cv2.warpPerspective(world, cam, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def project_box(
    box: tuple[float, float, float, float], cam: np.ndarray, size: tuple[int, int] = (W, H)
) -> tuple[float, float, float, float] | None:
    """Ground-truth bbox of a world box seen by `cam`, clipped to the image; None if not visible."""
    corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], dtype=np.float64)
    pts = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), cam).reshape(-1, 2)
    x1, y1, x2, y2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    area = (x2 - x1) * (y2 - y1)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, size[0]), min(y2, size[1])
    if cx2 - cx1 < 4 or cy2 - cy1 < 4 or (cx2 - cx1) * (cy2 - cy1) < 0.8 * area:
        return None  # not (almost entirely) in view, so a detector would not report it
    return (float(cx1), float(cy1), float(cx2), float(cy2))


def make_image(
    image_id: str,
    pixels: np.ndarray,
    items: list[tuple[str, tuple[float, float, float, float], float]],
) -> ComparisonImage:
    height, width = pixels.shape[:2]
    detections = [
        Detection(id=f"{image_id}_det_{i:03d}", label=label, confidence=conf, bbox=box)
        for i, (label, box, conf) in enumerate(items, start=1)
    ]
    return ComparisonImage(
        image_id=image_id,
        image=Image.fromarray(pixels),
        detections=ImageDetections(image_id=image_id, width=width, height=height, detections=detections),
    )


@dataclass
class Scenario:
    """Reference world + inspection world, each seen through its own camera."""

    background: np.ndarray = field(default_factory=lambda: textured_world(1))
    ref_objects: list[WorldObject] = field(default_factory=list)
    new_objects: list[WorldObject] | None = None  # defaults to the same objects
    new_background: np.ndarray | None = None  # defaults to the reference background
    new_background: np.ndarray | None = None  # defaults to the reference background
    ref_cam: np.ndarray = field(default_factory=camera)
    new_cam: np.ndarray = field(default_factory=lambda: camera(scale=1.08, rotation_deg=4, tx=18, ty=-12))
    ref_detect: bool = True
    detect_in_new: set[int] | None = None  # indexes of new_objects the detector "sees" (None = all)
    extra_new_detections: list[tuple[str, tuple[float, float, float, float], float]] = field(default_factory=list)
    new_pixel_transform: object = None  # optional callable(np.ndarray) -> np.ndarray

    def build(self) -> tuple[ComparisonImage, ComparisonImage]:
        new_objects = self.ref_objects if self.new_objects is None else self.new_objects
        ref_pixels = view(render_world(self.background, self.ref_objects), self.ref_cam)
        new_bg = self.background if self.new_background is None else self.new_background
        new_pixels = view(render_world(new_bg, new_objects), self.new_cam)
        if self.new_pixel_transform is not None:
            new_pixels = self.new_pixel_transform(new_pixels)  # type: ignore[operator]

        ref_items = []
        for obj in self.ref_objects:
            box = project_box(obj.box, self.ref_cam)
            if box and self.ref_detect:
                ref_items.append((obj.label, box, 0.9))

        new_items = []
        for index, obj in enumerate(new_objects):
            if self.detect_in_new is not None and index not in self.detect_in_new:
                continue
            box = project_box(obj.box, self.new_cam)
            if box:
                new_items.append((obj.label, box, 0.88))
        new_items.extend(self.extra_new_detections)

        return make_image("ref_1", ref_pixels, ref_items), make_image("new_1", new_pixels, new_items)
