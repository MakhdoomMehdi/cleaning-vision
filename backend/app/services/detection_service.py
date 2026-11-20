"""Detection stage: validated image in, normalized detections out."""

import logging

from app.providers.yolo_provider import DetectionProvider, RawDetection
from app.schemas.detection import Detection, DetectionModelInfo, ImageDetections
from app.services.image_service import ValidatedImage

logger = logging.getLogger(__name__)

MIN_BOX_SIDE_PX = 1.0


class DetectionService:
    def __init__(
        self,
        provider: DetectionProvider,
        default_conf_threshold: float,
        default_iou_threshold: float,
    ) -> None:
        self._provider = provider
        self._default_conf = default_conf_threshold
        self._default_iou = default_iou_threshold

    @property
    def provider(self) -> DetectionProvider:
        return self._provider

    def model_info(
        self, conf_threshold: float | None = None, iou_threshold: float | None = None
    ) -> DetectionModelInfo:
        return DetectionModelInfo(
            name=self._provider.model_name,
            conf_threshold=self._default_conf if conf_threshold is None else conf_threshold,
            iou_threshold=self._default_iou if iou_threshold is None else iou_threshold,
            device=self._provider.device,
        )

    def detect(
        self,
        image: ValidatedImage,
        image_id: str,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> ImageDetections:
        conf = self._default_conf if conf_threshold is None else conf_threshold
        iou = self._default_iou if iou_threshold is None else iou_threshold

        raw = self._provider.detect(image.image, conf_threshold=conf, iou_threshold=iou)
        detections = self._normalize(raw, image.width, image.height, conf)
        logger.info("Detected %d object(s) in %s", len(detections), image_id)

        return ImageDetections(
            image_id=image_id,
            width=image.width,
            height=image.height,
            detections=detections,
        )

    @staticmethod
    def _normalize(
        raw: list[RawDetection], width: int, height: int, min_confidence: float
    ) -> list[Detection]:
        """Clamp boxes to the image, drop invalid/degenerate ones, assign stable IDs."""
        valid: list[RawDetection] = []
        for item in raw:
            if not 0.0 <= item.confidence <= 1.0 or item.confidence < min_confidence:
                continue
            x1, y1, x2, y2 = item.bbox
            x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
            y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
            if x2 - x1 < MIN_BOX_SIDE_PX or y2 - y1 < MIN_BOX_SIDE_PX:
                logger.debug("Dropping degenerate box for %s", item.label)
                continue
            valid.append(
                RawDetection(
                    label=item.label,
                    confidence=item.confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        valid.sort(key=lambda d: d.confidence, reverse=True)
        return [
            Detection(
                id=f"det_{index:03d}",
                label=item.label,
                confidence=round(item.confidence, 4),
                bbox=tuple(round(v, 1) for v in item.bbox),  # type: ignore[arg-type]
            )
            for index, item in enumerate(valid, start=1)
        ]
