"""YOLO provider. Contains no business logic; it only turns an image into raw detections."""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

logger = logging.getLogger(__name__)


class DetectionProviderError(Exception):
    """Detector failed (model could not load or inference raised)."""


@dataclass(frozen=True)
class RawDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in input-image pixels


class DetectionProvider(Protocol):
    """Interface the DetectionService depends on; swap implementations freely (and mock in tests)."""

    @property
    def model_name(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def class_names(self) -> list[str]: ...

    def detect(
        self, image: Image.Image, conf_threshold: float, iou_threshold: float
    ) -> list[RawDetection]: ...


class YoloProvider:
    """Ultralytics YOLO wrapper. The model is loaded lazily once and reused."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self._model_path = model_path
        self._device = device
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def device(self) -> str:
        return self._device

    @property
    def class_names(self) -> list[str]:
        model = self._get_model()
        return [str(name) for _, name in sorted(model.names.items())]

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from ultralytics import YOLO  # heavy import; keep it lazy

                        logger.info("Loading YOLO model %s on %s", self._model_path, self._device)
                        self._model = YOLO(self._model_path)
                    except Exception as exc:
                        logger.exception("Failed to load YOLO model")
                        raise DetectionProviderError(
                            f"Could not load YOLO model '{self._model_path}'."
                        ) from exc
        return self._model

    def detect(
        self, image: Image.Image, conf_threshold: float, iou_threshold: float
    ) -> list[RawDetection]:
        model = self._get_model()
        try:
            # A PIL image is interpreted as RGB by Ultralytics (numpy arrays would be BGR).
            # Results are mapped back to the pixel space of the image passed in.
            with self._lock:  # one inference at a time; the model object is not thread-safe
                results = model.predict(
                    image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self._device,
                    verbose=False,
                )
        except Exception as exc:
            logger.exception("YOLO inference failed")
            raise DetectionProviderError("YOLO inference failed.") from exc

        detections: list[RawDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for coords, conf, cls in zip(xyxy, confs, classes, strict=True):
                x1, y1, x2, y2 = (float(v) for v in coords)
                detections.append(
                    RawDetection(
                        label=str(result.names[int(cls)]),
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections
