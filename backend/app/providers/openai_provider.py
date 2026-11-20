"""OpenAI vision provider. No business rules live here — it only calls the model."""

from __future__ import annotations

import base64
import io
import logging
from typing import Literal

from openai import APIError, APITimeoutError, OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from app.schemas.finding import CandidateDifference
from app.schemas.comparison import ComparisonSummary

logger = logging.getLogger(__name__)

_MAX_SIDE = 1024


class VisionProviderError(Exception):
    """The vision model could not be called or returned unusable output."""


class VisionRegion(BaseModel):
    """Normalised 0–1 box on the named image (after image for visible issues)."""

    image_id: str
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class VisionObservation(BaseModel):
    kind: Literal[
        "possible_missing",
        "possible_new",
        "possible_moved",
        "possible_damage",
        "possible_debris",
        "possible_stain",
        "not_visible",
        "possibly_occluded",
        "inconclusive",
    ]
    label: str
    description: str
    before_image_id: str | None = None
    after_image_id: str | None = None
    related_candidate_id: str | None = None
    related_detection_id: str | None = None
    region: VisionRegion | None = None
    confidence: Literal["low", "medium"] = "low"


class VisionComparisonOutput(BaseModel):
    observations: list[VisionObservation] = Field(default_factory=list)
    overall_notes: str = ""
    viewpoint_changed: bool = False
    cannot_assess: list[str] = Field(default_factory=list)


class PairImages(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    reference_image_id: str
    inspection_image_id: str
    reference: Image.Image
    inspection: Image.Image


def _jpeg_data_url(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, _MAX_SIDE / max(w, h))
    if scale < 1.0:
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _instructions() -> str:
    return (
        "You compare a property location BEFORE (reference) vs AFTER (inspection) for a "
        "cleaning/inspection check. Report only visual differences a reviewer should see: "
        "objects that appear gone, objects that appeared, objects that moved, damage "
        "(scratches, dents, broken items), stains/marks on walls or furniture, and debris "
        "or trash (paper, bottles, wood chips, dirt piles).\n"
        "Hard rules:\n"
        "- Never claim the property is completely clean.\n"
        "- A different camera angle, zoom, or crop is not proof something is missing. "
        "If it may be out of frame, occluded, or only the viewpoint changed, use "
        "not_visible or inconclusive — not possible_missing.\n"
        "- Do not invent detector bounding boxes or detection ids. Only reuse ids from the "
        "provided candidate/detection lists. related_detection_id must be one of those ids "
        "or null.\n"
        "- For stains, debris, damage, and other things with no detector box, set region as "
        "an approximate 0-1 box on the AFTER image. This is a hint, not a measurement.\n"
        "- If lighting or viewpoint makes you unsure, say so in cannot_assess / inconclusive.\n"
        "- Ignore tiny JPEG artefacts. Prefer fewer, high-signal observations."
    )


class OpenAIVisionProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries)

    @property
    def model_name(self) -> str:
        return self._model

    def compare(
        self,
        pairs: list[PairImages],
        summary: ComparisonSummary,
        candidates: list[CandidateDifference],
        known_detection_ids: list[str],
    ) -> VisionComparisonOutput:
        if not pairs:
            return VisionComparisonOutput(overall_notes="No image pairs were provided.")

        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": (
                    "Matcher coverage and viewpoint (deterministic, not from you):\n"
                    f"{summary.model_dump_json()}\n\n"
                    "Object-match candidates (leads, not proof):\n"
                    f"{[c.model_dump() for c in candidates]}\n\n"
                    f"Known detection ids (the only ids you may cite): {known_detection_ids}\n"
                ),
            }
        ]
        for pair in pairs:
            content.append(
                {
                    "type": "input_text",
                    "text": f"BEFORE / reference image {pair.reference_image_id}:",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": _jpeg_data_url(pair.reference),
                    "detail": "auto",
                }
            )
            content.append(
                {
                    "type": "input_text",
                    "text": f"AFTER / inspection image {pair.inspection_image_id}:",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": _jpeg_data_url(pair.inspection),
                    "detail": "auto",
                }
            )

        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=_instructions(),
                input=[{"role": "user", "content": content}],
                text_format=VisionComparisonOutput,
            )
        except (APIError, APITimeoutError) as exc:
            logger.exception("OpenAI vision call failed")
            raise VisionProviderError("The vision model could not analyse the images.") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise VisionProviderError("The vision model returned empty structured output.")
        return parsed
