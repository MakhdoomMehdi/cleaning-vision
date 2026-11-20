"""OpenAI vision provider. It only calls the model; it does not decide what to store."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Protocol

from openai import APIError, APITimeoutError, OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from app.schemas.review import ChangeMark, ReviewMode, RoomReview

logger = logging.getLogger(__name__)

_MAX_SIDE = 1024


class VisionProviderError(Exception):
    """The vision model could not be called or returned unusable output."""


@dataclass
class ReviewImage:
    """A photo handed to the model. `caption` is plain language, never a database id."""

    caption: str
    image: Image.Image


class VisionReviewer(Protocol):
    @property
    def model_name(self) -> str: ...

    def review(
        self, before: list[ReviewImage], after: list[ReviewImage], mode: ReviewMode
    ) -> RoomReview: ...


class SemanticOutput(BaseModel):
    damage: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    summary: str = ""
    views_compared: str = ""
    could_compare: bool = True


class BoxedOutput(BaseModel):
    marks: list[ChangeMark] = Field(default_factory=list)
    summary: str = ""
    views_compared: str = ""
    could_compare: bool = True


def _jpeg_data_url(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, _MAX_SIDE / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _shared_rules() -> str:
    return (
        "You compare photos of one room before a guest stay with photos taken after.\n"
        "The BEFORE photos are different angles of that room. The AFTER photos are the "
        "inspection. First, use only the BEFORE photos that show the same part of the room "
        "as the AFTER photo. Ignore angles that do not overlap.\n"
        "A different camera angle is not proof that something is missing. If the AFTER "
        "photo shows a part of the room that was never photographed before, set "
        "could_compare to false and explain that in summary.\n"
        "Do not say the room is clean, undamaged, or ready.\n"
        "views_compared is one everyday sentence about which views you used, such as "
        "'the bed wall and the window'.\n"
    )


def _semantic_instructions() -> str:
    return (
        _shared_rules()
        + "Report changes as short plain phrases in three lists:\n"
        "- damage: scratches, dents, stains, broken items, or similar harm\n"
        "- missing: visible in a comparable BEFORE photo and gone AFTER\n"
        "- new: present AFTER and not in the comparable BEFORE photos\n"
        "Do not return boxes or coordinates.\n"
        "If those views show no change, leave the three lists empty and say so in summary."
    )


def _box_instructions() -> str:
    return (
        _shared_rules()
        + "For each real change, return one box and a short description of what changed there.\n"
        "- missing: box on the BEFORE photo, around the item that is gone\n"
        "- new: box on the AFTER photo, around the item that appeared\n"
        "- damage: box on the AFTER photo, around the scratch, dent, stain, or break\n"
        "side is 'before' or 'after'. photo_number matches the caption: 1 for "
        "'Before photo 1' or 'After photo 1'.\n"
        "x1, y1, x2, y2 are 0 to 1 on that photo, top-left then bottom-right.\n"
        "description says what is in the box, such as 'scratch on the nightstand' or "
        "'chair by the window is gone'.\n"
        "If nothing changed, return no marks and say so in summary."
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

    def review(
        self, before: list[ReviewImage], after: list[ReviewImage], mode: ReviewMode
    ) -> RoomReview:
        if not before or not after:
            return RoomReview(
                mode=mode,
                could_compare=False,
                summary="A before photo and an after photo are both required.",
            )

        content = _message(before, after)
        boxed = mode == ReviewMode.BOXES
        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=_box_instructions() if boxed else _semantic_instructions(),
                input=[{"role": "user", "content": content}],
                text_format=BoxedOutput if boxed else SemanticOutput,
            )
        except (APIError, APITimeoutError) as exc:
            logger.exception("OpenAI vision call failed")
            raise VisionProviderError("The vision model could not analyse the images.") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise VisionProviderError("The vision model returned empty structured output.")
        if isinstance(parsed, BoxedOutput):
            return RoomReview(
                mode=ReviewMode.BOXES,
                marks=parsed.marks,
                summary=parsed.summary,
                views_compared=parsed.views_compared,
                could_compare=parsed.could_compare,
            )
        return RoomReview(
            mode=ReviewMode.SEMANTIC,
            damage=parsed.damage,
            missing=parsed.missing,
            new=parsed.new,
            summary=parsed.summary,
            views_compared=parsed.views_compared,
            could_compare=parsed.could_compare,
        )


def _message(before: list[ReviewImage], after: list[ReviewImage]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {
            "type": "input_text",
            "text": (
                "These BEFORE photos are different angles of the room. "
                "Choose which ones show the same area as the AFTER photo."
            ),
        }
    ]
    for photo in before:
        content.append({"type": "input_text", "text": photo.caption})
        content.append(
            {"type": "input_image", "image_url": _jpeg_data_url(photo.image), "detail": "auto"}
        )
    content.append({"type": "input_text", "text": "AFTER photos, taken after the guest stay."})
    for photo in after:
        content.append({"type": "input_text", "text": photo.caption})
        content.append(
            {"type": "input_image", "image_url": _jpeg_data_url(photo.image), "detail": "auto"}
        )
    return content
