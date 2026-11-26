"""OpenAI vision provider. It only calls the model; it does not decide what to store."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from openai import APIError, APITimeoutError, OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from app.schemas.review import ChangeMark, GuestIssue, ReviewMode, RoomReview

logger = logging.getLogger(__name__)

_Output = TypeVar("_Output", bound=BaseModel)

_MAX_SIDE = 1600


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

    def review(self, before: list[ReviewImage], after: list[ReviewImage]) -> RoomReview: ...

    def check_ready(self, photos: list[ReviewImage]) -> RoomReview: ...


class GuestIssueOutput(BaseModel):
    category: Literal["bedding", "cleanliness", "bathroom", "clutter", "damage", "setup"]
    severity: Literal["critical", "moderate", "minor"]
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)
    comment: str


class GuestReadyOutput(BaseModel):
    guest_ready: bool = False
    overall_score: int = Field(default=0, ge=0, le=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    image_quality_note: str | None = None
    issues: list[GuestIssueOutput] = Field(default_factory=list)


class BoxedOutput(BaseModel):
    matched_photo: int = Field(default=1, ge=1, le=12)
    marks: list[ChangeMark] = Field(default_factory=list)
    summary: str = ""
    views_compared: str = ""
    could_compare: bool = True


def _sent_size(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    scale = min(1.0, _MAX_SIDE / max(width, height))
    if scale < 1.0:
        return max(1, int(width * scale)), max(1, int(height * scale))
    return width, height


def _jpeg_data_url(image: Image.Image, quality: int) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, _MAX_SIDE / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _shared_rules() -> str:
    return (
        "You compare photos of one room before a guest stay with photos taken after.\n"
        "The BEFORE photos are different angles of that room. The AFTER photos are the "
        "inspection. First, use only the BEFORE photos that show the same part of the room "
        "as the AFTER photo. Ignore angles that do not overlap.\n"
        "A closer, wider, or slightly shifted camera is still the same view. An object is "
        "missing when the same surface is visible in both photos and the object is gone. "
        "Call it out of frame only when that surface itself is not in the AFTER photo. "
        "If the AFTER photo shows a part of the room that was never photographed before, set "
        "could_compare to false and explain that in summary.\n"
        "Do not say the room is clean, undamaged, or ready.\n"
        "views_compared is one everyday sentence about which views you used, such as "
        "'the bed wall and the window'.\n"
    )


def _box_instructions() -> str:
    return (
        _shared_rules()
        + "Work in this order.\n"
        "1. Pick the single BEFORE photo that shows the same view as the AFTER photo. "
        "matched_photo is that photo's number: 1 means 'Before photo 1'. Ignore every "
        "other BEFORE photo.\n"
        "2. List the distinct objects in that BEFORE photo that still fall inside the AFTER "
        "photo, such as a lamp, pillows, a remote, towels, bottles, and cups.\n"
        "3. Mark each listed object present, missing, or moved. Missing means the same "
        "surface is in both photos and the object is gone, even if the camera moved a little.\n"
        "4. Draw boxes only for objects you marked missing, moved, damaged, or new.\n"
        "Draw every box on the AFTER photo only. side is always 'after'. photo_number is "
        "the AFTER photo number, usually 1.\n"
        "Measure every box on the AFTER photo. Do not copy a position from the BEFORE photo. "
        "The camera may have moved, so the same spot is in a different place in the AFTER photo, "
        "and a copied position lands beside the object.\n"
        "Coordinates are fractions of that AFTER photo: x grows to the right, y grows "
        "downward, 0 is the top-left corner and 1 is the bottom-right. x1,y1 is the "
        "top-left corner of the object. x2,y2 is the bottom-right corner, not the width "
        "and not the height. The center of the box must sit on the object, or on the empty "
        "patch a missing object covered, not on the surface beside it.\n"
        "For damage, new, and moved, put the edges on the object's own silhouette. A box "
        "that covers the floor, a whole wall, or a lot of empty space around the object is wrong.\n"
        "- damage: one tight box around the scratch, dent, stain, or break\n"
        "- missing: one box on the bare spot where that object sat, including the empty "
        "surface. Do not cover the whole floor or a whole wall.\n"
        "- new: one tight box around the new object\n"
        "- moved: exactly one box, around the object where it is NOW. Also set origin_x "
        "and origin_y to the center of where that same object used to be, mapped onto "
        "this AFTER photo. Do not draw a second box for it, and do not also mark it "
        "missing or new. Leave origin_x and origin_y null for every other kind.\n"
        "description is a short label, such as 'scratch on the nightstand' or "
        "'remote missing from the nightstand'.\n"
        "If nothing changed, return no marks and say so in summary."
    )


def _missing_followup_instructions() -> str:
    return (
        "Look again only for objects that are in the BEFORE photo and gone from the AFTER photo.\n"
        "Missing means the same surface is visible in both photos and the object is gone, "
        "even if the camera is a little closer or shifted. A towel, remote, bottle, cup, "
        "pillow, or charger on a surface counts.\n"
        "Return a missing box on the bare spot where the object sat, including that empty "
        "surface. Do not cover the whole floor or a whole wall.\n"
        "Do not report damage, new objects, or moved objects.\n"
        "If every object is still there, or the surface itself is out of frame, return no marks.\n"
        "Every box is on the AFTER photo. side is 'after'. photo_number is 1. "
        "Coordinates are fractions of the AFTER photo, 0 at the top-left and 1 at the bottom-right. "
        "kind is 'missing'. Leave origin_x and origin_y null."
    )


class MissingOutput(BaseModel):
    marks: list[ChangeMark] = Field(default_factory=list)


class PlacedMark(BaseModel):
    """A corrected box, measured on a 0–1000 grid of the after photo."""

    index: int = Field(ge=1, le=12)
    x1: int = Field(ge=0, le=1000)
    y1: int = Field(ge=0, le=1000)
    x2: int = Field(ge=0, le=1000)
    y2: int = Field(ge=0, le=1000)
    origin_x: int | None = Field(default=None, ge=0, le=1000)
    origin_y: int | None = Field(default=None, ge=0, le=1000)


class PlacementOutput(BaseModel):
    marks: list[PlacedMark] = Field(default_factory=list)


_POOL_INTRO = (
    "These BEFORE photos are different angles of the room. "
    "Pick the single BEFORE photo that matches the AFTER photo. "
    "On that photo, list the objects still inside the AFTER frame, "
    "mark each one present, missing, or moved, then put every box on the AFTER photo."
)
_PAIR_INTRO = (
    "This BEFORE photo is the view that matches the AFTER photo. "
    "List the objects in the BEFORE photo that are still inside the AFTER frame, "
    "mark each one present, missing, or moved, then put every box on the AFTER photo."
)
def _readiness_instructions() -> str:
    return (
        "You are a hospitality quality inspector. Decide whether the room in the photo is "
        "ready for an incoming guest. There is no before photo. Judge only what you see, "
        "using the standard of a clean, well-kept hotel or rental room.\n"
        "Check bedding, cleanliness, the bathroom, clutter and leftovers, damage, and setup. "
        "Report only what is visibly present. Do not guess about anything hidden, cropped, or "
        "out of frame. If an area looks fine, do not flag it.\n"
        "If the photo is too dark, blurry, or cropped to judge, set image_quality_note and "
        "return no issues.\n"
        "Draw one tight box per distinct issue, on this photo. x_min,y_min is the top-left "
        "and x_max,y_max is the bottom-right, each a fraction from 0 to 1. A box must cover "
        "the problem itself, not the area beside it. For a large uncleaned area, box the "
        "whole affected area.\n"
        "comment says what is wrong and what is needed, in one short sentence, such as "
        "'Bedsheet is wrinkled and untucked; remake the bed.'\n"
        "severity is critical when the room is clearly not guest-ready, moderate when it "
        "needs fixing before arrival, and minor for a small touch-up.\n"
        "guest_ready is false when there is any critical issue or two or more moderate issues. "
        "If there are no issues, guest_ready is true and issues is empty."
    )


_MISSING_INTRO = (
    "The first look found no missing items. "
    "Look again for objects that are in the BEFORE photo and gone from the AFTER photo. "
    "Return only missing boxes on the bare spot. If none are gone, return no marks."
)


def _placement_instructions() -> str:
    return (
        "You only correct where each box sits. The findings are already decided.\n"
        "Coordinates are integers from 0 to 1000 on the AFTER photo. 0,0 is its top-left "
        "pixel and 1000,1000 is its bottom-right pixel. x1,y1 is the top-left corner of "
        "the object. x2,y2 is the bottom-right corner. They are not the width or the height.\n"
        "Look at the AFTER photo and cover the object, or the exact empty patch a missing "
        "object covered. The center of the box must lie on that patch. A box beside it, "
        "above it, or below it is wrong. A missing handkerchief gets a handkerchief-sized "
        "box on the patch it covered, not the table around that patch.\n"
        "The BEFORE photo only shows what the object looked like. Do not reuse its pixel "
        "position. The camera may have moved.\n"
        "index is the finding number you were given. For a moved finding, origin_x and "
        "origin_y are the center of where that object used to be, measured on the AFTER "
        "photo. Leave them null for every other kind."
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

    def review(self, before: list[ReviewImage], after: list[ReviewImage]) -> RoomReview:
        if not before or not after:
            return RoomReview(
                mode=ReviewMode.BOXES,
                could_compare=False,
                summary="A before photo and an after photo are both required.",
            )

        matched_photo = self._matched_photo(before, after)
        focused = ReviewImage(caption="Before photo 1", image=before[matched_photo - 1].image)
        report = self._compare_pair(focused, after, matched_photo)
        if report.could_compare and not any(mark.kind == "missing" for mark in report.marks):
            report = self._add_missing_followup(report, focused, after)
        if report.could_compare and report.marks:
            report = self._place_marks(report, focused, after)
        return report

    def check_ready(self, photos: list[ReviewImage]) -> RoomReview:
        if not photos:
            return RoomReview(
                readiness=True,
                could_compare=False,
                summary="A photo of the room is required.",
            )
        issues: list[GuestIssue] = []
        summaries: list[str] = []
        notes: list[str] = []
        scores: list[int] = []
        for index, photo in enumerate(photos, start=1):
            parsed = self._parse(
                instructions=_readiness_instructions(),
                content=_message(
                    [],
                    [photo],
                    intro="Judge this photo on its own. There is no before photo.",
                    before_detail="high",
                    after_detail="high",
                    quality=90,
                ),
                text_format=GuestReadyOutput,
            )
            note = " ".join((parsed.image_quality_note or "").split())
            if note and not parsed.issues:
                notes.append(note)
                continue
            scores.append(parsed.overall_score)
            summary = " ".join(parsed.summary.split())
            if summary:
                summaries.append(summary)
            for issue in parsed.issues:
                comment = " ".join(issue.comment.split())
                if not comment:
                    continue
                x1, x2 = sorted((issue.x_min, issue.x_max))
                y1, y2 = sorted((issue.y_min, issue.y_max))
                if x2 - x1 < 0.01 or y2 - y1 < 0.01:
                    continue
                issues.append(
                    GuestIssue(
                        category=issue.category,
                        severity=issue.severity,
                        photo_number=index,
                        comment=comment[:300],
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        if not scores and not issues:
            return RoomReview(
                readiness=True,
                could_compare=False,
                summary=notes[0] if notes else "The photo is not clear enough to judge.",
                image_quality_note=notes[0] if notes else "",
            )
        critical = sum(1 for issue in issues if issue.severity == "critical")
        moderate = sum(1 for issue in issues if issue.severity == "moderate")
        return RoomReview(
            readiness=True,
            could_compare=True,
            guest_ready=critical == 0 and moderate < 2,
            overall_score=min(scores) if scores else 0,
            summary=" ".join(summaries)[:1000],
            image_quality_note=notes[0] if notes else "",
            issues=issues[:12],
        )

    def _matched_photo(self, before: list[ReviewImage], after: list[ReviewImage]) -> int:
        if len(before) == 1:
            return 1
        picked = self._parse(
            instructions=_box_instructions(),
            content=_message(
                before,
                after,
                intro=_POOL_INTRO,
                before_detail="auto",
                after_detail="high",
                quality=80,
            ),
            text_format=BoxedOutput,
        )
        if 1 <= picked.matched_photo <= len(before):
            return picked.matched_photo
        return 1

    def _compare_pair(
        self, focused: ReviewImage, after: list[ReviewImage], matched_photo: int
    ) -> RoomReview:
        parsed = self._parse(
            instructions=_box_instructions(),
            content=_message(
                [focused],
                after,
                intro=_PAIR_INTRO,
                before_detail="high",
                after_detail="high",
                quality=90,
            ),
            text_format=BoxedOutput,
        )
        return RoomReview(
            mode=ReviewMode.BOXES,
            marks=parsed.marks,
            matched_photo=matched_photo,
            summary=parsed.summary,
            views_compared=parsed.views_compared,
            could_compare=parsed.could_compare,
        )

    def _add_missing_followup(
        self, report: RoomReview, focused: ReviewImage, after: list[ReviewImage]
    ) -> RoomReview:
        try:
            parsed = self._parse(
                instructions=_missing_followup_instructions(),
                content=_message(
                    [focused],
                    after,
                    intro=_MISSING_INTRO,
                    before_detail="high",
                    after_detail="high",
                    quality=90,
                ),
                text_format=MissingOutput,
            )
        except VisionProviderError:
            logger.warning("Missing-item follow-up failed; keeping the pair comparison")
            return report
        return _merge_missing(report, parsed.marks)

    def _place_marks(
        self, report: RoomReview, focused: ReviewImage, after: list[ReviewImage]
    ) -> RoomReview:
        width, height = _sent_size(after[0].image)
        findings = "\n".join(
            f"{index}. {mark.kind}: {mark.description}"
            for index, mark in enumerate(report.marks, start=1)
        )
        try:
            parsed = self._parse(
                instructions=_placement_instructions(),
                content=_message(
                    [focused],
                    after,
                    intro=(
                        f"The AFTER photo is {width} by {height} pixels. "
                        "Place each finding on that photo. The box must cover the object "
                        "or the empty patch it left, not the area beside it.\n"
                        f"{findings}"
                    ),
                    before_detail="high",
                    after_detail="high",
                    quality=90,
                ),
                text_format=PlacementOutput,
            )
        except VisionProviderError:
            logger.warning("Box placement failed; keeping the first boxes")
            return report
        return _apply_placement(report, parsed.marks)

    def _parse(
        self,
        *,
        instructions: str,
        content: list[dict[str, object]],
        text_format: type[_Output],
    ) -> _Output:
        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                text_format=text_format,
            )
        except (APIError, APITimeoutError) as exc:
            logger.exception("OpenAI vision call failed")
            raise VisionProviderError("The vision model could not analyse the images.") from exc

        parsed = response.output_parsed
        if not isinstance(parsed, BaseModel):
            raise VisionProviderError("The vision model returned empty structured output.")
        return parsed


def _merge_missing(report: RoomReview, extra: list[ChangeMark]) -> RoomReview:
    existing = {" ".join(mark.description.split()).casefold() for mark in report.marks}
    marks = list(report.marks)
    for mark in extra:
        if mark.kind != "missing":
            continue
        label = " ".join(mark.description.split())
        if not label or label.casefold() in existing:
            continue
        marks.append(mark.model_copy(update={"kind": "missing", "description": label}))
        existing.add(label.casefold())
    return report.model_copy(update={"marks": marks})


def _apply_placement(report: RoomReview, placed: list[PlacedMark]) -> RoomReview:
    marks = list(report.marks)
    for item in placed:
        index = item.index - 1
        if index < 0 or index >= len(marks) or item.x1 == item.x2 or item.y1 == item.y2:
            continue
        current = marks[index]
        update: dict[str, float | None] = {
            "x1": item.x1 / 1000,
            "y1": item.y1 / 1000,
            "x2": item.x2 / 1000,
            "y2": item.y2 / 1000,
        }
        if current.kind == "moved" and item.origin_x is not None and item.origin_y is not None:
            update["origin_x"] = item.origin_x / 1000
            update["origin_y"] = item.origin_y / 1000
        marks[index] = current.model_copy(update=update)
    return report.model_copy(update={"marks": marks})


def _message(
    before: list[ReviewImage],
    after: list[ReviewImage],
    *,
    intro: str,
    before_detail: str,
    after_detail: str,
    quality: int,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "input_text", "text": intro}]
    for photo in before:
        content.append({"type": "input_text", "text": photo.caption})
        content.append(
            {
                "type": "input_image",
                "image_url": _jpeg_data_url(photo.image, quality),
                "detail": before_detail,
            }
        )
    content.append({"type": "input_text", "text": "AFTER photos, taken after the guest stay."})
    for photo in after:
        content.append({"type": "input_text", "text": photo.caption})
        content.append(
            {
                "type": "input_image",
                "image_url": _jpeg_data_url(photo.image, quality),
                "detail": after_detail,
            }
        )
    return content
