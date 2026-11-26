"""OpenAI vision provider. It only calls the model; it does not decide what to store."""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Literal, Protocol

from openai import APIError, APITimeoutError, OpenAI
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from app.providers.grid import (
    Frame,
    box_for_cells,
    boxes_for_cells,
    draw_grid,
    issue_color,
    last_cell,
)
from app.schemas.review import ChangeMark, GuestIssue, ReviewMode, RoomReview

logger = logging.getLogger(__name__)

_MAX_SIDE = 1600
_TEMPERATURE = 0.2


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


class _MatchOutput(BaseModel):
    matched_photo: int = 1


class _GridMark(BaseModel):
    kind: Literal["damage", "missing", "new", "moved"] = "damage"
    description: str = ""
    photo_number: int = 1
    cells: list[str] = Field(default_factory=list)
    origin_cells: list[str] = Field(default_factory=list)


class _ComparisonOutput(BaseModel):
    matched_photo: int = 1
    could_compare: bool = True
    summary: str = ""
    views_compared: str = ""
    marks: list[_GridMark] = Field(default_factory=list)


class _MissingCells(BaseModel):
    marks: list[_GridMark] = Field(default_factory=list)


class _ReadyIssue(BaseModel):
    category: Literal["bedding", "cleanliness", "bathroom", "clutter", "damage", "setup"] = "cleanliness"
    severity: Literal["critical", "moderate", "minor"] = "moderate"
    cells: list[str] = Field(default_factory=list)
    comment: str = ""


class _ReadyOutput(BaseModel):
    guest_ready: bool = False
    overall_score: int = 0
    confidence: float = 0
    summary: str = ""
    image_quality_note: str | None = None
    issues: list[_ReadyIssue] = Field(default_factory=list)


def _readiness_instructions() -> str:
    end = last_cell()
    return (
        "You are a hospitality quality inspector. Decide if the room in the image is "
        "READY FOR AN INCOMING GUEST. No reference image: judge only what you see.\n"
        f"The image has a red grid. Columns are A through {end[0]} and rows are 1 through "
        f"{end[1:]}. Each cell is labeled like A1, B3, {end} (letter = column, number = row). "
        "Use these labels to say where each issue is. Never return pixel or decimal coordinates.\n"
        "Check: bedding (wrinkled, stained, untucked sheets, misplaced pillows), cleanliness "
        "(dust, stains, spills, dirty floors, smudged glass), bathroom, clutter and trash, damage, "
        "and untidy setup.\n"
        "category must be exactly one of these words: bedding, cleanliness, bathroom, clutter, "
        "damage, setup.\n"
        "Rules:\n"
        "- For each issue, list every cell it visibly overlaps, no others, forming one connected block.\n"
        "- Use as few cells as possible (small stain: 1-4 cells; whole bed: 15+).\n"
        "- Report only what is visible. Do not invent issues. Ignore the grid lines.\n"
        "- If the image is too dark, blurry or cropped, say so in image_quality_note and lower confidence.\n"
        "- Each comment: what is wrong and what is needed, one short sentence.\n"
        "- Severity: critical (not guest-ready), moderate (fix before arrival), minor (small touch-up).\n"
        "- guest_ready is false if any critical issue or 2+ moderate issues.\n"
        "Return ONLY valid JSON:\n"
        '{"guest_ready": true, "overall_score": 0, "confidence": 0, "summary": "one sentence", '
        '"image_quality_note": null, "issues": [{"category": "bedding", "severity": "critical", '
        '"cells": ["C4", "D4"], "comment": "Bedsheet is wrinkled and untucked; remake the bed."}]}'
    )


def _comparison_instructions() -> str:
    end = last_cell()
    return (
        "You compare photos of one room before a guest stay with one photo taken after.\n"
        "The BEFORE photo has no grid. The AFTER photo has a red grid. Columns run A through "
        f"{end[0]} and rows run 1 through {end[1:]}. A cell is labeled like A1 or {end}. "
        "Say where each change is with those labels on the AFTER photo. Never return pixel "
        "or decimal coordinates.\n"
        "A closer, wider, or slightly shifted camera is still the same view. An object is "
        "missing when the same surface is visible in both photos and the object is gone. "
        "Do not copy a cell from where the object sat in the BEFORE photo; find that surface "
        "on the AFTER grid.\n"
        "If the AFTER photo shows a part of the room that was never photographed, set "
        "could_compare to false, return no marks, and explain that in summary.\n"
        "List the objects in the BEFORE photo that still fall inside the AFTER photo, then "
        "mark each present, missing, or moved. Also note damage and anything new.\n"
        "For each mark, cells is every grid cell the issue overlaps, one connected block, "
        "as few cells as possible. kind is damage, missing, new, or moved.\n"
        "A moved item has cells where it is now, and origin_cells where it used to be on this "
        "same AFTER grid. Do not also mark it missing or new.\n"
        "description is a short label, such as 'remote missing from the nightstand'.\n"
        "Do not say the room is clean or ready. If nothing changed, return no marks.\n"
        "Return ONLY valid JSON:\n"
        '{"could_compare": true, "summary": "one sentence", "views_compared": "the bed wall", '
        '"marks": [{"kind": "missing", "description": "remote missing from the nightstand", '
        '"photo_number": 1, "cells": ["C4"], "origin_cells": []}]}'
    )


def _missing_followup_instructions() -> str:
    end = last_cell()
    return (
        "Look again only for objects in the BEFORE photo that are gone from the AFTER photo.\n"
        f"The AFTER photo's red grid runs from A1 to {end}. Return cells on that grid, never "
        "pixel coordinates. List every cell the empty spot overlaps and no others.\n"
        "Do not report damage, new objects, or moved objects.\n"
        "If nothing is gone, return no marks.\n"
        "Return ONLY valid JSON:\n"
        '{"marks": [{"kind": "missing", "description": "towel missing from the rack", '
        '"photo_number": 1, "cells": ["E6"]}]}'
    )


def _prepare(image: Image.Image, quality: int, *, grid: bool) -> tuple[str, Frame]:
    """Resize first, then draw the grid on that exact bitmap, so the labels match the model input."""
    rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    scale = min(1.0, _MAX_SIDE / max(original_width, original_height))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, int(original_width * scale)), max(1, int(original_height * scale))),
            Image.Resampling.LANCZOS,
        )
    sent_width, sent_height = rgb.size
    frame = Frame(original_width, original_height, sent_width, sent_height)
    if grid:
        rgb = draw_grid(rgb)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", frame


def _paint_marks(marks: list[ChangeMark]) -> list[ChangeMark]:
    return [
        mark.model_copy(update={"color": issue_color(index)})
        for index, mark in enumerate(marks)
    ]


def _marks_from_grid(items: list[_GridMark], frames: list[Frame]) -> list[ChangeMark]:
    marks: list[ChangeMark] = []
    for item in items:
        description = " ".join(item.description.split())[:300]
        if not description:
            continue
        photo_number = item.photo_number if 1 <= item.photo_number <= len(frames) else 1
        frame = frames[photo_number - 1]
        for box in boxes_for_cells(item.cells, frame):
            origin_x = origin_y = None
            if item.kind == "moved":
                origin = box_for_cells(item.origin_cells, frame)
                if origin is not None:
                    origin_x = (origin.x1 + origin.x2) / 2
                    origin_y = (origin.y1 + origin.y2) / 2
            marks.append(
                ChangeMark(
                    kind=item.kind,
                    description=description,
                    side="after",
                    photo_number=photo_number,
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    cells=list(box.cells),
                )
            )
    return _paint_marks(marks)


def _issues_from_grid(items: list[_ReadyIssue], frame: Frame, photo_number: int) -> list[GuestIssue]:
    issues: list[GuestIssue] = []
    for item in items:
        comment = " ".join(item.comment.split())[:300]
        if not comment:
            continue
        for box in boxes_for_cells(item.cells, frame):
            issues.append(
                GuestIssue(
                    category=item.category,
                    severity=item.severity,
                    photo_number=photo_number,
                    comment=comment,
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    cells=list(box.cells),
                )
            )
    return [
        issue.model_copy(update={"color": issue_color(index)})
        for index, issue in enumerate(issues)
    ]


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
        return report.model_copy(update={"marks": _paint_marks(report.marks)})

    def check_ready(self, photos: list[ReviewImage]) -> RoomReview:
        if not photos:
            return RoomReview(
                readiness=True,
                could_compare=False,
                retake=True,
                summary="A photo of the room is required.",
            )
        issues: list[GuestIssue] = []
        summaries: list[str] = []
        notes: list[str] = []
        scores: list[int] = []
        confidences: list[float] = []
        for index, photo in enumerate(photos, start=1):
            encoded, frame = _prepare(photo.image, 90, grid=True)
            parsed = self._complete_json(
                instructions=_readiness_instructions(),
                content=[
                    {"type": "input_text", "text": "Judge this photo on its own. There is no before photo."},
                    {"type": "input_text", "text": photo.caption},
                    {"type": "input_image", "image_url": encoded, "detail": "high"},
                ],
                model=_ReadyOutput,
            )
            note = " ".join((parsed.image_quality_note or "").split())
            confidences.append(parsed.confidence)
            if note:
                notes.append(note)
            if note and not parsed.issues:
                continue
            scores.append(min(100, max(0, parsed.overall_score)))
            summary = " ".join(parsed.summary.split())
            if summary:
                summaries.append(summary)
            issues.extend(_issues_from_grid(parsed.issues, frame, index))
        issues = [
            issue.model_copy(update={"color": issue_color(index)})
            for index, issue in enumerate(issues[:12])
        ]
        retake = any(confidence < 0.6 for confidence in confidences) or bool(notes)
        if not scores and not issues:
            return RoomReview(
                readiness=True,
                could_compare=False,
                retake=True,
                summary=notes[0] if notes else "The photo is not clear enough to judge.",
                image_quality_note=notes[0] if notes else "",
                confidence=min(confidences) if confidences else 0,
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
            confidence=min(confidences) if confidences else None,
            retake=retake,
            issues=issues,
        )

    def _matched_photo(self, before: list[ReviewImage], after: list[ReviewImage]) -> int:
        if len(before) == 1:
            return 1
        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": (
                    "These BEFORE photos are different angles. "
                    "Reply with JSON {\"matched_photo\": N} for the one BEFORE photo that "
                    "matches the AFTER photo. 1 means the first BEFORE photo."
                ),
            }
        ]
        for photo in before:
            encoded, _frame = _prepare(photo.image, 80, grid=False)
            content.append({"type": "input_text", "text": photo.caption})
            content.append({"type": "input_image", "image_url": encoded, "detail": "auto"})
        content.append({"type": "input_text", "text": "AFTER photo."})
        for photo in after:
            encoded, _frame = _prepare(photo.image, 80, grid=False)
            content.append({"type": "input_text", "text": photo.caption})
            content.append({"type": "input_image", "image_url": encoded, "detail": "high"})
        picked = self._complete_json(
            instructions="Return only JSON with matched_photo, an integer from 1 to the number of BEFORE photos.",
            content=content,
            model=_MatchOutput,
        )
        if 1 <= picked.matched_photo <= len(before):
            return picked.matched_photo
        return 1

    def _compare_pair(
        self, focused: ReviewImage, after: list[ReviewImage], matched_photo: int
    ) -> RoomReview:
        content, frames = _pair_content(focused, after)
        parsed = self._complete_json(
            instructions=_comparison_instructions(),
            content=content,
            model=_ComparisonOutput,
        )
        return RoomReview(
            mode=ReviewMode.BOXES,
            marks=_marks_from_grid(parsed.marks, frames),
            matched_photo=matched_photo,
            summary=" ".join(parsed.summary.split())[:1000],
            views_compared=" ".join(parsed.views_compared.split())[:500],
            could_compare=parsed.could_compare,
        )

    def _add_missing_followup(
        self, report: RoomReview, focused: ReviewImage, after: list[ReviewImage]
    ) -> RoomReview:
        try:
            content, frames = _pair_content(
                focused,
                after,
                intro=(
                    "The first look found no missing items. "
                    "Which objects in the BEFORE photo are gone from the AFTER photo?"
                ),
            )
            parsed = self._complete_json(
                instructions=_missing_followup_instructions(),
                content=content,
                model=_MissingCells,
            )
        except VisionProviderError:
            logger.warning("Missing-item follow-up failed; keeping the pair comparison")
            return report
        extra = [
            mark
            for mark in _marks_from_grid(parsed.marks, frames)
            if mark.kind == "missing"
        ]
        return _merge_missing(report, extra)

    def _complete_json(
        self,
        *,
        instructions: str,
        content: list[dict[str, object]],
        model: type[_ReadyOutput] | type[_ComparisonOutput] | type[_MissingCells] | type[_MatchOutput],
    ):
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                response = self._client.responses.create(
                    model=self._model,
                    instructions=instructions,
                    input=[{"role": "user", "content": content}],
                    temperature=_TEMPERATURE,
                )
            except (APIError, APITimeoutError) as exc:
                logger.exception("OpenAI vision call failed")
                raise VisionProviderError("The vision model could not analyse the images.") from exc
            raw = getattr(response, "output_text", "") or ""
            try:
                payload = json.loads(raw)
                if model is _ReadyOutput:
                    payload = _normalize_ready(payload)
                return model.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning("Vision model returned JSON that could not be used")
        raise VisionProviderError("The vision model returned invalid JSON.") from last_error


def _normalize_ready(payload: object) -> object:
    """Map loose category words so one label does not reject the whole result."""
    if not isinstance(payload, dict):
        return payload
    issues: list[dict[str, object]] = []
    for issue in payload.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        category = _ready_category(issue.get("category"))
        if category is None:
            continue
        severity = str(issue.get("severity", "moderate")).strip().lower()
        if severity not in {"critical", "moderate", "minor"}:
            severity = "moderate"
        cells = issue.get("cells") or []
        issues.append({**issue, "category": category, "severity": severity, "cells": cells})
    score = payload.get("overall_score", 0)
    confidence = payload.get("confidence", 0)
    try:
        score = int(float(score))
    except (TypeError, ValueError):
        score = 0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {**payload, "issues": issues, "overall_score": score, "confidence": confidence}


def _ready_category(value: object) -> str | None:
    raw = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "bedding": "bedding",
        "cleanliness": "cleanliness",
        "bathroom": "bathroom",
        "clutter": "clutter",
        "clutter/trash": "clutter",
        "trash": "clutter",
        "damage": "damage",
        "setup": "setup",
        "untidy setup": "setup",
        "untidy": "setup",
    }
    if raw in aliases:
        return aliases[raw]
    if "clutter" in raw or "trash" in raw:
        return "clutter"
    if "setup" in raw or "untidy" in raw:
        return "setup"
    if "bath" in raw:
        return "bathroom"
    if "bed" in raw:
        return "bedding"
    if "clean" in raw:
        return "cleanliness"
    if "damage" in raw or "broken" in raw:
        return "damage"
    return None


def _pair_content(
    before: ReviewImage,
    after: list[ReviewImage],
    intro: str = (
        "The BEFORE photo has no grid. The AFTER photo has the red grid. "
        "Name cells on the AFTER photo only."
    ),
) -> tuple[list[dict[str, object]], list[Frame]]:
    before_url, _before_frame = _prepare(before.image, 90, grid=False)
    content: list[dict[str, object]] = [
        {"type": "input_text", "text": intro},
        {"type": "input_text", "text": before.caption},
        {"type": "input_image", "image_url": before_url, "detail": "high"},
        {"type": "input_text", "text": "AFTER photo, with the red grid. Put every cell on this photo."},
    ]
    frames: list[Frame] = []
    for photo in after:
        encoded, frame = _prepare(photo.image, 90, grid=True)
        frames.append(frame)
        content.append({"type": "input_text", "text": photo.caption})
        content.append({"type": "input_image", "image_url": encoded, "detail": "high"})
    return content, frames


def _merge_missing(report: RoomReview, extra: list[ChangeMark]) -> RoomReview:
    existing = {" ".join(mark.description.split()).casefold() for mark in report.marks}
    marks = list(report.marks)
    for mark in extra:
        label = " ".join(mark.description.split())
        if not label or label.casefold() in existing:
            continue
        marks.append(mark)
        existing.add(label.casefold())
    return report.model_copy(update={"marks": marks})
