"""Room comparison: either short text lists, or boxes with a description on each change."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ReviewMode(StrEnum):
    SEMANTIC = "semantic"
    BOXES = "boxes"


class ChangeMark(BaseModel):
    """A change the model located on one of the photos it was shown."""

    kind: Literal["damage", "missing", "new", "moved"]
    description: str
    side: Literal["before", "after"]
    photo_number: int = Field(ge=1, le=12)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    # Center of where a moved item used to be, on the same after photo. Not a second box.
    origin_x: float | None = Field(default=None, ge=0.0, le=1.0)
    origin_y: float | None = Field(default=None, ge=0.0, le=1.0)


class RoomReview(BaseModel):
    mode: ReviewMode = ReviewMode.BOXES
    damage: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    moved: list[str] = Field(default_factory=list)
    marks: list[ChangeMark] = Field(default_factory=list)
    # 1-based index of the single before photo that matches the after photo.
    matched_photo: int = Field(default=1, ge=1, le=12)
    summary: str = ""
    views_compared: str = ""
    could_compare: bool = True
