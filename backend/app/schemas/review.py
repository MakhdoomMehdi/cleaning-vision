"""Room comparison: either short text lists, or boxes with a description on each change."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ReviewMode(StrEnum):
    SEMANTIC = "semantic"
    BOXES = "boxes"


class ChangeMark(BaseModel):
    """A change the model located on one of the photos it was shown."""

    kind: Literal["damage", "missing", "new"]
    description: str
    side: Literal["before", "after"]
    photo_number: int = Field(ge=1, le=12)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class RoomReview(BaseModel):
    mode: ReviewMode = ReviewMode.SEMANTIC
    damage: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    marks: list[ChangeMark] = Field(default_factory=list)
    summary: str = ""
    views_compared: str = ""
    could_compare: bool = True
