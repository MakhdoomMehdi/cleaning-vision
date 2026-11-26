"""The model names grid cells. The provider turns those names into boxes."""

import json
from types import SimpleNamespace

from PIL import Image

from app.providers.openai_provider import (
    OpenAIVisionProvider,
    ReviewImage,
    _comparison_instructions,
    _readiness_instructions,
)


class ScriptedResponses:
    def __init__(self, steps: list[object]) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, str):
            return SimpleNamespace(output_text=step)
        return SimpleNamespace(output_text=json.dumps(step))


def _photo(caption: str, size: tuple[int, int] = (100, 100)) -> ReviewImage:
    return ReviewImage(caption=caption, image=Image.new("RGB", size, (10, 20, 30)))


def _provider(steps: list[object]) -> tuple[OpenAIVisionProvider, ScriptedResponses]:
    provider = OpenAIVisionProvider("test-key", "fake-model", 1, 0)
    responses = ScriptedResponses(steps)
    provider._client = SimpleNamespace(responses=responses)  # type: ignore[assignment]
    return provider, responses


def _image_details(call: dict[str, object]) -> list[str]:
    content = call["input"][0]["content"]  # type: ignore[index]
    return [item["detail"] for item in content if item["type"] == "input_image"]


def test_prompts_name_the_grid_and_refuse_coordinates() -> None:
    ready = _readiness_instructions()
    compared = _comparison_instructions()
    assert "A1" in ready and "J10" in ready
    assert "Never return pixel or decimal coordinates" in ready
    assert "Never return pixel" in compared
    assert "J10" in compared


def test_readiness_turns_cells_into_a_box_and_retries_invalid_json() -> None:
    provider, responses = _provider(
        [
            "not json",
            {
                "guest_ready": False,
                "overall_score": 35,
                "confidence": 0.9,
                "summary": "Bed is not made.",
                "image_quality_note": None,
                "issues": [
                    {
                        "category": "bedding",
                        "severity": "critical",
                        "cells": ["C4", "D4", "nope"],
                        "comment": "Bedsheet is wrinkled and untucked; remake the bed.",
                    }
                ],
            },
        ]
    )
    report = provider.check_ready([_photo("Photo 1")])

    assert len(responses.calls) == 2
    assert responses.calls[0]["temperature"] == 0.2
    assert _image_details(responses.calls[1]) == ["high"]
    assert report.guest_ready is False and report.retake is False
    assert report.issues[0].cells == ["C4", "D4"]
    assert report.issues[0].x1 == 0.2 and report.issues[0].x2 == 0.4
    assert report.issues[0].color == "#2563eb"


def test_loose_category_words_do_not_fail_the_check() -> None:
    provider, responses = _provider(
        [
            {
                "guest_ready": False,
                "overall_score": 40,
                "confidence": 0.8,
                "summary": "The counter is messy.",
                "image_quality_note": None,
                "issues": [
                    {
                        "category": "clutter/trash",
                        "severity": "moderate",
                        "cells": ["B2"],
                        "comment": "Trash is left on the counter; throw it away.",
                    },
                    {
                        "category": "untidy setup",
                        "severity": "minor",
                        "cells": ["C3"],
                        "comment": "The chair is pulled out; push it in.",
                    },
                ],
            }
        ]
    )
    report = provider.check_ready([_photo("Photo 1")])

    assert len(responses.calls) == 1
    assert [issue.category for issue in report.issues] == ["clutter", "setup"]


def test_low_confidence_asks_for_a_retake() -> None:
    provider, _responses = _provider(
        [
            {
                "guest_ready": True,
                "overall_score": 80,
                "confidence": 0.4,
                "summary": "It looks tidy, but the photo is dim.",
                "image_quality_note": "The photo is dim.",
                "issues": [],
            }
        ]
    )
    report = provider.check_ready([_photo("Photo 1")])

    assert report.could_compare is False
    assert report.retake is True
    assert report.issues == []


def test_comparison_uses_the_gridded_after_photo() -> None:
    provider, responses = _provider(
        [
            {"matched_photo": 2},
            {
                "could_compare": True,
                "summary": "The remote is gone.",
                "views_compared": "the nightstand",
                "marks": [
                    {
                        "kind": "missing",
                        "description": "remote missing from the nightstand",
                        "photo_number": 1,
                        "cells": ["C4"],
                        "origin_cells": [],
                    }
                ],
            },
        ]
    )
    report = provider.review(
        [_photo("Before photo 1"), _photo("Before photo 2")],
        [_photo("After photo 1")],
    )

    assert len(responses.calls) == 2
    assert _image_details(responses.calls[0]) == ["auto", "auto", "high"]
    assert _image_details(responses.calls[1]) == ["high", "high"]
    assert report.matched_photo == 2
    assert report.marks[0].kind == "missing"
    assert report.marks[0].cells == ["C4"]
    assert report.marks[0].x1 == 0.2 and report.marks[0].y1 == 0.3
    assert report.marks[0].color == "#2563eb"
