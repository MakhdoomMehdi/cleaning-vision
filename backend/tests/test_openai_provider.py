"""The vision provider picks a match, compares that pair up close, then looks again for missing items."""

from types import SimpleNamespace

from PIL import Image

from app.providers.openai_provider import (
    OpenAIVisionProvider,
    ReviewImage,
    _box_instructions,
    _missing_followup_instructions,
)
from app.schemas.review import ChangeMark


class ScriptedResponses:
    def __init__(self, steps: list[object]) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return SimpleNamespace(output_parsed=step)


def _photo(caption: str) -> ReviewImage:
    return ReviewImage(caption=caption, image=Image.new("RGB", (32, 32), (10, 20, 30)))


def _mark(kind: str, description: str) -> ChangeMark:
    return ChangeMark(
        kind=kind,  # type: ignore[arg-type]
        description=description,
        side="after",
        photo_number=1,
        x1=0.2,
        y1=0.2,
        x2=0.4,
        y2=0.5,
    )


def _provider(steps: list[object]) -> tuple[OpenAIVisionProvider, ScriptedResponses]:
    provider = OpenAIVisionProvider("test-key", "fake-model", 1, 0)
    responses = ScriptedResponses(steps)
    provider._client = SimpleNamespace(responses=responses)  # type: ignore[assignment]
    return provider, responses


def _details(call: dict[str, object]) -> list[str]:
    content = call["input"][0]["content"]  # type: ignore[index]
    return [item["detail"] for item in content if item["type"] == "input_image"]


def test_box_instructions_list_objects_before_drawing_boxes() -> None:
    text = _box_instructions()
    assert "List the distinct objects" in text
    assert "including the empty surface" in text
    assert "slightly shifted camera is still the same view" in text
    followup = _missing_followup_instructions()
    assert "gone from the AFTER photo" in followup
    assert "Do not report damage" in followup


def test_matched_before_photo_is_compared_again_at_high_detail() -> None:
    from app.providers.openai_provider import BoxedOutput

    provider, responses = _provider(
        [
            BoxedOutput(matched_photo=2, summary="pick"),
            BoxedOutput(
                matched_photo=1,
                summary="scratch and a missing remote",
                marks=[_mark("damage", "scratch on the door"), _mark("missing", "remote")],
            ),
        ]
    )
    report = provider.review(
        [_photo("Before photo 1"), _photo("Before photo 2")],
        [_photo("After photo 1")],
    )

    assert len(responses.calls) == 2
    assert _details(responses.calls[0]) == ["auto", "auto", "high"]
    assert _details(responses.calls[1]) == ["high", "high"]
    assert report.matched_photo == 2
    assert [mark.description for mark in report.marks] == ["scratch on the door", "remote"]


def test_followup_adds_missing_items_and_keeps_the_other_findings() -> None:
    from app.providers.openai_provider import BoxedOutput, MissingOutput

    provider, responses = _provider(
        [
            BoxedOutput(matched_photo=2),
            BoxedOutput(marks=[_mark("damage", "scratch on the door"), _mark("new", "extra pillow")]),
            MissingOutput(marks=[_mark("missing", "remote on the nightstand"), _mark("damage", "ignore me")]),
        ]
    )
    report = provider.review(
        [_photo("Before photo 1"), _photo("Before photo 2")],
        [_photo("After photo 1")],
    )

    assert len(responses.calls) == 3
    assert _details(responses.calls[2]) == ["high", "high"]
    assert [mark.kind for mark in report.marks] == ["damage", "new", "missing"]
    assert report.marks[-1].description == "remote on the nightstand"
    assert report.matched_photo == 2


def test_followup_is_skipped_when_the_pair_could_not_compare() -> None:
    from app.providers.openai_provider import BoxedOutput

    provider, responses = _provider(
        [BoxedOutput(could_compare=False, summary="That wall was never photographed.")]
    )
    report = provider.review([_photo("Before photo 1")], [_photo("After photo 1")])

    assert len(responses.calls) == 1
    assert report.could_compare is False
    assert report.marks == []


def test_failed_followup_keeps_the_pair_result() -> None:
    from app.providers.openai_provider import BoxedOutput

    provider, responses = _provider(
        [
            BoxedOutput(marks=[_mark("moved", "lamp moved onto the dresser")]),
            None,
        ]
    )
    report = provider.review([_photo("Before photo 1")], [_photo("After photo 1")])

    assert len(responses.calls) == 2
    assert [mark.description for mark in report.marks] == ["lamp moved onto the dresser"]
