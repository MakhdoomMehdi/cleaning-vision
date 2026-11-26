"""Cell names become boxes on the original photo, including after a resize or pad."""

from PIL import Image

from app.providers.grid import (
    COLS,
    Frame,
    box_for_cells,
    boxes_for_cells,
    connected_groups,
    draw_grid,
    last_cell,
    valid_cells,
)


def test_invalid_cells_are_dropped_and_an_empty_issue_is_discarded() -> None:
    frame = Frame(100, 100, 100, 100)
    assert valid_cells(["C4", "nope", "Z9", "A0", "A11", "c4"]) == ["C4"]
    assert boxes_for_cells(["nope", "Z9"], frame) == []
    assert last_cell() == "J10"
    assert COLS == 10


def test_touching_cells_become_one_box() -> None:
    frame = Frame(100, 100, 100, 100)
    box = box_for_cells(["C4", "D4"], frame)
    assert box is not None
    assert box.cells == ("C4", "D4")
    assert (box.x1, box.y1, box.x2, box.y2) == (0.2, 0.3, 0.4, 0.4)


def test_disconnected_cells_split_into_separate_boxes() -> None:
    frame = Frame(100, 100, 100, 100)
    boxes = boxes_for_cells(["A1", "J10", "C4", "E4"], frame)
    assert connected_groups(["A1", "J10"]) == [["A1"], ["J10"]]
    assert [box.cells for box in boxes] == [("A1",), ("J10",), ("C4",), ("E4",)]


def test_box_maps_back_to_the_original_photo_after_resize_and_pad() -> None:
    scaled = box_for_cells(["A1"], Frame(2000, 1000, 1600, 800))
    assert scaled is not None
    assert (scaled.x1, scaled.y1, scaled.x2, scaled.y2) == (0.0, 0.0, 0.1, 0.1)

    padded = box_for_cells(
        ["C4"],
        Frame(800, 600, 1000, 1000, pad_left=100, pad_top=200, pad_right=100, pad_bottom=200),
    )
    assert padded is not None
    assert padded.x1 == 0.125
    assert padded.x2 == 0.25
    assert abs(padded.y1 - (300 - 200) / 600) < 1e-9
    assert abs(padded.y2 - (400 - 200) / 600) < 1e-9
    assert 0.0 <= padded.x1 <= padded.x2 <= 1.0
    assert 0.0 <= padded.y1 <= padded.y2 <= 1.0


def test_grid_is_drawn_on_a_copy_the_same_size_as_the_resized_photo() -> None:
    photo = Image.new("RGB", (100, 80), (10, 20, 30))
    gridded = draw_grid(photo)
    assert gridded.size == photo.size
    assert photo.getpixel((0, 0)) == (10, 20, 30)
    assert gridded.getpixel((0, 40))[:1] == (220,)
