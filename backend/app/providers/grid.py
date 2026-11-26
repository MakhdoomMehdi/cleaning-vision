"""Turn a photo into a labeled grid, then turn the model's cell names back into a box."""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

COLS = 10
ROWS = 10

# Same order as the findings list in the web app, so a box and its comment share a color.
ISSUE_COLORS = (
    "#2563eb",
    "#ea580c",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#0891b2",
    "#ca8a04",
    "#db2777",
    "#4f46e5",
    "#0d9488",
    "#c026d3",
    "#65a30d",
)


@dataclass(frozen=True)
class Frame:
    """How the photo was resized and padded before the grid was drawn on it."""

    original_width: int
    original_height: int
    sent_width: int
    sent_height: int
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0


@dataclass(frozen=True)
class CellBox:
    cells: tuple[str, ...]
    x1: float
    y1: float
    x2: float
    y2: float


def cell_name(column: int, row: int) -> str:
    """Column 0, row 0 is A1. Column 9, row 9 is J10."""
    return f"{chr(ord('A') + column)}{row + 1}"


def last_cell() -> str:
    return cell_name(COLS - 1, ROWS - 1)


def _cell_pattern() -> re.Pattern[str]:
    last_column = chr(ord("A") + COLS - 1)
    if ROWS < 10:
        rows = f"[1-{ROWS}]"
    elif ROWS == 10:
        rows = r"(?:[1-9]|10)"
    else:
        rows = rf"(?:[1-9]|[1-9][0-9])"
    return re.compile(rf"^([A-{last_column}])({rows})$", re.IGNORECASE)


_CELL_RE = _cell_pattern()


def parse_cell(label: str) -> tuple[int, int] | None:
    match = _CELL_RE.fullmatch(label.strip())
    if match is None:
        return None
    column = ord(match.group(1).upper()) - ord("A")
    row = int(match.group(2)) - 1
    if not (0 <= column < COLS and 0 <= row < ROWS):
        return None
    return column, row


def valid_cells(labels: list[str]) -> list[str]:
    """Drop anything that is not a cell on this grid. Keep first-seen order."""
    kept: list[str] = []
    seen: set[str] = set()
    for label in labels:
        parsed = parse_cell(str(label))
        if parsed is None:
            continue
        name = cell_name(*parsed)
        if name not in seen:
            seen.add(name)
            kept.append(name)
    return kept


def connected_groups(labels: list[str]) -> list[list[str]]:
    """Split cells that do not touch by an edge into separate groups."""
    names = valid_cells(labels)
    pending = {parse_cell(name): name for name in names}
    groups: list[list[str]] = []
    while pending:
        start = next(iter(pending))
        stack = [start]
        group: list[str] = []
        while stack:
            column, row = stack.pop()
            name = pending.pop((column, row), None)
            if name is None:
                continue
            group.append(name)
            for neighbor in (
                (column - 1, row),
                (column + 1, row),
                (column, row - 1),
                (column, row + 1),
            ):
                if neighbor in pending:
                    stack.append(neighbor)
        groups.append(group)
    return groups


def _fraction(pixel: float, pad_start: int, content_size: int) -> float:
    if content_size <= 0:
        return 0.0
    return min(1.0, max(0.0, (pixel - pad_start) / content_size))


def box_for_cells(labels: list[str], frame: Frame) -> CellBox | None:
    """One box around the cells. Coordinates are fractions of the original photo."""
    names = valid_cells(labels)
    if not names:
        return None
    columns = [parse_cell(name)[0] for name in names]  # type: ignore[index]
    rows = [parse_cell(name)[1] for name in names]  # type: ignore[index]
    x0 = min(columns) * frame.sent_width / COLS
    x1 = (max(columns) + 1) * frame.sent_width / COLS
    y0 = min(rows) * frame.sent_height / ROWS
    y1 = (max(rows) + 1) * frame.sent_height / ROWS
    content_width = frame.sent_width - frame.pad_left - frame.pad_right
    content_height = frame.sent_height - frame.pad_top - frame.pad_bottom
    return CellBox(
        cells=tuple(names),
        x1=_fraction(x0, frame.pad_left, content_width),
        y1=_fraction(y0, frame.pad_top, content_height),
        x2=_fraction(x1, frame.pad_left, content_width),
        y2=_fraction(y1, frame.pad_top, content_height),
    )


def boxes_for_cells(labels: list[str], frame: Frame) -> list[CellBox]:
    boxes: list[CellBox] = []
    for group in connected_groups(labels):
        box = box_for_cells(group, frame)
        if box is not None and box.x2 > box.x1 and box.y2 > box.y1:
            boxes.append(box)
    return boxes


def issue_color(index: int) -> str:
    return ISSUE_COLORS[index % len(ISSUE_COLORS)]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_grid(image: Image.Image) -> Image.Image:
    """Red lines and yellow cell names on a copy. The copy is the same size as `image`."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    red = (220, 38, 38)
    yellow = (250, 204, 21)
    for column in range(COLS + 1):
        x = round(column * width / COLS)
        draw.line([(x, 0), (x, height - 1)], fill=red, width=2)
    for row in range(ROWS + 1):
        y = round(row * height / ROWS)
        draw.line([(0, y), (width - 1, y)], fill=red, width=2)
    label_size = max(8, min(width // COLS, height // ROWS) // 3)
    font = _font(label_size)
    for column in range(COLS):
        for row in range(ROWS):
            draw.text(
                (column * width / COLS + 3, row * height / ROWS + 2),
                cell_name(column, row),
                fill=yellow,
                font=font,
            )
    return canvas
