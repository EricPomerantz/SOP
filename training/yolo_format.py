"""YOLO label lines (`<class> cx cy w h`, normalised) ↔ pixel boxes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabeledBox:
    cls: str
    xyxy: tuple[float, float, float, float]  # pixels
    conf: float


def to_yolo(box: LabeledBox, w: int, h: int, classes: list[str]) -> str:
    if box.cls not in classes:
        raise ValueError(f"unknown class '{box.cls}'")
    x1, y1, x2, y2 = box.xyxy
    cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
    return f"{classes.index(box.cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def from_yolo(line: str, w: int, h: int, classes: list[str], conf: float = 1.0) -> LabeledBox:
    idx, cx, cy, bw, bh = line.split()
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return LabeledBox(classes[int(idx)], (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), conf)
