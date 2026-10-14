"""Annotations drawn on the live video: the anchor, each slot's state, hands, tools, and a performance footer.

Slots are outlined, never filled, so the part in them stays visible: a ring around the matched detection (or the slot's
position) in the part's colour when installed, grey dashed when empty, and dashed in the colour of what it last held
while hidden (amber if nothing yet). A torqued slot gets a second ring and a check. Each slot's tag sits outside the
anchor box, pointing away from its centre, so tags don't cover the parts or each other.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from sopvision.config import SopConfig
from sopvision.perception.detector import Detection
from sopvision.scene.types import SceneState, SlotView

# BGR
INSTALLED = (0, 200, 0)
EMPTY = (160, 160, 160)
UNSURE = (0, 191, 255)  # occluded / unknown
HAND = (255, 140, 0)
TOOL = (255, 0, 200)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)
STATE_COLOURS = {"installed": INSTALLED, "empty": EMPTY, "occluded": UNSURE, "unknown": UNSURE}
# A part class naming a colour (lug_bolt_gold) is drawn in that colour and tagged with it.
PART_COLOURS = {"blue": (255, 150, 30), "green": (60, 210, 60), "red": (50, 50, 235), "gold": (0, 200, 245),
                "yellow": (0, 230, 255), "black": (90, 90, 90), "white": (240, 240, 240), "orange": (0, 140, 255)}
FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


@dataclass(frozen=True)
class OverlayStats:
    fps: float
    det_ms: float
    jev_ms: float | None  # the deciding model's last request time
    model: str = "Jev"


def part_colour(part: str | None) -> tuple[str | None, tuple[int, int, int]]:
    """(colour word, BGR) for a part class: the colour its name ends with, else plain 'installed' green."""
    word = (part or "").rsplit("_", 1)[-1].lower()
    return (word, PART_COLOURS[word]) if word in PART_COLOURS else (None, INSTALLED)


def _dashed_rect(img, box, colour, thickness, dash):
    x1, y1, x2, y2 = (int(v) for v in box)
    for (ax, ay), (bx, by) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = int(np.hypot(bx - ax, by - ay))
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            p = (int(ax + (bx - ax) * s / length), int(ay + (by - ay) * s / length))
            q = (int(ax + (bx - ax) * e / length), int(ay + (by - ay) * e / length))
            cv2.line(img, p, q, colour, thickness, AA)


def _dashed_circle(img, centre, r, colour, thickness, segments=16):
    step = 360 / segments
    for i in range(segments):
        cv2.ellipse(img, centre, (r, r), 0, i * step, i * step + step * 0.55, colour, thickness, AA)


def _blend(img, x1, y1, x2, y2, colour, alpha):
    """A translucent filled rectangle, clipped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        roi[:] = (roi * (1 - alpha) + np.array(colour, np.float32) * alpha).astype(np.uint8)


def _tag(img, text, anchor, colour, scale, unit, align_right=False, check=False):
    """A dark rounded-looking pill with a colour swatch, the text, and optionally a check mark."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    pad, sw = 4 * unit, 6 * unit
    cw = (th + 2 * unit) if check else 0
    width = pad + sw + pad + tw + (pad + cw if check else 0) + pad
    height = th + base + 2 * pad
    x, y = anchor
    x1 = x - width if align_right else x
    y1 = y - height // 2
    _blend(img, x1, y1, x1 + width, y1 + height, DARK, 0.72)
    cv2.rectangle(img, (x1 + pad, y1 + pad), (x1 + pad + sw, y1 + height - pad), colour, -1)
    tx = x1 + pad + sw + pad
    cv2.putText(img, text, (tx, y1 + pad + th), FONT, scale, WHITE, 1, AA)
    if check:
        cx = tx + tw + pad
        cy1, cy2 = y1 + pad, y1 + height - pad
        pts = np.array([(cx, (cy1 + cy2) // 2), (cx + cw // 3, cy2 - unit), (cx + cw, cy1 + unit)], np.int32)
        cv2.polylines(img, [pts], False, (90, 230, 90), max(2, unit), AA)


def _slot_label(name: str) -> str:
    return name.replace("_", " ")


def _draw_slot(out, name, view: SlotView, at_px, centre_px, unit, scale):
    state, held = view.state, view.effective()  # held: what the slot holds, also while it's hidden
    if held == "installed":
        word, colour = part_colour(view.part)
    elif held == "empty":
        word, colour = "empty", EMPTY
    else:
        word, colour = None, UNSURE
    if view.box is not None:
        bx1, by1, bx2, by2 = view.box
        cx, cy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
        r = int(max(bx2 - bx1, by2 - by1) / 2) + 3 * unit
        square = max(bx2 - bx1, by2 - by1) <= 1.5 * min(bx2 - bx1, by2 - by1)
    else:
        (cx, cy), r, square = at_px, 12 * unit, True
    big = not square or r > 60 * unit  # a large or long part (a fan, a cover): outline its box instead of a ring
    if state == "installed" and big and view.box is not None:
        x1, y1, x2, y2 = (int(v) for v in view.box)
        cv2.rectangle(out, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), DARK, 2 * unit + 2, AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2 * unit, AA)
    elif state == "installed":
        cv2.circle(out, (cx, cy), r, DARK, 2 * unit + 2, AA)  # dark edge keeps the ring readable on any background
        cv2.circle(out, (cx, cy), r, colour, 2 * unit, AA)
    else:
        _dashed_circle(out, (cx, cy), r, DARK, unit + 2)
        _dashed_circle(out, (cx, cy), r, colour, unit)
    if view.torqued and not big:
        cv2.circle(out, (cx, cy), r + 3 * unit, (90, 230, 90), unit, AA)

    dx, dy = cx - centre_px[0], cy - centre_px[1]
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    gap = r + 6 * unit + (3 * unit if view.torqued else 0)
    ax, ay = int(cx + ux * gap), int(cy + uy * gap)
    hidden = state in ("occluded", "unknown")
    text = _slot_label(name) + (f" {word}" if word else "")
    if hidden:
        text += " (hidden)" if held else " hidden"
    _tag(out, text, (ax, ay), colour, scale, unit, align_right=ux < -0.3, check=view.torqued)


def draw_overlay(image: np.ndarray, scene: SceneState, detections: list[Detection], sop: SopConfig,
                 stats: OverlayStats) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    unit = max(1, h // 540)  # line width grows with resolution
    scale = 0.38 * max(1.0, h / 540)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        if d.cls == "hand":
            cv2.rectangle(out, (x1, y1), (x2, y2), HAND, unit, AA)
        elif d.cls in sop.layout.tools:
            cv2.rectangle(out, (x1, y1), (x2, y2), TOOL, 2 * unit, AA)
            _tag(out, d.cls.replace("_", " "), (x1, y2 + 12 * unit), TOOL, scale, unit)  # below: tags go above/beside
    if scene.chassis_box is not None and scene.chassis_visible:
        x1, y1, x2, y2 = scene.chassis_box
        _dashed_rect(out, scene.chassis_box, WHITE, 1, 6 * unit)
        centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        for name, spec in sop.layout.slots.items():
            view = scene.slots.get(name)
            if view is None:
                continue
            at = (int(x1 + spec.at[0] * (x2 - x1)), int(y1 + spec.at[1] * (y2 - y1)))
            _draw_slot(out, name, view, at, centre, unit, scale)
    else:
        _tag(out, "Anchor not visible" if sop.layout.anchor != "chassis" else "Chassis not visible",
             (10 * unit, 20 * unit), UNSURE, scale * 1.3, unit)
    footer = f"{stats.fps:.0f} fps   YOLO {stats.det_ms:.0f} ms   {stats.model} " + \
             ("-" if stats.jev_ms is None else f"{stats.jev_ms:.0f} ms")
    if scene.held:
        names = []
        for part in scene.held:
            word, _ = part_colour(part)
            names.append(f"{word} {part.rsplit('_', 1)[0].replace('_', ' ')}" if word else part.replace("_", " "))
        footer = "holding " + ", ".join(names) + "     " + footer
    (tw, th), base = cv2.getTextSize(footer, FONT, scale, 1)
    _blend(out, 0, h - th - base - 12 * unit, tw + 20 * unit, h, DARK, 0.6)
    cv2.putText(out, footer, (10 * unit, h - base - 6 * unit), FONT, scale, WHITE, 1, AA)
    return out
