"""Annotations drawn on the live video: chassis, slot states, hands, and a performance footer."""

from dataclasses import dataclass

import cv2
import numpy as np

from sopvision.config import SopConfig
from sopvision.perception.detector import Detection
from sopvision.scene.types import SceneState

# BGR
INSTALLED = (0, 200, 0)
EMPTY = (160, 160, 160)
UNSURE = (0, 191, 255)  # occluded / unknown
HAND = (255, 140, 0)
WHITE = (255, 255, 255)
STATE_COLOURS = {"installed": INSTALLED, "empty": EMPTY, "occluded": UNSURE, "unknown": UNSURE}


@dataclass(frozen=True)
class OverlayStats:
    fps: float
    det_ms: float
    jev_ms: float | None


def _dashed_rect(img, box, colour, thickness, dash):
    x1, y1, x2, y2 = (int(v) for v in box)
    for (ax, ay), (bx, by) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = int(np.hypot(bx - ax, by - ay))
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            p = (int(ax + (bx - ax) * s / length), int(ay + (by - ay) * s / length))
            q = (int(ax + (bx - ax) * e / length), int(ay + (by - ay) * e / length))
            cv2.line(img, p, q, colour, thickness)


def draw_overlay(image: np.ndarray, scene: SceneState, detections: list[Detection], sop: SopConfig,
                 stats: OverlayStats) -> np.ndarray:
    out = image.copy()
    h = out.shape[0]
    unit = max(1, h // 360)  # line width / text scale grow with resolution
    font = 0.4 * unit
    if scene.chassis_box is not None and scene.chassis_visible:
        x1, y1, x2, y2 = scene.chassis_box
        _dashed_rect(out, scene.chassis_box, WHITE, unit, 8 * unit)
        for name, spec in sop.layout.slots.items():
            view = scene.slots.get(name)
            if view is None:
                continue
            cx, cy = int(x1 + spec.at[0] * (x2 - x1)), int(y1 + spec.at[1] * (y2 - y1))
            cv2.circle(out, (cx, cy), 7 * unit, STATE_COLOURS[view.state], -1)
            cv2.circle(out, (cx, cy), 7 * unit, (0, 0, 0), 1)
            cv2.putText(out, name, (cx + 9 * unit, cy - 4 * unit), cv2.FONT_HERSHEY_SIMPLEX, font,
                        STATE_COLOURS[view.state], unit)
    else:
        cv2.putText(out, "Chassis not visible", (10 * unit, 24 * unit), cv2.FONT_HERSHEY_SIMPLEX, font * 1.5, UNSURE,
                    unit + 1)
    for d in detections:
        if d.cls == "hand":
            x1, y1, x2, y2 = (int(v) for v in d.xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), HAND, unit + 1)
    if scene.held:
        cv2.putText(out, "held: " + ", ".join(scene.held), (10 * unit, h - 26 * unit), cv2.FONT_HERSHEY_SIMPLEX, font,
                    HAND, unit)
    jev = "-" if stats.jev_ms is None else f"{stats.jev_ms:.0f} ms"
    footer = f"{stats.fps:.0f} fps  YOLO {stats.det_ms:.0f} ms  Jev {jev}"
    cv2.putText(out, footer, (10 * unit, h - 8 * unit), cv2.FONT_HERSHEY_SIMPLEX, font, WHITE, unit)
    return out
