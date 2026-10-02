"""Chassis-relative geometry: a smoothed chassis anchor and detection-to-slot matching (spec §6.2)."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from sopvision.config import SopConfig
from sopvision.perception.detector import Detection

Box = tuple[float, float, float, float]
_NO_MATCH = 1e6


@dataclass(frozen=True)
class AnchorState:
    box: Box | None
    visible: bool
    lost_event: bool
    found_event: bool


class ChassisAnchor:
    """Exponential moving average of the chassis box; reports when it goes missing and comes back."""

    def __init__(self, alpha: float = 0.1, lost_after: float = 3.0):
        self.alpha, self.lost_after = alpha, lost_after
        self.box: Box | None = None
        self.last_seen: float | None = None
        self.lost = False

    def update(self, t: float, box: Box | None) -> AnchorState:
        if box is None:
            lost_event = (self.last_seen is not None and not self.lost and t - self.last_seen >= self.lost_after)
            self.lost = self.lost or lost_event
            return AnchorState(self.box, self.box is not None and not self.lost, lost_event, False)
        found = self.lost
        if self.box is None or found:  # first sighting, or back after a loss (it may have moved): start fresh
            self.box = tuple(float(v) for v in box)
        else:
            self.box = tuple((1 - self.alpha) * o + self.alpha * n for o, n in zip(self.box, box))
        self.lost, self.last_seen = False, t
        return AnchorState(self.box, True, False, found)


def to_norm(xy, chassis_box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = chassis_box
    return ((xy[0] - x1) / (x2 - x1), (xy[1] - y1) / (y2 - y1))


def box_distance_norm(uv, box_xyxy: Box, chassis_box: Box) -> float:
    """Distance from a normalised point to a pixel box, in chassis-normalised units (0 when inside)."""
    bx1, by1 = to_norm(box_xyxy[:2], chassis_box)
    bx2, by2 = to_norm(box_xyxy[2:], chassis_box)
    dx = max(bx1 - uv[0], 0.0, uv[0] - bx2)
    dy = max(by1 - uv[1], 0.0, uv[1] - by2)
    return math.hypot(dx, dy)


def center_norm(det: Detection, chassis_box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = det.xyxy
    return to_norm(((x1 + x2) / 2, (y1 + y2) / 2), chassis_box)


def match_slots(dets: list[Detection], sop: SopConfig, chassis_box: Box) -> dict[str, Detection]:
    """One-to-one assignment of class-compatible detections to the nearest slot centres within each slot's radius."""
    slots = list(sop.layout.slots.items())
    if not dets or not slots:
        return {}
    cost = np.full((len(dets), len(slots)), _NO_MATCH)
    for i, det in enumerate(dets):
        u, v = center_norm(det, chassis_box)
        for j, (_, spec) in enumerate(slots):
            if det.cls in (spec.filled_by, spec.empty_as):
                cost[i, j] = math.hypot(u - spec.at[0], v - spec.at[1])
    rows, cols = linear_sum_assignment(cost)
    return {slots[j][0]: dets[i] for i, j in zip(rows, cols) if cost[i, j] <= sop.slot_radius(slots[j][0])}
