"""Shared test fakes and builders (import as `tests.helpers`)."""

import cv2
import numpy as np


class FakeClock:
    """A monotonic clock that only moves when `sleep` is called."""

    def __init__(self, start: float = 0.0):
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeCapture:
    """Stands in for cv2.VideoCapture.

    frames: good reads before end of stream; fail_after: good reads before a read failure
    (same effect, named for streams); pos_msec: per-read CAP_PROP_POS_MSEC values.
    """

    def __init__(self, fps: float = 10.0, frames: int | None = None, fail_after: int | None = None,
                 open_ok: bool = True, pos_msec: list[float] | None = None, size=(48, 64)):
        self.fps = fps
        self.limit = fail_after if fail_after is not None else frames if frames is not None else (len(pos_msec) if pos_msec else 10)
        self.open_ok = open_ok
        self.pos_msec = pos_msec
        self.size = size
        self.reads = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.open_ok

    def read(self):
        if not self.open_ok or self.reads >= self.limit:
            return False, None
        frame = np.full((*self.size, 3), self.reads % 256, np.uint8)
        self.reads += 1
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC and self.pos_msec:
            return self.pos_msec[self.reads - 1]
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.limit)
        return 0.0

    def release(self) -> None:
        self.released = True


# --- scene builders (Tasks 11–12) ---
import math

from sopvision.perception.detector import Detection

CHASSIS = (100.0, 100.0, 1100.0, 600.0)  # 1000 x 500 px


def det_at(cls: str, u: float, v: float, size: float = 40, conf: float = 0.9, chassis=CHASSIS, w: float | None = None,
           h: float | None = None) -> Detection:
    """A detection centred at chassis-normalised (u, v)."""
    x1, y1, x2, y2 = chassis
    cx, cy = x1 + u * (x2 - x1), y1 + v * (y2 - y1)
    w, h = w or size, h or size
    return Detection(cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), conf, None)


def rotate(uv, deg: float, chassis=CHASSIS):
    """Rotate a normalised point about the chassis centre by `deg`, in pixel space."""
    x1, y1, x2, y2 = chassis
    w, h = x2 - x1, y2 - y1
    px, py = (uv[0] - 0.5) * w, (uv[1] - 0.5) * h
    a = math.radians(deg)
    rx, ry = px * math.cos(a) - py * math.sin(a), px * math.sin(a) + py * math.cos(a)
    return (0.5 + rx / w, 0.5 + ry / h)
