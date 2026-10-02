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
