"""Recorded video played as if live (realtime) or frame by frame (lockstep, for eval)."""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import cv2

from sopvision.sources.base import Frame, SourceUnavailableError

PAUSE_POLL = 0.05


class FileSource:
    def __init__(self, path, mode: Literal["realtime", "lockstep"] = "realtime", speed: float = 1.0,
                 open_capture=cv2.VideoCapture, sleep=time.sleep, clock=time.monotonic):
        self.path = Path(path)
        self.mode = mode
        self.speed = speed
        self.fps = 0.0
        self._open_capture = open_capture
        self._sleep = sleep
        self._clock = clock
        self._cap = None
        self._closed = False
        self._paused = False
        self._anchor: tuple[float, float] | None = None  # (wall time, video time) pacing reference
        self._last_t = 0.0

    def frames(self) -> Iterator[Frame]:
        cap = self._open_capture(str(self.path))
        self._cap = cap
        if not cap.isOpened():
            raise SourceUnavailableError(f"Can't open video {self.path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        index = 0
        try:
            while not self._closed:
                ok, image = cap.read()
                if not ok:
                    return
                # Some files report 0 fps; fall back to the decoder's timestamp.
                t = index / self.fps if self.fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if self.mode == "realtime":
                    self._pace(t)
                self._last_t = t
                yield Frame(t, index, image)
                index += 1
        finally:
            cap.release()

    def _pace(self, t: float) -> None:
        while self._paused and not self._closed:
            self._sleep(PAUSE_POLL)
        if self._anchor is None:
            self._anchor = (self._clock(), t)
        wall0, t0 = self._anchor
        delay = wall0 + (t - t0) / self.speed - self._clock()
        if delay > 0:
            self._sleep(delay)

    def _reanchor(self) -> None:
        self._anchor = (self._clock(), self._last_t)

    def set_speed(self, speed: float) -> None:
        self.speed = speed
        self._reanchor()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._reanchor()

    def close(self) -> None:
        """Stop the stream. The reading thread releases the capture on its way out: OpenCV captures aren't
        safe to release while another thread is inside read()."""
        self._closed = True
