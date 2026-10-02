"""USB webcam (native runs only; Docker Desktop on Windows can't see USB cameras)."""

import time
from collections.abc import Iterator

import cv2

from sopvision.sources.base import Frame, SourceUnavailableError


class WebcamSource:
    def __init__(self, device: int, open_capture=cv2.VideoCapture, clock=time.monotonic):
        self.device = device
        self.fps = 0.0
        self._open_capture = open_capture
        self._clock = clock
        self._cap = None
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        cap = self._open_capture(self.device)
        self._cap = cap
        if not cap.isOpened():
            raise SourceUnavailableError(f"Can't open webcam {self.device}. Is it plugged in and not used by another app?")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        start = self._clock()
        index = 0
        try:
            while not self._closed:
                ok, image = cap.read()
                if not ok:
                    return
                yield Frame(self._clock() - start, index, image)
                index += 1
        finally:
            cap.release()

    def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            self._cap.release()
