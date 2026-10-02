"""RTSP camera stream that reconnects with backoff instead of ending."""

import time
from collections.abc import Callable, Iterator

import cv2

from sopvision.sources.base import Frame


class RtspSource:
    def __init__(self, url: str, open_capture=cv2.VideoCapture, sleep=time.sleep,
                 on_lost: Callable[[], None] | None = None, backoff=(1, 2, 4, 8, 16), clock=time.monotonic):
        self.url = url
        self.fps = 0.0
        self.on_lost = on_lost
        self._open_capture = open_capture
        self._sleep = sleep
        self._backoff = backoff
        self._clock = clock
        self._cap = None
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        start: float | None = None
        index = 0
        attempt = 0
        outage_reported = False
        while not self._closed:
            cap = self._open_capture(self.url)
            self._cap = cap
            if cap.isOpened():
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                while not self._closed:
                    ok, image = cap.read()
                    if not ok:
                        break
                    attempt, outage_reported = 0, False
                    if start is None:
                        start = self._clock()
                    yield Frame(self._clock() - start, index, image)
                    index += 1
            cap.release()
            if self._closed:
                return
            if not outage_reported:
                outage_reported = True
                if self.on_lost:
                    self.on_lost()
            self._sleep(self._backoff[min(attempt, len(self._backoff) - 1)])
            attempt += 1

    def close(self) -> None:
        """Stop the stream. The reading thread releases the capture on its way out: OpenCV captures aren't
        safe to release while another thread is inside read()."""
        self._closed = True
