"""RTSP camera stream that reconnects with backoff instead of ending."""

import os
import threading
import time
from collections.abc import Callable, Iterator

import cv2

from sopvision.sources.base import Frame

# FFmpeg options for the lowest delay OpenCV's RTSP reader can give: TCP (no lost packets), and no input buffering.
LOW_LATENCY = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_options_lock = threading.Lock()


class RtspSource:
    def __init__(self, url: str, open_capture=cv2.VideoCapture, sleep=time.sleep,
                 on_lost: Callable[[], None] | None = None, backoff=(1, 2, 4, 8, 16), clock=time.monotonic,
                 capture_options: str | None = None):
        self.url, self.capture_options = url, capture_options
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
            cap = self._open()
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

    def _open(self):
        """Open the capture with this source's FFmpeg options. OpenCV only reads them from the environment, so they
        are set for the duration of the open and then put back, leaving other sources (e.g. video files) alone."""
        if not self.capture_options:
            return self._open_capture(self.url)
        with _options_lock:
            before = os.environ.get(_OPTIONS_ENV)
            os.environ[_OPTIONS_ENV] = self.capture_options
            try:
                return self._open_capture(self.url)
            finally:
                if before is None:
                    os.environ.pop(_OPTIONS_ENV, None)
                else:
                    os.environ[_OPTIONS_ENV] = before

    def close(self) -> None:
        """Stop the stream. The reading thread releases the capture on its way out: OpenCV captures aren't
        safe to release while another thread is inside read()."""
        self._closed = True
