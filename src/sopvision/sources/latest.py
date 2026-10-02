"""Reads a source on a background thread and keeps only the newest frame.

Live mode always works on the newest frame, so a slow detector drops frames instead of
building up lag (spec §4.2).
"""

import threading
import time

from sopvision.sources.base import Frame, VideoSource


class LatestFrameReader:
    def __init__(self, source: VideoSource):
        self._source = source
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._consumed = -1
        self._closed = False
        self.dropped = 0
        self.exhausted = False
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="frame-reader", daemon=True)
        self._thread.start()

    @property
    def fps(self) -> float:
        return self._source.fps

    def _run(self) -> None:
        try:
            for frame in self._source.frames():
                with self._cond:
                    if self._latest is not None and self._latest.index > self._consumed:
                        self.dropped += 1
                    self._latest = frame
                    self._cond.notify_all()
                if self._closed:
                    break
        except BaseException as e:  # surfaced to the consumer via .error
            self.error = e
        finally:
            with self._cond:
                self.exhausted = True
                self._cond.notify_all()

    def get(self, timeout: float) -> Frame | None:
        """Newest frame not yet returned, or None on timeout / end of stream / close."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._closed:
                if self._latest is not None and self._latest.index > self._consumed:
                    self._consumed = self._latest.index
                    return self._latest
                if self.exhausted:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._source.close()
        self._thread.join(timeout=2.0)
