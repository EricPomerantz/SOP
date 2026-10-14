"""Record a live camera into a dataset run: video.mp4 on a real-time clock, plus torque presses on its timeline.

A camera delivers frames irregularly (an Amcrest set to 30 fps measured ~23), and a replayed file's time is
frame index / file fps. Writing every received frame would make the replay run faster than real time. So frames are
written on a fixed clock (`fps`): each tick writes the newest frame, repeating it if none arrived and skipping extras.
Video time is then real time since the first frame was written.

A torque press happens in reality; the camera shows that moment `delay` seconds after it, so on the video's timeline
it is at (press - start) + delay.
"""

import csv
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class TorquePress:
    time_s: float  # on the video's timeline
    ok: bool
    torque_nm: float | None = None


class Recorder:
    def __init__(self, path: Path, fps: float = 25.0, delay_s: float = 0.0, clock: Callable[[], float] = time.monotonic,
                 open_writer=None):
        self.path, self.fps, self.delay_s, self._clock = Path(path), fps, delay_s, clock
        self._open_writer = open_writer or (lambda p, size: cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"),
                                                                            fps, size))
        self._lock = threading.Lock()
        self._latest = None
        self._writer = None
        self.start: float | None = None  # clock() when the first frame was written (video time 0)
        self.written = 0
        self.received = 0
        self.presses: list[TorquePress] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def offer(self, image) -> None:
        """The newest camera frame (called for every frame the camera delivers)."""
        with self._lock:
            self._latest = image
            self.received += 1

    def run(self) -> None:
        """The writing clock; call on its own thread (start_thread)."""
        period = 1.0 / self.fps
        while not self._stop.is_set():
            with self._lock:
                image = self._latest
            if image is None:
                time.sleep(0.005)
                continue
            if self._writer is None:
                h, w = image.shape[:2]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = self._open_writer(self.path, (w, h))
                self.start = self._clock()
            self._writer.write(image)
            self.written += 1
            next_tick = self.start + self.written * period
            time.sleep(max(0.0, next_tick - self._clock()))

    def start_thread(self) -> None:
        self._thread = threading.Thread(target=self.run, name="recorder", daemon=True)
        self._thread.start()

    def video_time(self, at: float | None = None) -> float | None:
        """Where a moment of reality (clock() now, or `at`) falls on the video's timeline."""
        if self.start is None:
            return None
        return (self._clock() if at is None else at) - self.start + self.delay_s

    def press(self, ok: bool, torque_nm: float | None = None) -> TorquePress | None:
        t = self.video_time()
        if t is None:
            return None  # nothing recorded yet
        p = TorquePress(round(t, 3), ok, torque_nm)
        self.presses.append(p)
        return p

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._writer is not None:
            self._writer.release()
        return {"frames": self.written, "seconds": round(self.written / self.fps, 2), "received": self.received,
                "presses": len(self.presses)}

    def write_torque_csv(self, path: Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "ok", "torque_nm"])
            for p in self.presses:
                w.writerow([p.time_s, int(p.ok), "" if p.torque_nm is None else p.torque_nm])
