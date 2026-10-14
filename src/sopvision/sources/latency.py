"""How far a live camera's frames trail reality: measured with a QR clock, stored per source.

The dashboard's latency page (`/#/latency`) shows a QR code of the current time (on the app's clock), held for 50 ms at a
time so a camera frame catches one clean code rather than a blend of several. With the camera pointed at that screen, every frame that arrives carries the moment it was filmed;
arrival time minus that moment is the delay from scene to app: encoder, network and decoding together (the screen's
own refresh adds a few milliseconds). Stored in runs/camera_latency.json and used to time torque results.
"""

import json
import statistics
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from sopvision.settings import Settings
from sopvision.sources.base import Frame
from sopvision.sources.registry import SourceSpec

PREFIX = "SVL:"  # SVL:<hold ms>:<time ms modulo 1e8>; short, so the QR code has few, large modules
FILE = "camera_latency.json"
WRAP_MS = 100_000_000  # ~28 h: the time is sent modulo this, and recovered from the frame's arrival time


def parse_clock(text: str | None, arrived_ms: float) -> float | None:
    """The delay (ms) one decoded QR clock gives, or None if it isn't one.

    The page shows each code for `hold` ms; the camera catches it anywhere within that window, so on average half of
    it is added to the delay: that half is subtracted. The time is modulo WRAP_MS; the most recent matching time before
    the frame's arrival is the one meant.
    """
    if not text or not text.startswith(PREFIX):
        return None
    try:
        hold, shown = (float(x) for x in text[len(PREFIX):].split(":"))
    except ValueError:
        return None
    return (arrived_ms - shown) % WRAP_MS - hold / 2


def decode_qr(image) -> str | None:
    import cv2

    text, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    return text or None


def measure(frames: Iterator[Frame], seconds: float, decode: Callable = decode_qr,
            clock: Callable[[], float] = time.time) -> dict:
    """Read frames for `seconds` (wall time); every decodable QR clock gives one delay sample (ms).

    Decoding a QR code in a 1080p frame is slower than a camera's frame interval, so frames are read on their own thread
    at full speed (each stamped on arrival) and only the newest one is decoded. Decoding every frame in turn would let
    unread frames queue up in the stream, and the measured delay would be that queue, growing for as long as it runs.
    """
    lock = threading.Lock()
    latest: list = [None]  # (arrival time, frame) of the newest frame not yet decoded
    count = [0]
    stop = threading.Event()

    def read() -> None:
        for frame in frames:
            arrived = clock()
            with lock:
                latest[0], count[0] = (arrived, frame), count[0] + 1
            if stop.is_set():
                return

    reader = threading.Thread(target=read, name="latency-reader", daemon=True)
    start = clock()
    reader.start()
    samples = []
    while clock() - start < seconds and (reader.is_alive() or latest[0] is not None):
        with lock:
            item, latest[0] = latest[0], None
        if item is None:
            time.sleep(0.002)
            continue
        arrived, frame = item
        delay = parse_clock(decode(frame.image), arrived * 1000)
        if delay is not None:
            samples.append(delay)
    stop.set()
    elapsed = max(clock() - start, 1e-6)
    result = {"frames": count[0], "fps": round(count[0] / elapsed, 1), "decoded": len(samples)}
    if samples:
        s = sorted(samples)
        result |= {"delay_ms": round(statistics.median(s), 1), "p10_ms": round(s[len(s) // 10], 1),
                   "p90_ms": round(s[min(len(s) - 1, (9 * len(s)) // 10)], 1), "min_ms": round(s[0], 1),
                   "max_ms": round(s[-1], 1)}
    return result


def _path(settings: Settings) -> Path:
    return settings.runs_dir / FILE


def load_delays(settings: Settings) -> dict:
    try:
        return json.loads(_path(settings).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def save_delay(settings: Settings, source_id: str, result: dict) -> None:
    delays = load_delays(settings) | {source_id: result | {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    path = _path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(delays, indent=2) + "\n", encoding="utf-8")


def camera_delay(spec: SourceSpec, settings: Settings) -> float:
    """Seconds this source's frames trail reality: 0 for recorded videos, else measured or the configured default."""
    if spec.kind == "file":
        return 0.0
    measured = load_delays(settings).get(spec.id, {}).get("delay_ms")
    return (measured if measured is not None else settings.camera_delay_ms) / 1000
