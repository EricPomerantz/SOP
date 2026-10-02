"""Annotated frames for the MJPEG stream: JPEG encoding and a latest-frame buffer shared across threads."""

import asyncio
import threading

import cv2
import numpy as np

POLL = 0.01


def encode_jpeg(image: np.ndarray, max_size: tuple[int, int] = (1280, 720), quality: int = 80) -> bytes:
    """Fit inside max_size keeping the aspect ratio (never upscaling), then JPEG-encode."""
    h, w = image.shape[:2]
    scale = min(1.0, max_size[0] / w, max_size[1] / h)
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class FrameBuffer:
    """Newest JPEG plus a sequence number; `put` from any thread, `await next(after)` from the event loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self._jpeg = b""

    def put(self, jpeg: bytes) -> None:
        with self._lock:
            self._seq += 1
            self._jpeg = jpeg

    def latest(self) -> tuple[int, bytes]:
        with self._lock:
            return self._seq, self._jpeg

    async def next(self, after_seq: int) -> tuple[int, bytes]:
        while True:
            seq, jpeg = self.latest()
            if seq > after_seq:
                return seq, jpeg
            await asyncio.sleep(POLL)
