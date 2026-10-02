import time

import numpy as np

from sopvision.sources.base import Frame
from sopvision.sources.latest import LatestFrameReader


class FastSource:
    fps = 1000.0

    def __init__(self, n: int):
        self.n = n
        self.closed = False

    def frames(self):
        for i in range(self.n):
            if self.closed:
                return
            time.sleep(0.001)
            yield Frame(t=i / 1000, index=i, image=np.zeros((2, 2, 3), np.uint8))

    def close(self):
        self.closed = True


def test_latest_reader_drops_old_frames():
    reader = LatestFrameReader(FastSource(300))
    seen = []
    while True:
        f = reader.get(timeout=1.0)
        if f is None:
            break
        seen.append(f.index)
        time.sleep(0.05)
    assert reader.exhausted and reader.dropped > 0
    assert seen == sorted(set(seen)) and len(seen) < 300


def test_get_returns_none_after_close():
    src = FastSource(10_000)
    reader = LatestFrameReader(src)
    assert reader.get(timeout=1.0) is not None
    reader.close()
    assert src.closed
