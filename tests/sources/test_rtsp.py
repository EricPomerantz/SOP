import itertools

from sopvision.sources.rtsp import RtspSource
from tests.helpers import FakeCapture


def test_rtsp_reconnects_with_backoff():
    caps = [FakeCapture(fail_after=2), FakeCapture(open_ok=False), FakeCapture(frames=3)]
    lost, sleeps = [], []
    src = RtspSource("rtsp://x", open_capture=lambda _: caps.pop(0), sleep=sleeps.append, on_lost=lambda: lost.append(1))
    got = list(itertools.islice(src.frames(), 5))
    assert len(got) == 5 and lost == [1] and sleeps == [1, 2]


def test_backoff_caps_at_last_delay():
    caps = [FakeCapture(open_ok=False) for _ in range(7)] + [FakeCapture(frames=1)]
    sleeps = []
    src = RtspSource("rtsp://x", open_capture=lambda _: caps.pop(0), sleep=sleeps.append, backoff=(1, 2, 4))
    next(iter(src.frames()))
    assert sleeps == [1, 2, 4, 4, 4, 4, 4]


def test_indices_keep_increasing_across_reconnects():
    caps = [FakeCapture(fail_after=2), FakeCapture(frames=2)]
    src = RtspSource("rtsp://x", open_capture=lambda _: caps.pop(0), sleep=lambda s: None)
    assert [f.index for f in itertools.islice(src.frames(), 4)] == [0, 1, 2, 3]
