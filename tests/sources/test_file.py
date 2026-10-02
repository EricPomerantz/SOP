import pytest

from sopvision.sources.file import FileSource
from tests.helpers import FakeCapture, FakeClock


def test_lockstep_timestamps(tiny_video):
    ts = [f.t for f in FileSource(tiny_video, mode="lockstep").frames()]
    assert ts == pytest.approx([i / 10 for i in range(10)])


def test_frames_carry_index_and_image(tiny_video):
    frames = list(FileSource(tiny_video, mode="lockstep").frames())
    assert [f.index for f in frames] == list(range(10))
    assert frames[3].image.shape == (48, 64, 3) and abs(int(frames[3].image.mean()) - 60) <= 3


def test_lockstep_never_sleeps(tiny_video):
    clock = FakeClock()
    list(FileSource(tiny_video, mode="lockstep", sleep=clock.sleep, clock=clock).frames())
    assert clock.sleeps == []


def test_realtime_paces_by_speed(tiny_video):
    clock = FakeClock()
    list(FileSource(tiny_video, speed=2.0, sleep=clock.sleep, clock=clock).frames())
    assert sum(clock.sleeps) == pytest.approx(0.45, abs=0.01)


def test_set_speed_mid_stream(tiny_video):
    clock = FakeClock()
    src = FileSource(tiny_video, sleep=clock.sleep, clock=clock)
    for f in src.frames():
        if f.index == 4:
            src.set_speed(2.0)
    # 0.4 s of video at 1x, then 0.5 s at 2x
    assert sum(clock.sleeps) == pytest.approx(0.4 + 0.25, abs=0.01)


def test_zero_fps_metadata_uses_pos_msec():  # Review Focus 2
    cap = FakeCapture(fps=0.0, pos_msec=[0, 40, 80])
    assert [f.t for f in FileSource("x", mode="lockstep", open_capture=lambda _: cap).frames()] == [0, 0.04, 0.08]


def test_close_releases_capture():
    cap = FakeCapture(frames=3)
    src = FileSource("x", mode="lockstep", open_capture=lambda _: cap)
    next(iter(src.frames()))
    src.close()
    assert cap.released
