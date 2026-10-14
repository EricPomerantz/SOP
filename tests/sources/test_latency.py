import time

from sopvision.settings import Settings
from sopvision.sources.base import Frame
from sopvision.sources.latency import (PREFIX, WRAP_MS, camera_delay, load_delays, measure, parse_clock,
                                       save_delay)
from sopvision.sources.registry import SourceSpec


def camera(fps: float, delay_s: float, n: int):
    """A live camera: a frame every 1/fps s of real time, showing the QR clock as it was `delay_s` before arrival."""
    for i in range(n):
        time.sleep(1 / fps)
        yield Frame(i / fps, i, f"{PREFIX}0:{((time.time() - delay_s) * 1000) % WRAP_MS}")


def test_measure_reports_the_delay_of_the_frames():
    r = measure(camera(fps=50, delay_s=0.3, n=200), seconds=1.0, decode=lambda image: image)
    assert r["decoded"] >= 20 and 295 <= r["delay_ms"] <= 330


def test_a_slow_decoder_does_not_build_a_backlog():
    """Decoding (60 ms) is slower than the frame interval (20 ms): only the newest frame is decoded, so the measured
    delay stays at the camera's 100 ms instead of growing with every frame."""
    def slow(image):
        time.sleep(0.06)
        return image

    r = measure(camera(fps=50, delay_s=0.1, n=500), seconds=1.5, decode=slow)
    assert r["frames"] >= 60  # read at the camera's pace, not the decoder's
    assert r["max_ms"] < 100 + 60 + 40  # at most one decode plus a frame or two behind


def test_measure_without_any_qr_reports_no_delay():
    frames = (Frame(i, i, None) for i in range(10))
    r = measure(frames, seconds=0.2, decode=lambda image: None)
    assert r["decoded"] == 0 and "delay_ms" not in r


def test_saved_delay_is_used_for_that_camera_only(tmp_path):
    s = Settings(runs_dir=tmp_path, camera_delay_ms=150)
    cam, other = SourceSpec(id="rtsp:0", kind="rtsp", label="c"), SourceSpec(id="rtsp:1", kind="rtsp", label="d")
    video = SourceSpec(id="test:Install_12", kind="file", label="v")
    save_delay(s, "rtsp:0", {"delay_ms": 420.0})
    assert load_delays(s)["rtsp:0"]["delay_ms"] == 420.0
    assert (camera_delay(cam, s), camera_delay(other, s), camera_delay(video, s)) == (0.42, 0.15, 0.0)


def test_decode_qr_reads_the_latency_page_format():
    import cv2

    from sopvision.sources.latency import decode_qr

    text = f"{PREFIX}50:34567890"
    code = cv2.QRCodeEncoder.create().encode(text)  # 1 px per module
    image = cv2.resize(code, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    image = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    assert decode_qr(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)) == text


def test_parse_clock_unwraps_the_time_and_removes_half_the_hold():
    arrived = 1_791_234_568_300.0  # 34_568_300 modulo 1e8
    assert parse_clock("SVL:50:34567890", arrived) == 410 - 25
    assert parse_clock("SVL:0:99999900", 1_800_000_000_100.0) == 200  # shown just before the wrap
    assert parse_clock("other", arrived) is None and parse_clock(None, arrived) is None
    assert parse_clock("SVL:x:1", arrived) is None
