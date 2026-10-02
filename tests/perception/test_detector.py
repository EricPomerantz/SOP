from pathlib import Path

import numpy as np
import pytest

from sopvision.config import ConfigError
from sopvision.perception.detector import Detection, YoloDetector
from sopvision.settings import Settings
from sopvision.sources.base import Frame

CLASSES = ["chassis", "fan", "fan_bay_empty", "psu", "psu_bay_empty", "cover", "hand"]


class _Arr:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _Boxes:
    def __init__(self, rows):
        self.xyxy = _Arr([r[0] for r in rows])
        self.cls = _Arr([float(r[1]) for r in rows])
        self.conf = _Arr([r[2] for r in rows])
        ids = [r[3] for r in rows]
        self.id = None if all(i is None for i in ids) else _Arr(ids)


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class FakeYolo:
    def __init__(self, names, boxes=()):
        self.names = names
        self.rows = list(boxes)
        self.calls = []

    def track(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_Result(self.rows)]


def test_class_mismatch_is_rejected():
    fake = FakeYolo(names={0: "chassis", 1: "fan"})
    with pytest.raises(ConfigError, match="don't match SOP detector.classes"):
        YoloDetector(Path("w.pt"), CLASSES, model_factory=lambda _: fake)


def test_results_become_detections():
    fake = FakeYolo(names=dict(enumerate(CLASSES)), boxes=[([10, 20, 30, 40], 1, 0.9, 7), ([0, 0, 5, 5], 6, 0.8, None)])
    dets = YoloDetector(Path("w.pt"), CLASSES, model_factory=lambda _: fake).detect(Frame(0.0, 0, np.zeros((8, 8, 3), np.uint8)))
    assert dets == [Detection("fan", (10, 20, 30, 40), 0.9, 7), Detection("hand", (0, 0, 5, 5), 0.8, None)]


def test_tracking_options_are_passed_through():
    fake = FakeYolo(names=dict(enumerate(CLASSES)))
    YoloDetector(Path("w.pt"), CLASSES, imgsz=960, model_factory=lambda _: fake).detect(Frame(0.0, 0, np.zeros((8, 8, 3), np.uint8)))
    assert fake.calls[0] == {"persist": True, "imgsz": 960, "tracker": "bytetrack.yaml", "verbose": False}


@pytest.mark.gpu
def test_real_weights_find_chassis():
    from sopvision.sources.file import FileSource

    weights = Path("models/server_fan/best.pt")
    if not weights.exists():
        pytest.skip("no trained weights yet")
    frame = next(iter(FileSource(Settings().sop_data_root / "raw" / "Install_12.MP4", mode="lockstep").frames()))
    assert any(d.cls == "chassis" for d in YoloDetector(weights, CLASSES).detect(frame))
