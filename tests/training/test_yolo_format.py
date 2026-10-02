import pytest

from training.yolo_format import LabeledBox, from_yolo, to_yolo

CLASSES = ["chassis", "fan", "fan_bay_empty", "psu", "psu_bay_empty", "cover", "hand"]


def test_yolo_round_trip():
    b = LabeledBox("fan", (100.0, 50.0, 300.0, 250.0), 0.9)
    back = from_yolo(to_yolo(b, 1920, 1080, CLASSES), 1920, 1080, CLASSES)
    assert back.cls == "fan" and back.xyxy == pytest.approx(b.xyxy, abs=1e-2)


def test_yolo_line_format():
    assert to_yolo(LabeledBox("chassis", (0, 0, 960, 540), 1.0), 1920, 1080, CLASSES) == "0 0.250000 0.250000 0.500000 0.500000"


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        to_yolo(LabeledBox("dog", (0, 0, 1, 1), 1.0), 10, 10, CLASSES)
