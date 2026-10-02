import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from sopvision.calibrate import calibration_targets, draw_layout_check, write_layout
from sopvision.dataset import ClipInfo, RunIndex
from tests.helpers import CHASSIS


def fake_run_from_seq(seq, durations):
    clips = [ClipInfo(Path(f"c{i}.mp4"), a, 4, 1, i + 1, d) for i, (a, d) in enumerate(zip(seq, durations))]
    return RunIndex(4, "train", clips, Path("raw.MP4"), sum(durations))


def test_targets_use_end_of_installing_step(sop):
    run = fake_run_from_seq([10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], durations=[5] * 11)
    t = {c.slot: c for c in calibration_targets(sop, run)}
    assert t["fan_bay_1"].step_id == "fan_1" and t["fan_bay_1"].t == pytest.approx(9.8)
    assert t["cover"].step_id == "cover" and t["cover"].t == pytest.approx(49.8)
    assert len(t) == 9


def test_write_layout_preserves_comments(tmp_path):
    p = tmp_path / "s.yaml"
    shutil.copy("sops/server_fan.yaml", p)
    before = yaml.safe_load(p.read_text(encoding="utf-8"))
    write_layout(p, {"fan_bay_1": (0.4567, 0.7891)})
    text = p.read_text(encoding="utf-8")
    assert "# slot coords are normalised" in text and "[0.457, 0.789]" in text
    assert "\n  - id: fan_1\n" in text  # list indentation kept
    data = yaml.safe_load(text)
    assert data["layout"]["slots"]["fan_bay_1"]["at"] == [0.457, 0.789]
    assert data["layout"]["slots"]["fan_bay_2"] == before["layout"]["slots"]["fan_bay_2"]  # untouched
    assert data["steps"][0]["id"] == "fan_1"


def test_draw_layout_check_marks_each_slot_centre(sop):
    img = np.zeros((700, 1200, 3), np.uint8)
    out = draw_layout_check(img, CHASSIS, sop)
    u, v = sop.layout.slots["fan_bay_1"].at
    x, y = int(100 + u * 1000), int(100 + v * 500)
    assert out[y, x].any() and not img.any()
