import json
from pathlib import Path

import cv2
import numpy as np

from sopvision.dataset import ClipInfo, RunIndex
from sopvision.evaluation.scenarios import SCENARIOS, ScenarioSpec, render_scenario

SEQ = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BITS = 5  # a frame's index is drawn as 5 black/white bands, which survive two lossy encodes


def index_frame(n):
    img = np.zeros((48, 64, 3), np.uint8)
    for b in range(BITS):
        if (n >> b) & 1:
            img[:, b * 12:(b + 1) * 12] = 255
    return img


def frame_index(img):
    return sum(1 << b for b in range(BITS) if img[:, b * 12 + 2:(b + 1) * 12 - 2].mean() > 127)


def tiny_run(tmp_path, clips, fps=10):
    """Clips whose frames carry their global frame index (see index_frame)."""
    infos, n = [], 0
    for pos, count in enumerate(clips, start=1):
        path = tmp_path / f"clip{pos}.mp4"
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
        for _ in range(count):
            w.write(index_frame(n))
            n += 1
        w.release()
        infos.append(ClipInfo(path, SEQ[pos - 1], 12, 1, pos, count / fps))
    return RunIndex(12, "test", infos, tmp_path / "raw.MP4", n / fps)


def read_levels(path):
    cap, levels = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return levels
        levels.append(frame_index(f))


def test_reverse_and_freeze(tmp_path):
    run = tiny_run(tmp_path, clips=[5, 5, 5, 5])
    path = render_scenario(run, ScenarioSpec("x", [1, 2, 2], True, []), tmp_path / "out", freeze_seconds=0.3)
    levels = read_levels(path)
    assert levels[:10] == list(range(10)) and levels[10:15] == [9, 8, 7, 6, 5] and levels[15:] == [5, 5, 5]
    assert json.loads(path.with_suffix(".json").read_text())["expected_alerts"] == []


def test_sidecar_names_run_and_expected_alerts(tmp_path):
    run = tiny_run(tmp_path, clips=[3, 3])
    spec = ScenarioSpec("y", [1, 2], False, [{"kind": "sop_incomplete"}])
    path = render_scenario(run, spec, tmp_path / "out", freeze_seconds=0)
    assert json.loads(path.with_suffix(".json").read_text()) == {"run": 12, "expected_alerts": [{"kind": "sop_incomplete"}]}
    assert len(read_levels(path)) == 6


def test_defined_scenarios_match_the_spec():
    by_name = {s.name: s for s in SCENARIOS}
    assert by_name["incomplete_run"].positions == [1, 2, 3, 4, 5]
    assert by_name["incomplete_run"].expected_alerts == [
        {"kind": "sop_incomplete", "missing": ["fan_5", "fan_6", "psu_1", "psu_2", "cover"]}]
    assert by_name["fan_removed"].positions == [1, 2, 3, 4, 4] and by_name["fan_removed"].reverse_last
    assert {"kind": "part_removed", "step_id": "fan_3"} in by_name["fan_removed"].expected_alerts


def test_hold_before_reverse_keeps_the_part_in_place(tmp_path):
    run = tiny_run(tmp_path, clips=[5, 5])
    spec = ScenarioSpec("z", [1, 2, 2], True, [], hold_before_reverse=0.3)
    levels = read_levels(render_scenario(run, spec, tmp_path / "out", freeze_seconds=0))
    assert levels == list(range(10)) + [9, 9, 9] + [9, 8, 7, 6, 5]


def test_fan_removed_holds_long_enough_to_count_as_done():
    fan_removed = next(s for s in SCENARIOS if s.name == "fan_removed")
    assert fan_removed.hold_before_reverse >= 3.0
