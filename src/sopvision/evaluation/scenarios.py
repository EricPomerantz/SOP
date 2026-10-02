"""Simulated-mistake videos built from a recorded run's clips (spec §10).

Wrong order and skipped steps can't be faked from this footage (each clip shows all prior installs);
an incomplete run and a removed part can.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sopvision.dataset import RunIndex


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    positions: list[int]  # clip positions to play, in order
    reverse_last: bool  # play the last listed clip backwards (a part being taken out)
    expected_alerts: list[dict]


SCENARIOS = [
    ScenarioSpec("incomplete_run", [1, 2, 3, 4, 5], False,
                 [{"kind": "sop_incomplete", "missing": ["fan_5", "fan_6", "psu_1", "psu_2", "cover"]}]),
    ScenarioSpec("fan_removed", [1, 2, 3, 4, 4], True,
                 [{"kind": "part_removed", "step_id": "fan_3"}, {"kind": "sop_incomplete"}]),
]


def _frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def _reversed(path: Path):
    """Decode a clip backwards, holding it as JPEG bytes rather than raw frames to keep memory small."""
    buffered = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes() for f in _frames(path)]
    for data in reversed(buffered):
        yield cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def render_scenario(run: RunIndex, spec: ScenarioSpec, out_dir: Path, freeze_seconds: float = 3.0) -> Path:
    clips = {c.position: c for c in run.clips}
    probe = cv2.VideoCapture(str(clips[spec.positions[0]].path))
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    probe.release()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    last = None
    try:
        for i, pos in enumerate(spec.positions):
            backwards = spec.reverse_last and i == len(spec.positions) - 1
            for frame in (_reversed if backwards else _frames)(clips[pos].path):
                writer.write(frame)
                last = frame
        for _ in range(round(freeze_seconds * fps)):  # let the final state settle before the run ends
            writer.write(last)
    finally:
        writer.release()
    path.with_suffix(".json").write_text(json.dumps({"run": run.run, "expected_alerts": spec.expected_alerts}),
                                         encoding="utf-8")
    return path
