"""SOP layout calibration helpers: where slots are, written back into the SOP YAML.

Slot centres for server_fan come from the user's reference boxes (`python -m training.reference_labels slots`);
`sopvision calibrate --check` draws the configured centres on a recording to verify them.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from sopvision.config import SopConfig
from sopvision.dataset import ACTION_TO_STEP, RunIndex

JUST_INSTALLED = 0.2  # seconds before a step's clip ends: its part is in place


@dataclass(frozen=True)
class CalTarget:
    slot: str
    step_id: str
    video: Path
    t: float


def calibration_targets(sop: SopConfig, run: RunIndex) -> list[CalTarget]:
    """For each slot, the moment in `run` right after the step that fills it."""
    end_of = {ACTION_TO_STEP[c.action]: seg.end for c, seg in zip(run.clips, run.segments)}
    targets = []
    for step in sop.steps:
        for slot in step.evidence:
            if step.id in end_of:
                targets.append(CalTarget(slot, step.id, run.raw_path, end_of[step.id] - JUST_INSTALLED))
    return targets


def write_layout(path: Path, coords: dict[str, tuple[float, float]]) -> None:
    """Set `layout.slots.<name>.at` for the given slots, keeping comments, order and everything else."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)  # keep the "  - id:" list style
    data = yaml.load(Path(path).read_text(encoding="utf-8"))
    for slot, (u, v) in coords.items():
        seq = CommentedSeq([round(float(u), 3), round(float(v), 3)])
        seq.fa.set_flow_style()
        data["layout"]["slots"][slot]["at"] = seq
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


def draw_layout_check(image: np.ndarray, chassis_box, sop: SopConfig) -> np.ndarray:
    """Configured slot centres (and match radii) drawn over a frame, relative to its chassis box."""
    out = image.copy()
    x1, y1, x2, y2 = chassis_box
    w, h = x2 - x1, y2 - y1
    cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
    for name, spec in sop.layout.slots.items():
        cx, cy = int(x1 + spec.at[0] * w), int(y1 + spec.at[1] * h)
        r = sop.slot_radius(name)  # normalised: x by chassis width, y by height → an ellipse in pixels
        cv2.ellipse(out, (cx, cy), (max(2, int(r * w)), max(2, int(r * h))), 0, 0, 360, (0, 200, 255), 2)
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(out, name, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out
