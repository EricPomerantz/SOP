"""Index of the server_fan recordings and their ground-truth step segments (spec §3.2).

Clip names are `{action:02d}_Install_{run}_{occurrence}_{position}.mp4`; clips are contiguous
slices of `raw/Install_{run}.MP4`, so cumulative clip durations give the step boundaries.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ACTION_TO_STEP: dict[int, str] = {
    **{i: f"fan_{i}" for i in range(1, 7)},
    7: "psu_1",
    8: "psu_2",
    9: "cover",
    10: "off_sop",
}
VAL_RUNS = {11}
TIMING_TOLERANCE = 0.1

_CLIP = re.compile(r"^(\d+)_Install_(\d+)_(\d+)_(\d+)\.mp4$")


def parse_clip_name(name: str) -> tuple[int, int, int, int]:
    """Return (action, run, occurrence, position) for a clip file name."""
    m = _CLIP.match(name)
    if not m:
        raise ValueError(f"not a clip name: {name}")
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    action: int
    run: int
    occurrence: int
    position: int
    duration: float


@dataclass(frozen=True)
class Segment:
    step_id: str
    start: float
    end: float


@dataclass(frozen=True)
class RunIndex:
    run: int
    split: Literal["train", "val", "test"]
    clips: list[ClipInfo]
    raw_path: Path
    raw_duration: float

    @property
    def segments(self) -> list[Segment]:
        out, t = [], 0.0
        for clip in self.clips:
            out.append(Segment(ACTION_TO_STEP[clip.action], t, t + clip.duration))
            t += clip.duration
        return out

    @property
    def timing_valid(self) -> bool:
        return abs(sum(c.duration for c in self.clips) - self.raw_duration) <= TIMING_TOLERANCE

    @property
    def actions_by_position(self) -> list[int]:
        return [c.action for c in self.clips]


def video_duration(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0:
            raise ValueError(f"{path}: no frame rate in metadata")
        return frames / fps
    finally:
        cap.release()


def load_index(data_root: Path, probe: Callable[[Path], float] = video_duration) -> list[RunIndex]:
    data_root = Path(data_root)
    runs = []
    for split_dir in ("train", "test"):
        for run_dir in sorted((data_root / split_dir).glob("Install_*")):
            if not run_dir.is_dir():
                continue
            clips = []
            for path in run_dir.glob("*.mp4"):
                action, run, occurrence, position = parse_clip_name(path.name)
                clips.append(ClipInfo(path, action, run, occurrence, position, probe(path)))
            clips.sort(key=lambda c: c.position)
            run = int(run_dir.name.split("_")[1])
            split = "test" if split_dir == "test" else "val" if run in VAL_RUNS else "train"
            raw = data_root / "raw" / f"Install_{run}.MP4"
            runs.append(RunIndex(run, split, clips, raw, probe(raw)))
    return sorted(runs, key=lambda r: r.run)
