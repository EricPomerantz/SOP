"""An SOP's recordings and their step timelines, whatever the dataset's layout (SOP file: `dataset`).

timeline (new SOPs; see docs/lug-bolts-poc.md):
    <datasets_dir>/<dir>/splits.yaml              {train: [ids], val: [ids], test: [ids]}
    <datasets_dir>/<dir>/runs/<id>/video.mp4
    <datasets_dir>/<dir>/runs/<id>/steps.csv      step_id, start_s, end_s   (step_id may be off_sop)
    <datasets_dir>/<dir>/runs/<id>/torque.csv     time_s, ok, torque_nm     (optional)
clips (server_fan): sopvision.dataset's per-action clips, mapped to steps by `dataset.actions`.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from sopvision.config import ConfigError, SopConfig
from sopvision.dataset import Segment, load_index
from sopvision.settings import Settings

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Recording:
    id: str  # e.g. Install_12, or the run folder name
    split: Split
    video: Path
    segments: list[Segment]  # the step timeline, in order
    torque_log: Path | None  # recorded torque results, on the video's timeline
    valid: bool  # the timeline can be used as labels
    clean: bool = True  # the run follows the SOP: no off_sop, each step once and in order (anomaly is known False)

    @property
    def source_id(self) -> str:
        """How runs and logs name this recording: '<split>:<id>' (the frozen benchmark uses test:Install_12)."""
        return f"{self.split}:{self.id}"


def dataset_root(sop: SopConfig, settings: Settings) -> Path:
    if sop.dataset.format == "clips":
        return Path(settings.sop_data_root)
    return Path(settings.datasets_dir) / (sop.dataset.dir or sop.sop)


def load_recordings(sop: SopConfig, settings: Settings, probe=None) -> list[Recording]:
    root = dataset_root(sop, settings)
    if sop.dataset.format == "clips":
        return _clips(sop, root, probe)
    return _timeline(sop, root)


def _clips(sop: SopConfig, root: Path, probe) -> list[Recording]:
    actions = sop.dataset.actions
    if not actions:
        raise ConfigError(f"SOP {sop.sop}: dataset.format clips needs dataset.actions (clip action -> step)")
    out = []
    for run in load_index(root, **({"probe": probe} if probe else {})):
        t, segments = 0.0, []
        for clip in run.clips:
            segments.append(Segment(actions[clip.action], t, t + clip.duration))
            t += clip.duration
        out.append(Recording(f"Install_{run.run}", run.split, run.raw_path, segments, None, run.timing_valid))  # clean installs
    return out


def read_steps(path: Path, sop: SopConfig) -> list[Segment]:
    known = {s.id for s in sop.steps} | {"off_sop", "idle"}
    segments = []
    for row in csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()):
        step = row["step_id"].strip()
        if step not in known:
            raise ConfigError(f"{path}: unknown step '{step}' (SOP {sop.sop})")
        segments.append(Segment("off_sop" if step == "idle" else step, float(row["start_s"]), float(row["end_s"])))
    return sorted(segments, key=lambda s: s.start)


def _timeline(sop: SopConfig, root: Path) -> list[Recording]:
    splits_file = root / "splits.yaml"
    if not splits_file.exists():
        raise ConfigError(f"{splits_file} not found: list the run ids per split (train/val/test)")
    splits = yaml.safe_load(splits_file.read_text(encoding="utf-8")) or {}
    out = []
    for split in ("train", "val", "test"):
        for run_id in splits.get(split) or []:
            folder = root / "runs" / str(run_id)
            steps, torque = folder / "steps.csv", folder / "torque.csv"
            segments = read_steps(steps, sop) if steps.exists() else []
            out.append(Recording(str(run_id), split, folder / "video.mp4", segments,
                                 torque if torque.exists() else None, bool(segments),
                                 steps.exists() and follows_sop(_step_ids(steps), sop)))
    return out


def _step_ids(path: Path) -> list[str]:
    rows = csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines())
    return [r["step_id"].strip() for r in sorted(rows, key=lambda r: float(r["start_s"]))]


def follows_sop(step_ids: list[str], sop: SopConfig) -> bool:
    """A timeline (idle allowed) doing every SOP step exactly once, each after the steps that must precede it."""
    work = [s for s in step_ids if s != "idle"]
    if "off_sop" in work or sorted(work) != sorted(s.id for s in sop.steps):
        return False
    return all(sop.must_precede(s) <= set(work[:i]) for i, s in enumerate(work))
