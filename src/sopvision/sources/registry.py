"""The sources a run can use in this environment, and how to open them."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from sopvision.settings import Settings
from sopvision.sources.base import SourceUnavailableError, VideoSource
from sopvision.sources.file import FileSource
from sopvision.sources.rtsp import LOW_LATENCY, RtspSource

TEST_RUNS = (12, 13)

__all__ = ["SourceSpec", "SourceUnavailableError", "list_sources", "open_source"]


class SourceSpec(BaseModel):
    id: str
    kind: Literal["file", "rtsp"]
    label: str
    path: Path | None = None
    url: str | None = None
    torque_log: Path | None = None  # a recording's torque results (times on its video's timeline), replayed with it


def list_sources(settings: Settings) -> list[SourceSpec]:
    specs = []
    for run in TEST_RUNS:
        path = settings.sop_data_root / "raw" / f"Install_{run}.MP4"
        if path.exists():
            specs.append(SourceSpec(id=f"test:Install_{run}", kind="file", label=f"Install_{run} (test run)", path=path))
    if settings.scenarios_dir.is_dir():
        for path in sorted(settings.scenarios_dir.glob("*.mp4")):
            label = "Simulated: " + path.stem.replace("_", " ")
            specs.append(SourceSpec(id=f"scenario:{path.stem}", kind="file", label=label, path=path))
    specs += _test_recordings(settings.datasets_dir)
    for i, url in enumerate(settings.sopvision_rtsp_urls):
        specs.append(SourceSpec(id=f"rtsp:{i}", kind="rtsp", label=f"RTSP camera {i + 1}", url=url))
    return specs


def _test_recordings(datasets_dir: Path) -> list[SourceSpec]:
    """The test-split runs of each timeline dataset (sopvision.recordings), with their torque logs."""
    specs = []
    for splits in sorted(Path(datasets_dir).glob("*/splits.yaml")):
        try:
            test = (yaml.safe_load(splits.read_text(encoding="utf-8")) or {}).get("test") or []
        except (OSError, yaml.YAMLError):
            continue
        for run_id in test:
            video = splits.parent / "runs" / str(run_id) / "video.mp4"
            if video.exists():
                torque = video.parent / "torque.csv"
                specs.append(SourceSpec(id=f"recording:{splits.parent.name}:{run_id}", kind="file",
                                        label=f"{splits.parent.name}: {run_id} (test recording)", path=video,
                                        torque_log=torque if torque.exists() else None))
    return specs


def open_source(spec: SourceSpec, settings: Settings,
                mode: Literal["realtime", "lockstep"] = "realtime") -> VideoSource:
    if spec.kind == "file":
        return FileSource(spec.path, mode=mode)
    return RtspSource(spec.url, capture_options=LOW_LATENCY if settings.rtsp_low_latency else None)
