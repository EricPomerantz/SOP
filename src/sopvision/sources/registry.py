"""The sources a run can use in this environment, and how to open them."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from sopvision.settings import Settings
from sopvision.sources.base import SourceUnavailableError, VideoSource
from sopvision.sources.file import FileSource
from sopvision.sources.rtsp import RtspSource
from sopvision.sources.webcam import WebcamSource

TEST_RUNS = (12, 13)
WEBCAM_IN_CONTAINER = "Webcam isn't available in the container — run natively or use an RTSP source."

__all__ = ["SourceSpec", "SourceUnavailableError", "list_sources", "open_source"]


class SourceSpec(BaseModel):
    id: str
    kind: Literal["file", "webcam", "rtsp"]
    label: str
    path: Path | None = None
    device: int | None = None
    url: str | None = None


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
    if not settings.sopvision_in_container:
        specs.append(SourceSpec(id="webcam:0", kind="webcam", label="USB webcam", device=0))
    for i, url in enumerate(settings.sopvision_rtsp_urls):
        specs.append(SourceSpec(id=f"rtsp:{i}", kind="rtsp", label=f"RTSP camera {i + 1}", url=url))
    return specs


def open_source(spec: SourceSpec, settings: Settings,
                mode: Literal["realtime", "lockstep"] = "realtime") -> VideoSource:
    if spec.kind == "file":
        return FileSource(spec.path, mode=mode)
    if spec.kind == "webcam":
        if settings.sopvision_in_container:
            raise SourceUnavailableError(WEBCAM_IN_CONTAINER)
        return WebcamSource(spec.device or 0)
    return RtspSource(spec.url)
