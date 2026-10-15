"""Video source contract shared by file and RTSP sources."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Frame:
    t: float  # seconds: video time for files, time since first frame for live sources
    index: int
    image: np.ndarray  # BGR


class VideoSource(Protocol):
    fps: float

    def frames(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class SourceUnavailableError(RuntimeError):
    """A source can't be opened in this environment; the message says what to do instead."""
