"""Append-only JSONL log of everything that happens in a run (spec §9, §4.1)."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

EventType = Literal["run_started", "scene_event", "decision", "decision_comparison", "step_update", "alert", "run_summary"]


class RunLogClosedError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    t: float
    type: EventType
    payload: dict


class RunLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def write(self, event: RunEvent) -> None:
        if self._f is None:
            raise RunLogClosedError(f"run log {self.path} is closed")
        self._f.write(json.dumps(asdict(event), default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def closed(self) -> bool:
        return self._f is None


def read_events(path: Path) -> list[RunEvent]:
    return [RunEvent(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
