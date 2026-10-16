"""Jobs as folders: the app queues them, the worker service runs them (sopvision.jobs.worker).

    <jobs_dir>/<job id>/job.json   what to run and how far it got (written by the app once, then only by the worker)
                        log.txt    every step's output
                        cancel     present: the app asked the worker to stop the job

A job is an ordered list of steps, each a command line run in the app's working directory. Commands come from the
job kinds (sopvision.jobs.kinds), never from a request. The app and the worker never write the same file: the app
creates job.json and the cancel marker, the worker updates job.json (atomically, via a temporary file).
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

State = Literal["queued", "running", "done", "failed", "cancelled"]
FINISHED = ("done", "failed", "cancelled")


@dataclass
class Step:
    name: str
    argv: list[str]
    state: State = "queued"
    started: float | None = None
    ended: float | None = None
    returncode: int | None = None


@dataclass
class Job:
    id: str
    kind: str
    title: str
    steps: list[Step]
    sop: str | None = None
    params: dict = field(default_factory=dict)
    created: float = 0.0
    state: State = "queued"
    error: str | None = None
    result: dict = field(default_factory=dict)  # paths the job wrote, for the page that started it

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(**{**d, "steps": [Step(**s) for s in d["steps"]]})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def current_step(self) -> Step | None:
        return next((s for s in self.steps if s.state == "running"), None)


class JobNotFound(Exception):
    pass


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, job_id: str) -> Path:
        if not job_id or "/" in job_id or job_id.startswith("."):
            raise JobNotFound(job_id)
        return self.root / job_id

    def create(self, kind: str, title: str, steps: list[Step], sop: str | None = None, params: dict | None = None,
               result: dict | None = None) -> Job:
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        n = 1
        while (self.root / f"{stamp}-{kind}-{n}").exists():
            n += 1
        job = Job(f"{stamp}-{kind}-{n}", kind, title, steps, sop, params or {}, time.time(), result=result or {})
        self._dir(job.id).mkdir()
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        path = self._dir(job.id) / "job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def get(self, job_id: str) -> Job:
        path = self._dir(job_id) / "job.json"
        try:
            return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise JobNotFound(job_id) from None

    def list(self, sop: str | None = None, kind: str | None = None) -> list[Job]:
        """Newest first."""
        jobs = []
        for path in sorted(self.root.glob("*/job.json"), reverse=True):
            try:
                job = Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue  # a folder being created, or not a job
            if (sop is None or job.sop == sop) and (kind is None or job.kind == kind):
                jobs.append(job)
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def next_queued(self) -> Job | None:
        """The oldest queued job."""
        queued = [j for j in self.list() if j.state == "queued"]
        return min(queued, key=lambda j: j.created) if queued else None

    def log_path(self, job_id: str) -> Path:
        return self._dir(job_id) / "log.txt"

    def log_tail(self, job_id: str, max_bytes: int = 20_000) -> str:
        path = self.log_path(job_id)
        if not path.exists():
            self.get(job_id)  # unknown job -> JobNotFound
            return ""
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - max_bytes))
            return f.read().decode("utf-8", errors="replace")

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state not in FINISHED:
            (self._dir(job_id) / "cancel").touch()
        return job

    def cancel_requested(self, job_id: str) -> bool:
        return (self._dir(job_id) / "cancel").exists()
