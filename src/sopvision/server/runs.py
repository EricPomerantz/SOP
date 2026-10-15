"""Startup checks and the one-run-at-a-time manager behind the API."""

import logging
import secrets
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from sopvision.config import ConfigError, SopConfig, list_sops
from sopvision.doctor import GpuReport, require_gpu
from sopvision.pipeline import RunContext
from sopvision.server.frames import FrameBuffer
from sopvision.settings import Settings
from sopvision.sources.registry import SourceSpec, SourceUnavailableError, list_sources
from sopvision.storage.report import RunReport, build_report
from sopvision.storage.runlog import RunLog, read_events

logger = logging.getLogger(__name__)


class RunConflictError(RuntimeError):
    pass


class NotFoundError(LookupError):
    pass


class NotPausableError(RuntimeError):
    pass


class NoTorqueToolError(RuntimeError):
    pass


class RunInfo(BaseModel):
    run_id: str
    sop: str
    source: str
    decision_model: str = "jev"  # the model driving the tracker: jev | decider | laya


def startup_checks(settings: Settings, gpu: GpuReport) -> list[SopConfig]:
    """Everything `serve`/`eval` need, checked up front with messages that say what to fix."""
    require_gpu(gpu)
    settings.check_data_root()
    sops = list_sops(settings.sops_dir)
    for sop in sops:
        if not Path(sop.detector.weights).exists():
            raise ConfigError(f"Weights not found: {sop.detector.weights}. Train the detector (training/train.py) "
                              "or fix detector.weights.")
    check_decision_providers(settings)
    return sops


def check_decision_providers(settings: Settings, readiness=None) -> None:
    """Jev needs its key; Decider must answer a real request when it decides, and should when it shadows."""
    from sopvision.decide.providers import local_provider, local_readiness, uses_jev

    if uses_jev(settings) and not settings.typesafe_api_key.get_secret_value():
        raise ConfigError("TYPESAFE_API_KEY is not set. Add it to .env.")
    provider = local_provider(settings)
    if provider:
        r = (readiness or local_readiness)(settings, provider)
        if not r.ready and settings.decision_mode != "compare":
            raise ConfigError(f"DECISION_MODE={provider} but the local {provider} service isn't ready: {r.detail}. "
                              f"Start it with `docker compose --profile {provider} up -d {provider}` or set DECISION_MODE=jev.")
        if not r.ready:
            # Compare mode keeps Jev's behaviour; the shadow's failures are recorded per request.
            logger.warning(f"DECISION_MODE=compare: {r.detail}; Jev still decides, {provider} errors will be logged")


def default_run_factory(settings: Settings) -> Callable:
    """Builds real LiveRuns; detectors are loaded once per SOP and reused across runs."""
    from sopvision.decide.providers import clients_for
    from sopvision.perception.detector import YoloDetector
    from sopvision.pipeline import LiveRun
    from sopvision.sources.latency import camera_delay
    from sopvision.sources.registry import open_source

    detectors: dict[str, YoloDetector] = {}

    def factory(ctx: RunContext, spec: SourceSpec):
        sop = ctx.sop
        if sop.sop not in detectors:
            detectors[sop.sop] = YoloDetector(sop.detector.weights, sop.detector.classes)
        detector = detectors[sop.sop]
        detector.reset()
        client, shadow = clients_for(settings)
        return LiveRun(ctx, open_source(spec, settings), detector, client, shadow, camera_delay=camera_delay(spec, settings))

    return factory


class RunManager:
    def __init__(self, settings: Settings, sops: dict[str, SopConfig], run_factory: Callable):
        self.settings, self.sops, self.run_factory = settings, sops, run_factory
        self.frames = FrameBuffer()
        self.subscribers: set = set()  # asyncio.Queue per WebSocket client
        self.run = None
        self.info: RunInfo | None = None
        self.spec: SourceSpec | None = None
        self.last_decision: dict | None = None
        self.recording = None  # RecordingSession while recording (server/recording.py)

    def broadcast(self, msg: dict) -> None:
        if msg.get("type") == "decision":
            self.last_decision = msg
        for queue in list(self.subscribers):
            queue.put_nowait(msg)

    def _active(self) -> bool:
        return self.run is not None and not self.run.done.is_set()

    async def start(self, sop_id: str, source_id: str) -> RunInfo:
        if self._active():
            raise RunConflictError("A run is already in progress. Stop it first.")
        if self.recording is not None:
            raise RunConflictError("A recording is in progress. Stop it first.")
        sop = self.sops.get(sop_id)
        spec = next((s for s in list_sources(self.settings) if s.id == source_id), None)
        if spec is None and source_id.startswith("webcam:"):
            raise SourceUnavailableError("USB webcams are no longer supported. Use an RTSP source or a video file.")
        if sop is None or spec is None:
            raise NotFoundError(f"Unknown {'SOP ' + sop_id if sop is None else 'source ' + source_id}")
        run_id = time.strftime("run-%Y%m%d-%H%M%S-") + secrets.token_hex(2)  # unique even within one second
        runlog = RunLog(self.settings.runs_dir / f"{run_id}.jsonl")
        ctx = RunContext(run_id, sop, spec, runlog, self.broadcast, self.frames)
        self.run, self.spec, self.last_decision = self.run_factory(ctx, spec), spec, None
        from sopvision.decide.providers import authoritative_provider

        self.info = RunInfo(run_id=run_id, sop=sop.sop, source=spec.id, decision_model=authoritative_provider(self.settings))
        await self.run.start()
        return self.info

    def _require(self, run_id: str):
        if self.info is None or self.info.run_id != run_id:
            raise NotFoundError(f"Unknown run {run_id}")
        return self.run

    async def stop(self, run_id: str) -> None:
        await self._require(run_id).stop()

    def torque(self, run_id: str, ok: bool, torque_nm: float | None) -> float:
        run = self._require(run_id)
        if not self.sops[self.info.sop].layout.tools:
            raise NoTorqueToolError(f"SOP {self.info.sop} has no torque tool, so it takes no torque results.")
        return run.torque(ok, torque_nm)

    # -- recording (no detector: for collecting a new SOP's data) --------------------------------------------------
    def recording_sops(self) -> list[dict]:
        """SOP files whose dataset is in the timeline format, drafts included (they have no detector yet)."""
        return self._scan_sops()[0]

    def recording_sop_errors(self) -> list[dict]:
        """SOP files that don't load, with the reason: shown on the Record page instead of silently leaving them out."""
        return self._scan_sops()[1]

    def _scan_sops(self) -> tuple[list[dict], list[dict]]:
        from sopvision.config import load_sop
        from sopvision.recordings import dataset_root

        files = sorted(self.settings.sops_dir.glob("*.yaml")) + sorted((self.settings.sops_dir / "drafts").glob("*.yaml"))
        out, errors = [], []
        for path in files:
            try:
                sop = load_sop(path)
            except ConfigError as e:
                errors.append({"file": str(path), "error": str(e)})
                continue
            if sop.dataset.format == "timeline":
                out.append({"file": str(path), "id": sop.sop, "name": sop.display_name,
                            "dataset": str(dataset_root(sop, self.settings)), "torque": bool(sop.layout.tools)})
        return out, errors

    def start_recording(self, sop_file: str, source_id: str, split: str, name: str | None, fps: float = 25.0) -> dict:
        from sopvision.config import load_sop
        from sopvision.server.recording import RecordingSession

        if self._active() or self.recording is not None:
            raise RunConflictError("Stop the current run or recording first.")
        allowed = {s["file"] for s in self.recording_sops()}
        if sop_file not in allowed:
            raise NotFoundError(f"Unknown SOP file {sop_file}")
        spec = next((s for s in list_sources(self.settings) if s.id == source_id), None)
        if spec is None:
            raise NotFoundError(f"Unknown source {source_id}")
        if spec.kind == "file":
            raise ConfigError("Record from a live camera, not a video file.")
        run_id = name or time.strftime("r%Y%m%d-%H%M%S")
        session = RecordingSession(self.settings, load_sop(Path(sop_file)), sop_file, spec, split, run_id, self.frames, fps)
        session.start()
        self.recording = session
        return session.status()

    def recording_torque(self, ok: bool, torque_nm: float | None) -> float:
        if self.recording is None:
            raise NotFoundError("No recording in progress")
        t = self.recording.torque(ok, torque_nm)
        if t is None:
            raise NotPausableError("No frames recorded yet; wait for the video to start")
        return t

    def discard_recording(self) -> dict:
        if self.recording is None:
            raise NotFoundError("No recording in progress")
        session, self.recording = self.recording, None
        return session.discard()

    def stop_recording(self) -> dict:
        if self.recording is None:
            raise NotFoundError("No recording in progress")
        session, self.recording = self.recording, None
        return session.stop()

    def control(self, run_id: str, action: str, speed: float | None) -> None:
        run = self._require(run_id)
        if self.spec is None or self.spec.kind != "file":
            raise NotPausableError("Only recorded videos can be paused.")
        if action == "pause":
            run.pause()
        elif action == "resume":
            run.resume()
        if speed:
            run.set_speed(speed)

    def current(self) -> dict | None:
        if not self._active():
            return None
        return {**self.info.model_dump(), **self.run.snapshot(), "last_decision": self.last_decision}

    def report(self, run_id: str) -> RunReport:
        path = self.settings.runs_dir / f"{run_id}.jsonl"
        if not path.exists():
            raise NotFoundError(f"Unknown run {run_id}")
        return build_report(read_events(path))
