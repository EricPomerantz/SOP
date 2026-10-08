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
from sopvision.sources.registry import SourceSpec, check_available, list_sources
from sopvision.storage.report import RunReport, build_report
from sopvision.storage.runlog import RunLog, read_events

logger = logging.getLogger(__name__)


class RunConflictError(RuntimeError):
    pass


class NotFoundError(LookupError):
    pass


class NotPausableError(RuntimeError):
    pass


class RunInfo(BaseModel):
    run_id: str
    sop: str
    source: str
    decision_model: str = "jev"  # the model driving the tracker: jev | decider | laya


def startup_checks(settings: Settings, gpu: GpuReport) -> list[SopConfig]:
    """Everything `serve`/`eval` need, checked up front with messages that say what to fix."""
    require_gpu(settings.sopvision_in_container, gpu)
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
    from sopvision.sources.registry import open_source

    detectors: dict[str, YoloDetector] = {}

    def factory(ctx: RunContext, spec: SourceSpec):
        sop = ctx.sop
        if sop.sop not in detectors:
            detectors[sop.sop] = YoloDetector(sop.detector.weights, sop.detector.classes)
        detector = detectors[sop.sop]
        detector.reset()
        client, shadow = clients_for(settings)
        return LiveRun(ctx, open_source(spec, settings), detector, client, shadow)

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
        sop = self.sops.get(sop_id)
        spec = next((s for s in list_sources(self.settings) if s.id == source_id), None)
        if spec is None and source_id.startswith("webcam:") and source_id[7:].isdigit():
            # Not listed in the container, but still a webcam: let check_available explain why it can't be used.
            spec = SourceSpec(id=source_id, kind="webcam", label="USB webcam", device=int(source_id[7:]))
        if sop is None or spec is None:
            raise NotFoundError(f"Unknown {'SOP ' + sop_id if sop is None else 'source ' + source_id}")
        check_available(spec, self.settings)
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
