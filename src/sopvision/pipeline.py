"""Wires sources, detector, scene, Jev and tracker into a run (spec §4.2).

Live: a pipeline thread reads the newest frame, detects, builds the scene and draws the overlay;
the event loop runs the tracker, the Jev decider, the run log and the broadcast. Video never
waits on Jev. Lockstep (eval): every frame in order, all timers on video time.
"""

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from sopvision.config import SopConfig
from sopvision.decide.decider import JevDecider
from sopvision.decide.types import Decision
from sopvision.messages import AlertMsg, DecisionMsg, RunSummaryMsg, SceneMsg, StepUpdateMsg, to_ws
from sopvision.perception.detector import Detector
from sopvision.scene.builder import SceneBuilder
from sopvision.scene.overlay import OverlayStats, draw_overlay
from sopvision.scene.types import SceneEvent, SceneState
from sopvision.server.frames import FrameBuffer, encode_jpeg
from sopvision.sop.tracker import SopTracker
from sopvision.sop.types import Alert, StepState, StepUpdate
from sopvision.sources.base import VideoSource
from sopvision.sources.latest import LatestFrameReader
from sopvision.sources.registry import SourceSpec
from sopvision.storage.report import build_report
from sopvision.storage.runlog import RunEvent, RunLog

SCENE_MSG_INTERVAL = 0.1  # at most 10 scene messages per second to the dashboard
POLL_INTERVAL = 0.05


@dataclass(frozen=True)
class FrameRecord:
    t: float
    slots: dict[str, str]


@dataclass(frozen=True)
class LockstepResult:
    frames: list[FrameRecord]
    steps: list[StepState]
    alerts: list[Alert]
    latencies_ms: list[float]
    decisions: list[tuple[float, str]]  # (request time, Jev's current_step answer)


@dataclass
class RunContext:
    run_id: str
    sop: SopConfig
    source_spec: SourceSpec
    runlog: RunLog | None
    broadcast: Callable[[dict], None]
    frames: FrameBuffer


class RunCore:
    """Tracker + logging + broadcast, shared by live and lockstep runs so they behave identically."""

    def __init__(self, sop: SopConfig, run_id: str, source_id: str, runlog: RunLog | None,
                 broadcast: Callable[[dict], None] | None = None):
        self.sop, self.run_id, self.runlog = sop, run_id, runlog
        self.broadcast = broadcast or (lambda msg: None)
        self.tracker = SopTracker(sop)
        self.events: list[RunEvent] = []
        self.alerts: list[Alert] = []
        self.latencies_ms: list[float] = []
        self.decisions: list[tuple[float, str]] = []
        self.last_scene: SceneState | None = None
        self.last_t = 0.0
        self._last_scene_msg = -1e9
        self.closed = False
        self._log(0.0, "run_started", {"sop": sop.sop, "source": source_id, "steps": [s.id for s in sop.steps]})

    def _log(self, t: float, type_: str, payload: dict) -> None:
        event = RunEvent(self.run_id, t, type_, payload)
        self.events.append(event)
        if self.runlog is not None:
            self.runlog.write(event)

    def emit(self, items: list[StepUpdate | Alert]) -> None:
        durations = {s.id: s.duration for s in self.tracker.steps()}
        for item in items:
            if isinstance(item, StepUpdate):
                duration = durations.get(item.step_id) if item.status == "done" else None
                self._log(item.t, "step_update", {"step_id": item.step_id, "status": item.status,
                                                  "source": item.source, "duration": duration})
                self.broadcast(to_ws(StepUpdateMsg(t=item.t, step_id=item.step_id, status=item.status,
                                                   source=item.source, duration=duration)))
            else:
                self.alerts.append(item)
                self._log(item.t, "alert", {"level": item.level, "kind": item.kind, "message": item.message,
                                            "step_id": item.step_id})
                self.broadcast(to_ws(AlertMsg(t=item.t, level=item.level, kind=item.kind, message=item.message,
                                              step_id=item.step_id)))

    def on_scene(self, scene: SceneState, events: list[SceneEvent]) -> None:
        if self.closed:
            return
        self.last_scene, self.last_t = scene, scene.t
        for e in events:
            self._log(e.t, "scene_event", {"description": e.describe(), "kind": e.kind})
            if e.kind == "chassis_lost":
                self.emit([Alert(e.t, "warning", "chassis_lost", "Chassis not visible")])
        self.emit(self.tracker.on_scene(scene))
        if scene.t - self._last_scene_msg >= SCENE_MSG_INTERVAL:
            self._last_scene_msg = scene.t
            self.broadcast(to_ws(SceneMsg(t=scene.t, scene=scene.to_jev_dict())))

    def on_decision(self, d: Decision) -> None:
        if self.closed or self.last_scene is None:
            return
        a = d.answers
        self.latencies_ms.append(a.latency_ms)
        self.decisions.append((d.t_request, a.current_step))
        self._log(d.t_request, "decision", {"version": d.version, "active_step_id": d.active_step_id} | asdict(a))
        self.broadcast(to_ws(DecisionMsg(t=d.t_request, current_step=a.current_step, confidence=a.current_confidence,
                                         active_step_id=d.active_step_id, active_step_done=a.active_step_done,
                                         anomaly=a.anomaly, off_sop=a.current_probabilities.get("off_sop", 0.0),
                                         latency_ms=a.latency_ms)))
        self.emit(self.tracker.on_decision(d, self.last_scene))

    def on_status(self, status: str, t: float) -> None:
        if not self.closed:
            self.emit(self.tracker.set_jev_online(status == "online", t))

    def source_lost(self) -> None:
        if not self.closed:
            self.emit([Alert(self.last_t, "warning", "source_lost", "Video source lost — reconnecting")])

    def finish(self, check_incomplete: bool = True) -> dict:
        """End the run: incomplete check, summary event, close the log. Nothing is written afterwards.

        A run aborted by a system failure (source or processing error) skips the incomplete check: the operator
        didn't leave the SOP unfinished, the system stopped watching.
        """
        if check_incomplete:
            self.emit(self.tracker.end_run(self.last_t))
        report = asdict(build_report(self.events))
        self._log(self.last_t, "run_summary", {"report": report})
        self.broadcast(to_ws(RunSummaryMsg(t=self.last_t, report=report)))
        self.closed = True
        if self.runlog is not None:
            self.runlog.close()
        return report


async def run_lockstep(sop: SopConfig, source: VideoSource, detector: Detector, client, runlog: RunLog | None = None,
                       run_id: str = "eval", source_id: str = "file") -> LockstepResult:
    core = RunCore(sop, run_id, source_id, runlog)
    builder = SceneBuilder(sop)
    decider = JevDecider(sop, client, "lockstep", on_decision=core.on_decision, on_status=core.on_status)
    frames: list[FrameRecord] = []
    for frame in source.frames():
        scene, events = builder.update(frame.t, detector.detect(frame))
        core.on_scene(scene, events)
        decider.observe(scene, events, core.tracker.statuses(), core.tracker.active_step_id())
        await decider.poll(frame.t)
        frames.append(FrameRecord(frame.t, {k: v.state for k, v in scene.slots.items()}))
    await decider.aclose()
    core.finish()
    return LockstepResult(frames, core.tracker.steps(), core.alerts, core.latencies_ms, core.decisions)


class LiveRun:
    def __init__(self, ctx: RunContext, source: VideoSource, detector: Detector, client):
        self.ctx, self.source, self.detector = ctx, source, detector
        self.core = RunCore(ctx.sop, ctx.run_id, ctx.source_spec.id, ctx.runlog, ctx.broadcast)
        self.builder = SceneBuilder(ctx.sop)
        self.decider = JevDecider(ctx.sop, client, "live", on_decision=self.core.on_decision,
                                  on_status=self.core.on_status)
        self.done = asyncio.Event()
        self.reader: LatestFrameReader | None = None
        self.report: dict | None = None
        self._stopping = False
        self._aborted = False
        self._stop_lock = asyncio.Lock()
        self._jev_ms: float | None = None
        if hasattr(source, "on_lost"):
            source.on_lost = self._source_lost_threadsafe

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.reader = LatestFrameReader(self.source)
        self._thread = threading.Thread(target=self._pipeline, name="sop-pipeline", daemon=True)
        self._thread.start()
        self._poller = asyncio.create_task(self._poll())

    def _pipeline(self) -> None:
        try:
            self._process_frames()
        except Exception as e:  # noqa: BLE001 — any failure must end the run visibly, not freeze it
            self.loop.call_soon_threadsafe(self._failed, "pipeline_error", f"Processing failed: {e}")

    def _process_frames(self) -> None:
        last = time.perf_counter()
        fps = 0.0
        while not self._stopping:
            frame = self.reader.get(timeout=0.5)
            if frame is None:
                if self.reader.exhausted:
                    if self.reader.error is not None:
                        self.loop.call_soon_threadsafe(self._failed, "source_lost",
                                                       f"Video source failed: {self.reader.error}")
                    else:
                        self.loop.call_soon_threadsafe(self._source_ended)
                    return
                continue
            start = time.perf_counter()
            detections = self.detector.detect(frame)
            det_ms = (time.perf_counter() - start) * 1000
            scene, events = self.builder.update(frame.t, detections)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6)) if fps else 1 / max(now - last, 1e-6)
            last = now
            if self.core.latencies_ms:
                self._jev_ms = self.core.latencies_ms[-1]
            image = draw_overlay(frame.image, scene, detections, self.ctx.sop, OverlayStats(fps, det_ms, self._jev_ms))
            self.ctx.frames.put(encode_jpeg(image))
            if self._stopping:
                return
            self.loop.call_soon_threadsafe(self._on_scene, scene, events)

    def _on_scene(self, scene: SceneState, events: list[SceneEvent]) -> None:
        if self._stopping:
            return
        self.core.on_scene(scene, events)
        self.decider.observe(scene, events, self.core.tracker.statuses(), self.core.tracker.active_step_id())

    async def _poll(self) -> None:
        while not self._stopping:
            await self.decider.poll(self.core.last_t)
            await asyncio.sleep(POLL_INTERVAL)

    def _source_ended(self) -> None:
        if not self._stopping:
            asyncio.ensure_future(self.stop())

    def _failed(self, kind: str, message: str) -> None:
        if self._stopping:
            return
        self.core.emit([Alert(self.core.last_t, "warning", kind, message)])
        self._aborted = True
        asyncio.ensure_future(self.stop())

    def _source_lost_threadsafe(self) -> None:
        self.loop.call_soon_threadsafe(self.core.source_lost)

    async def stop(self) -> dict | None:
        async with self._stop_lock:
            if self.done.is_set():
                return self.report
            self._stopping = True
            await self.decider.aclose()  # no Jev answer can arrive after this
            self._poller.cancel()
            if self.reader is not None:
                await asyncio.to_thread(self.reader.close)
            await asyncio.to_thread(self._thread.join, 2.0)
            self.report = self.core.finish(check_incomplete=not self._aborted)
            self.done.set()
            return self.report

    def snapshot(self) -> dict:
        """Current steps and alerts, for a dashboard that opens mid-run."""
        return {"steps": [asdict(s) for s in self.core.tracker.steps()], "alerts": [asdict(a) for a in self.core.alerts]}

    def pause(self) -> None:
        getattr(self.source, "pause", lambda: None)()

    def resume(self) -> None:
        getattr(self.source, "resume", lambda: None)()

    def set_speed(self, speed: float) -> None:
        getattr(self.source, "set_speed", lambda s: None)(speed)
