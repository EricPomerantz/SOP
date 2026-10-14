import asyncio
import time

import numpy as np

from sopvision.decide.types import JevAnswers
from sopvision.perception.detector import Detection
from sopvision.pipeline import LiveRun, RunContext, run_lockstep
from sopvision.server.frames import FrameBuffer
from sopvision.sources.base import Frame
from sopvision.sources.registry import SourceSpec
from sopvision.storage.runlog import RunLog, read_events
from tests.helpers import CHASSIS, det_at


class ScriptedSource:
    """Frames at `fps` for `seconds`; paced like a camera when `paced`."""

    def __init__(self, seconds: float, fps: float = 30.0, paced: bool = False):
        self.fps, self.n, self.paced = fps, int(seconds * fps), paced
        self.closed = False

    def frames(self):
        for i in range(self.n):
            if self.closed:
                return
            if self.paced:
                time.sleep(1 / self.fps)
            yield Frame(i / self.fps, i, np.zeros((48, 64, 3), np.uint8))

    def close(self):
        self.closed = True


class ScriptedDetector:
    """Chassis always; step k's part appears installed from 2 + 3.5k seconds."""

    def __init__(self, sop, hands: bool = False, delay: float = 0.0):
        self.sop, self.hands, self.delay = sop, hands, delay
        self.seen: list[float] = []

    def detect(self, frame):
        self.seen.append(frame.t)
        if self.delay:
            time.sleep(self.delay)
        dets = [Detection("chassis", CHASSIS, 0.95, None)]
        for k, step in enumerate(self.sop.steps):
            (slot,) = step.evidence
            spec = self.sop.layout.slots[slot]
            if frame.t >= 2 + 3.5 * k:
                dets.append(det_at(spec.filled_by, *spec.at, size=300 if spec.filled_by == "cover" else 40))
            elif spec.empty_as:
                dets.append(det_at(spec.empty_as, *spec.at))
        if self.hands:
            dets.append(det_at("hand", 0.9, 0.9, size=40))
        return dets

    def reset(self):
        pass


class OracleClient:
    """Answers like a perfect Jev would, from the scene it is shown."""

    def __init__(self, sop, delay: float = 0.0):
        self.sop, self.delay = sop, delay

    async def ask(self, state, qs):
        if self.delay:
            await asyncio.sleep(self.delay)
        slots = state["scene"]["slots"]

        def installed(step_id):
            (slot,) = self.sop.step(step_id).evidence
            return slots.get(slot, {}).get("state") == "installed"

        current = next((s.id for s in self.sop.steps if not installed(s.id)), "idle")
        done = 0.95 if installed(qs.active_step_id) else 0.05
        return JevAnswers(current, 0.95, {current: 0.95}, done, 0.0, 50.0)


async def test_lockstep_completes_scripted_run(sop, tmp_path):
    res = await run_lockstep(sop, ScriptedSource(40), ScriptedDetector(sop), OracleClient(sop), RunLog(tmp_path / "r.jsonl"))
    assert [s.status for s in res.steps] == ["done"] * 9
    assert not [a for a in res.alerts if a.level == "violation"]
    assert len(res.frames) == 40 * 30 and res.frames[-1].slots["cover"] == "installed"
    types = [e.type for e in read_events(tmp_path / "r.jsonl")]
    assert types[0] == "run_started" and "step_update" in types and "decision" in types


async def test_lockstep_reports_incomplete_run(sop):
    res = await run_lockstep(sop, ScriptedSource(12), ScriptedDetector(sop), OracleClient(sop))
    (incomplete,) = [a for a in res.alerts if a.kind == "sop_incomplete"]
    assert "cover" in incomplete.message


def live_run(sop, tmp_path, source, detector, client):
    messages = []
    ctx = RunContext("run1", sop, SourceSpec(id="test:x", kind="file", label="x"), RunLog(tmp_path / "run.jsonl"),
                     messages.append, FrameBuffer())
    run = LiveRun(ctx, source, detector, client)
    run.messages = messages
    return run


async def test_live_drops_frames_under_slow_detector(sop, tmp_path):
    detector = ScriptedDetector(sop, delay=0.05)
    run = live_run(sop, tmp_path, ScriptedSource(2, paced=True), detector, OracleClient(sop))
    await run.start()
    await asyncio.wait_for(run.done.wait(), 10)
    assert run.reader.dropped > 0 and len(detector.seen) < 60
    assert detector.seen == sorted(set(detector.seen))
    assert run.ctx.frames.latest()[0] > 0  # annotated frames reached the MJPEG buffer
    assert any(m["type"] == "run_summary" for m in run.messages)


async def test_stop_during_inflight_call_writes_nothing_after_summary(sop, tmp_path):  # Review Focus 4
    run = live_run(sop, tmp_path, ScriptedSource(10, paced=True), ScriptedDetector(sop, hands=True),
                   OracleClient(sop, delay=1.0))
    await run.start()
    await asyncio.sleep(0.3)
    await run.stop()
    await run.stop()  # idempotent
    await asyncio.sleep(1.2)  # the cancelled Jev call would have returned by now
    types = [e.type for e in read_events(tmp_path / "run.jsonl")]
    assert types[-1] == "run_summary" and types.count("run_summary") == 1


class BrokenSource:
    fps = 30.0

    def frames(self):
        from sopvision.sources.base import SourceUnavailableError
        raise SourceUnavailableError("Can't open webcam 0. Is it plugged in and not used by another app?")
        yield  # pragma: no cover

    def close(self):
        pass


class FailingDetector(ScriptedDetector):
    def detect(self, frame):
        if frame.index >= 3:
            raise RuntimeError("CUDA error: out of memory")
        return super().detect(frame)


async def test_source_that_fails_to_open_reports_its_error_not_an_incomplete_sop(sop, tmp_path):
    run = live_run(sop, tmp_path, BrokenSource(), ScriptedDetector(sop), OracleClient(sop))
    await run.start()
    await asyncio.wait_for(run.done.wait(), 5)
    alerts = [m for m in run.messages if m["type"] == "alert"]
    assert [a["kind"] for a in alerts] == ["source_lost"]
    assert alerts[0]["message"] == "Video source failed: Can't open webcam 0. Is it plugged in and not used by another app?"


async def test_processing_error_ends_the_run_with_an_alert(sop, tmp_path):
    run = live_run(sop, tmp_path, ScriptedSource(10, paced=True), FailingDetector(sop), OracleClient(sop))
    await run.start()
    await asyncio.wait_for(run.done.wait(), 5)
    alerts = [m for m in run.messages if m["type"] == "alert"]
    assert [a["kind"] for a in alerts] == ["pipeline_error"]
    assert alerts[0]["message"] == "Processing failed: CUDA error: out of memory"
    assert read_events(tmp_path / "run.jsonl")[-1].type == "run_summary"



def test_decisions_and_offline_alerts_name_the_deciding_model(sop):
    from sopvision.decide.types import Decision
    from sopvision.pipeline import RunCore
    from sopvision.scene.types import SceneState

    sent = []
    core = RunCore(sop, "r", "file", None, sent.append, model="Laya")
    core.last_scene = SceneState(1, 1.0, True, 0.9, {}, [], [], None)
    core.on_decision(Decision(1, 1.0, "fan_1", JevAnswers("fan_1", 0.9, {"fan_1": 0.9}, 0.1, 0.0, 80.0, "laya")))
    (msg,) = [m for m in sent if m["type"] == "decision"]
    assert msg["provider"] == "laya"
    core.on_status("offline", 2.0)
    assert any(m["type"] == "alert" and m["message"] == "Laya offline — using evidence only" for m in sent)


# -- lug bolts: vision + torque results, end to end -------------------------------------------------------------------
STAR = ["hole_1", "hole_3", "hole_5", "hole_2", "hole_4"]


class LugDetector:
    """Wheel always; bolt on hole k from t = 1 + k; the wrench at `wrench[i][1]` during [T - 1, T + 0.5] around each
    torque time T. `missing` holes never get a bolt."""

    def __init__(self, sop, wrench: list[tuple[float, str]], missing=()):
        self.sop, self.wrench, self.missing = sop, wrench, set(missing)

    def detect(self, frame):
        dets = [Detection("hub", CHASSIS, 0.95, None)]
        for k in range(1, 6):
            slot = f"hole_{k}"
            spec = self.sop.layout.slots[slot]
            cls = "lug_bolt_gold" if frame.t >= 1 + k and slot not in self.missing else "hole_empty"
            dets.append(det_at(cls, *spec.at, size=24))
        for t, slot in self.wrench:
            if t - 1 <= frame.t <= t + 0.5:
                dets.append(det_at("torque_wrench", *self.sop.layout.slots[slot].at, size=20))
        return dets

    def reset(self):
        pass


class NeverActs:
    provider = "stub"

    async def ask(self, state, qs):
        return JevAnswers("idle", 0.0, {"idle": 1.0}, 0.0, 0.0, 1.0, "stub")


async def lug_run(lug_sop, order=STAR, missing=(), torque=None, wrench=None):
    from sopvision.sop.torque import TorqueResult

    times = [10.0 + 3 * i for i in range(len(order))]
    wrench = wrench if wrench is not None else list(zip(times, order))
    torque = torque if torque is not None else [TorqueResult(t, True, 110.0) for t in times]
    return await run_lockstep(lug_sop, ScriptedSource(26), LugDetector(lug_sop, wrench, missing), NeverActs(),
                              torque=torque)


def kinds(res):
    return [a.kind for a in res.alerts]


async def test_lug_bolts_correct_run_completes_every_step_in_order(lug_sop):
    res = await lug_run(lug_sop)
    assert [s.status for s in res.steps] == ["done"] * 10
    assert {s.id: s.source for s in res.steps if s.id.startswith("torque")} == {f"torque_{k}": "torque" for k in range(1, 6)}
    assert not [a for a in res.alerts if a.level in ("warning", "violation")]
    assert kinds(res).count("torque") == 5


async def test_lug_bolts_out_of_sequence_torque_is_a_violation(lug_sop):
    res = await lug_run(lug_sop, order=["hole_1", "hole_2", "hole_3", "hole_4", "hole_5"])  # clockwise, not star
    violations = [a.message for a in res.alerts if a.kind == "order_violation"]
    assert "torque_2 completed before torque_3, torque_5" in violations


async def test_lug_bolts_torque_on_a_hole_with_no_nut(lug_sop):
    res = await lug_run(lug_sop, missing={"hole_5"})
    assert "torque_without_part" in kinds(res)
    assert next(s for s in res.steps if s.id == "torque_5").status != "done"


async def test_lug_bolts_torque_not_reached_and_unattributed(lug_sop):
    from sopvision.sop.torque import TorqueResult

    times = [10.0 + 3 * i for i in range(5)]
    torque = [TorqueResult(times[0], False, 60.0)] + [TorqueResult(t, True, 110.0) for t in times[1:]]
    res = await lug_run(lug_sop, torque=torque, wrench=list(zip(times, STAR))[:4])  # wrench never seen for the last
    assert "torque_not_reached" in kinds(res) and "torque_unattributed" in kinds(res)


async def test_lug_bolts_torque_results_are_in_the_run_log(lug_sop, tmp_path):
    from sopvision.sop.torque import TorqueResult

    det = LugDetector(lug_sop, [(10.0, "hole_1")])
    await run_lockstep(lug_sop, ScriptedSource(12), det, NeverActs(), RunLog(tmp_path / "r.jsonl"),
                       torque=[TorqueResult(10.0, True, 110.0)])
    (ev,) = [e for e in read_events(tmp_path / "r.jsonl") if e.type == "torque"]
    assert ev.payload["slot"] == "hole_1" and ev.payload["accepted"] is True


def test_live_torque_is_timed_at_the_latest_frame_plus_the_camera_delay(lug_sop, tmp_path):
    ctx = RunContext("run1", lug_sop, SourceSpec(id="rtsp:0", kind="rtsp", label="cam"), None, lambda m: None, FrameBuffer())
    run = LiveRun(ctx, ScriptedSource(1), LugDetector(lug_sop, []), NeverActs(), camera_delay=0.4)
    run.core.last_t = 12.0
    assert run.torque(True, 110.0) == 12.4
    assert run.core.torque._pending[0].t == 12.4


def test_a_replayed_recording_brings_its_torque_log(lug_sop, tmp_path):
    log = tmp_path / "torque.csv"
    log.write_text("time_s,ok,torque_nm\n12.5,1,110\n20.0,0,\n")
    spec = SourceSpec(id="recording:lug_bolts:r1", kind="file", label="r1", path=tmp_path / "v.mp4", torque_log=log)
    ctx = RunContext("run1", lug_sop, spec, None, lambda m: None, FrameBuffer())
    run = LiveRun(ctx, ScriptedSource(1), LugDetector(lug_sop, []), NeverActs())
    assert [(r.t, r.ok) for r in run.core.torque._pending] == [(12.5, True), (20.0, False)]
