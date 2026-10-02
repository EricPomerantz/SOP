"""Lockstep replay of the test runs and scenario videos, scored against the acceptance criteria (spec §12)."""

import asyncio
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from sopvision.config import SopConfig
from sopvision.dataset import RunIndex, load_index
from sopvision.evaluation.metrics import EvalReport, EvalVideo, check_acceptance, offsop_clean, step_errors
from sopvision.pipeline import LockstepResult, run_lockstep
from sopvision.settings import Settings
from sopvision.sources.file import FileSource

TEST_RUNS = (12, 13)


def latency_stats(ms: list[float]) -> dict[str, float]:
    if not ms:
        return {"p50": 0.0, "p90": 0.0, "max": 0.0}
    s = sorted(ms)
    return {"p50": s[len(s) // 2], "p90": s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)], "max": s[-1]}


def video_from_result(vid: str, kind: Literal["test", "scenario"], res: LockstepResult, run: RunIndex | None,
                      expected: list[dict]) -> EvalVideo:
    errors = {k: None if e is None else round(e, 3) for k, e in (step_errors(run.segments, res.steps) if run else {}).items()}
    steps = {s.id: {"status": s.status, "done_t": s.done_t, "error": errors.get(s.id), "source": s.source}
             for s in res.steps}
    order = [s.id for s in sorted((s for s in res.steps if s.status == "done"), key=lambda s: s.done_t)]
    return EvalVideo(
        id=vid, kind=kind, step_order=order, steps=steps, alerts=[asdict(a) for a in res.alerts],
        frames=[{"t": round(f.t, 3), "slots": f.slots} for f in res.frames],
        offsop=offsop_clean(run.segments, res.steps, res.decisions) if run else [],
        latency_ms=latency_stats(res.latencies_ms), expected_alerts=expected)


async def _run_all(settings: Settings, sop: SopConfig, detector, client, progress) -> list[EvalVideo]:
    index = {r.run: r for r in load_index(settings.sop_data_root)}
    jobs: list[tuple[str, Literal["test", "scenario"], Path, RunIndex | None, list[dict]]] = [
        (f"test:Install_{n}", "test", index[n].raw_path, index[n], []) for n in TEST_RUNS]
    for mp4 in sorted(settings.scenarios_dir.glob("*.mp4")):
        sidecar = json.loads(mp4.with_suffix(".json").read_text(encoding="utf-8"))
        jobs.append((f"scenario:{mp4.stem}", "scenario", mp4, None, sidecar["expected_alerts"]))
    videos = []
    for vid, kind, path, run, expected in jobs:
        progress(f"evaluating {vid} …")
        detector.reset()
        res = await run_lockstep(sop, FileSource(path, mode="lockstep"), detector, client, run_id="eval", source_id=vid)
        videos.append(video_from_result(vid, kind, res, run, expected))
    return videos


def run_eval(settings: Settings, sop: SopConfig, out: Path, env: dict, progress=print) -> EvalReport:
    from sopvision.decide.jev_client import JevClient
    from sopvision.perception.detector import YoloDetector

    detector = YoloDetector(sop.detector.weights, sop.detector.classes)

    async def main():
        client = JevClient(settings.typesafe_api_key.get_secret_value(), settings.typesafe_base_url, settings.typesafe_model)
        try:
            return await _run_all(settings, sop, detector, client, progress)
        finally:
            await client.aclose()

    videos = asyncio.run(main())
    report = EvalReport(env=env, videos=videos, acceptance={}, passed=False)
    acceptance = check_acceptance(report)
    report = report.model_copy(update={"acceptance": acceptance, "passed": all(acceptance.values())})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(), encoding="utf-8")
    return report


def summary_lines(report: EvalReport) -> list[str]:
    tests = [v for v in report.videos if v.kind == "test"]
    errors = [abs(s["error"]) for v in tests for s in v.steps.values() if s["error"] is not None]
    lines = [f"{v.id}: {sum(s['status'] == 'done' for s in v.steps.values())}/{len(v.steps)} steps, "
             f"alerts {sorted({a['kind'] for a in v.alerts})}, Jev p50 {v.latency_ms['p50']:.0f} ms" for v in report.videos]
    if errors:
        lines.append(f"step timing error: median {sorted(errors)[len(errors) // 2]:.2f} s, max {max(errors):.2f} s")
    names = {"c1": "all steps done in order", "c2": "timing within 2 s median / 4 s max", "c3": "off-SOP segments: idle/off-SOP, no step progress",
             "c4": "no violations on clean runs", "c5": "scenarios raise expected alerts"}
    lines += [f"{'PASS' if ok else 'FAIL'}  {k}: {names[k]}" for k, ok in report.acceptance.items()]
    lines.append("passed" if report.passed else "FAILED")
    return lines
