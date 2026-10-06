"""Spike: Jev (hosted) vs Kev (local) on the same replays, scored against the dataset's ground-truth steps.

Replays the test runs in lockstep once per backend, records every answer, and scores `current_step` against the
step the operator was really doing at that moment. Kev must be running (docker-compose.kev.yml publishes it on
127.0.0.1:8009). Usage: python spikes/compare_deciders.py [--kev-url http://127.0.0.1:8009]
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from sopvision.config import list_sops
from sopvision.dataset import load_index
from sopvision.decide.jev_client import JevClient
from sopvision.evaluation.run_eval import TEST_RUNS
from sopvision.perception.detector import YoloDetector
from sopvision.pipeline import run_lockstep
from sopvision.settings import Settings
from sopvision.sources.file import FileSource

ACCEPT = 0.8  # the SOP's accept_confidence, Jev's threshold


class Recorder:
    """Wraps a JevClient and keeps every answer (and every failure) it returns."""

    def __init__(self, inner: JevClient):
        self.inner, self.calls, self.failures = inner, [], 0

    async def ask(self, state, qs):
        try:
            a = await self.inner.ask(state, qs)
        except Exception:
            self.failures += 1
            raise
        self.calls.append({"t": state["scene"]["t"], "active": qs.active_step_id, "answer": a.current_step,
                           "conf": a.current_confidence, "done": a.active_step_done, "anomaly": a.anomaly,
                           "ms": a.latency_ms})
        return a

    async def aclose(self):
        await self.inner.aclose()


def truth_at(segments, t: float) -> str | None:
    return next((s.step_id for s in segments if s.start <= t < s.end), None)


def score(calls: list[dict], segments) -> list[dict]:
    for c in calls:
        gt = truth_at(segments, c["t"])
        c["truth"] = gt
        c["correct"] = c["answer"] == gt or (gt == "off_sop" and c["answer"] in ("idle", "off_sop"))
    return calls


def summarize(name: str, calls: list[dict], failures: int, threshold: float, anomaly_th: float) -> dict:
    confident = [c for c in calls if c["conf"] >= threshold]
    ms = sorted(c["ms"] for c in calls)
    pct = lambda xs, n: f"{100 * sum(xs) / n:.0f}%" if n else "–"
    return {
        "backend": name,
        "calls": len(calls),
        "failed calls": failures,
        "step named correctly": pct([c["correct"] for c in calls], len(calls)),
        "step threshold": f"{threshold}",
        "acts on (≥ its threshold)": pct([c["conf"] >= threshold for c in calls], len(calls)),
        "correct when it acts": pct([c["correct"] for c in confident], len(confident)),
        "median confidence": f"{statistics.median(c['conf'] for c in calls):.2f}" if calls else "–",
        "anomaly threshold": f"{anomaly_th}",
        "false anomaly flags": pct([c["anomaly"] >= anomaly_th for c in calls], len(calls)),
        "latency p50 / p90 / max": (f"{ms[len(ms) // 2]:.0f} / {ms[int(0.9 * (len(ms) - 1))]:.0f} / {ms[-1]:.0f} ms"
                                    if ms else "–"),
    }


async def replay(sop, settings, detector, client, run) -> Recorder:
    rec = Recorder(client)
    detector.reset()
    await run_lockstep(sop, FileSource(run.raw_path, mode="lockstep"), detector, rec, run_id="compare",
                       source_id=f"test:Install_{run.run}")
    return rec


async def main(kev_url: str, out: Path, kev_threshold: float, kev_anomaly: float, kev_done: float) -> None:
    settings = Settings()
    sop = list_sops(settings.sops_dir)[0]
    index = {r.run: r for r in load_index(settings.sop_data_root)}
    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    backends = {  # name -> (client factory, step threshold, anomaly threshold)
        "Jev (hosted)": (lambda: JevClient(settings.typesafe_api_key.get_secret_value(), settings.typesafe_base_url,
                                           settings.typesafe_model), ACCEPT, ACCEPT),
        "Kev-4B (local)": (lambda: JevClient("kev-local", kev_url, "kev-4b"), kev_threshold, kev_anomaly),
    }
    # Each replay's tracker uses that model's thresholds, so its activations feed back as they would live.
    kev_sop = sop.model_copy(update={"decisions": sop.decisions.model_copy(update={
        "step_confidence": kev_threshold, "anomaly_confidence": kev_anomaly, "done_confidence": kev_done})})
    sops = {"Jev (hosted)": sop, "Kev-4B (local)": kev_sop}
    results, raw = [], {}
    for name, (make, threshold, anomaly_th) in backends.items():
        calls, failures = [], 0
        for n in TEST_RUNS:
            print(f"{name}: replaying Install_{n} …", flush=True)
            client = make()
            try:
                rec = await replay(sops[name], settings, detector, client, index[n])
            finally:
                await client.aclose()
            calls += [c | {"run": n} for c in score(rec.calls, index[n].segments)]
            failures += rec.failures
        raw[name] = calls
        results.append(summarize(name, calls, failures, threshold, anomaly_th))
    out.write_text(json.dumps({"summary": results, "calls": raw}, indent=1), encoding="utf-8")
    width = max(len(k) for k in results[0])
    print()
    print(f"{'':{width}}  " + "  ".join(f"{r['backend']:>18}" for r in results))
    for key in list(results[0])[1:]:
        print(f"{key:{width}}  " + "  ".join(f"{str(r[key]):>18}" for r in results))
    print(f"\nraw answers: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kev-url", default="http://127.0.0.1:8009")
    ap.add_argument("--out", type=Path, default=Path("runs/compare-deciders.json"))
    ap.add_argument("--kev-step-threshold", type=float, default=ACCEPT, help="Kev's tuned step threshold")
    ap.add_argument("--kev-anomaly-threshold", type=float, default=ACCEPT, help="Kev's tuned anomaly threshold")
    ap.add_argument("--kev-done-threshold", type=float, default=ACCEPT, help="Kev's tuned done threshold")
    args = ap.parse_args()
    asyncio.run(main(args.kev_url, args.out, args.kev_step_threshold, args.kev_anomaly_threshold, args.kev_done_threshold))
