"""Replay an SOP's recordings end to end (detector, tracker, torque log, the configured decision model) and report
what an operator would have seen: steps done, alerts, and how often the model's current step matched the timeline.

    docker compose run --rm --entrypoint python sopvision scripts/replay_recordings.py --sop sops/lug_bolts.yaml \
        [--split test] [--recording dev4-idle-offsop] [--out runs/replay-lug_bolts-test.json]

Lockstep, as `sopvision eval`: every request is answered before the video moves on, and the answer lands 0.5 s later
in video time, so results don't depend on the machine. The model is DECISION_MODE's (jev, decider or laya).
Current-step accuracy uses the recording's steps.csv (sopvision.evaluation.decisions.expected_steps: idle and
off-SOP stretches accept either answer; the first second of each step is not scored).
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from sopvision.config import load_sop
from sopvision.decide.providers import clients_for
from sopvision.evaluation.decisions import expected_steps
from sopvision.perception.detector import YoloDetector
from sopvision.pipeline import run_lockstep
from sopvision.recordings import load_recordings
from sopvision.settings import Settings
from sopvision.sop.torque import read_torque_log
from sopvision.sources.file import FileSource


async def replay(settings: Settings, sop, recordings, laya_model: str | None = None) -> list[dict]:
    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    client, _ = clients_for(settings, sop.sop, laya_model)
    out = []
    try:
        for rec in recordings:
            detector.reset()
            torque = read_torque_log(rec.torque_log) if rec.torque_log else None
            res = await run_lockstep(sop, FileSource(rec.video, mode="lockstep"), detector, client,
                                     run_id=f"replay-{rec.id}", source_id=rec.source_id, torque=torque)
            scored = [(t, a, expected_steps(rec.segments, t)) for t, a in res.decisions]
            scored = [(t, a, e) for t, a, e in scored if e]
            misses = Counter(f"{'idle' if 'idle' in e else e[0]} -> {a}" for _, a, e in scored if a not in e)
            idle = [(t, a, e) for t, a, e in scored if "idle" in e]
            out.append({
                "run": rec.id, "split": rec.split,
                "done": [s.id for s in sorted((s for s in res.steps if s.status == "done"), key=lambda s: s.done_t)],
                "not_done": [s.id for s in res.steps if s.status != "done"],
                "alerts": [f"{a.t:.1f}s {a.level} {a.kind}: {a.message}" for a in res.alerts
                           if a.level in ("warning", "violation")],
                "requests": len(res.decisions),
                "current_step": round(sum(a in e for _, a, e in scored) / len(scored), 4) if scored else None,
                "idle_or_off_sop": f"{sum(a in e for _, a, e in idle)}/{len(idle)}",
                "top_misses": misses.most_common(5),
            })
            r = out[-1]
            print(f"\n== {rec.split}:{rec.id}  requests {r['requests']}, current step {r['current_step']}, "
                  f"idle/off-SOP {r['idle_or_off_sop']}", flush=True)
            print(f"   done: {' '.join(r['done'])}" + (f"   NOT DONE: {' '.join(r['not_done'])}" if r["not_done"] else ""))
            for a in r["alerts"] or ["no alerts"]:
                print(f"   {a}")
            if r["top_misses"]:
                print(f"   misses: {r['top_misses']}")
    finally:
        await client.aclose()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sop", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--recording", action="append", default=[], help="only these recording ids (repeatable)")
    ap.add_argument("--out", type=Path, default=None, help="also write the results as JSON")
    ap.add_argument("--laya-model", default=None, help="the Laya model to ask (default: the SOP's checkpoint in the "
                                                       "registry, else LAYA_MODEL)")
    args = ap.parse_args()
    settings, sop = Settings(), load_sop(args.sop)
    recs = [r for r in load_recordings(sop, settings) if r.valid and (args.split == "all" or r.split == args.split)
            and (not args.recording or r.id in args.recording)]
    print(f"{sop.sop}: {len(recs)} recordings, decisions from {settings.decision_mode}", flush=True)
    results = asyncio.run(replay(settings, sop, recs, args.laya_model))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"sop": sop.sop, "decision_mode": settings.decision_mode, "runs": results},
                                       indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
