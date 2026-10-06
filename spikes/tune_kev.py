"""Spike: tune Kev's thresholds on the training recordings (never seen by Jev or Kev; test runs stay held out).

Sends every recorded request (spikes/dump_scenes.py) to Kev, labels each answer against ground truth, and picks:
  step:    lowest threshold where Kev's current_step is right >= 90% of the calls it would act on
  done:    lowest threshold where "active step done" is right >= 90% (right = that step's true segment has ended)
  anomaly: lowest threshold with no false alarm on these clean runs (they contain no real anomalies)
Usage: python spikes/tune_kev.py [--kev-url http://kev:8009] [--runs 1 3 4 ...]
"""

import argparse
import asyncio
import json
from pathlib import Path

from sopvision.config import list_sops
from sopvision.dataset import load_index
from sopvision.decide.jev_client import JevClient
from sopvision.decide.questions import build_questions
from sopvision.settings import Settings

TARGET = 0.90
MIN_CALLS = 30  # a threshold has to leave at least this many calls to judge it by
GRID = [round(0.01 * i, 2) for i in range(1, 100)]


def pick(calls, score, right) -> float | None:
    for th in GRID:
        acted = [c for c in calls if c[score] >= th]
        if len(acted) >= MIN_CALLS and sum(c[right] for c in acted) / len(acted) >= TARGET:
            return th
    return None


def table(calls, score, right, ths) -> str:
    rows = []
    for th in ths:
        acted = [c for c in calls if c[score] >= th]
        rows.append(f"  >= {th:<5} acts on {100 * len(acted) / len(calls):3.0f}%, right "
                    + (f"{100 * sum(c[right] for c in acted) / len(acted):3.0f}%" if acted else "  –"))
    return "\n".join(rows)


async def main(kev_url: str, runs: list[int]) -> None:
    settings = Settings()
    sop = list_sops(settings.sops_dir)[0]
    index = {r.run: r for r in load_index(settings.sop_data_root)}
    client = JevClient("kev-local", kev_url, "kev-4b")
    sem = asyncio.Semaphore(4)

    async def one(run_no, r):
        segs = index[run_no].segments
        truth = next((s.step_id for s in segs if s.start <= r["t"] < s.end), None)
        if truth is None:
            return None
        active_end = next((s.end for s in segs if s.step_id == r["active"]), None)
        async with sem:
            a = await client.ask(r["state"], build_questions(sop, r["active"]))
        return {"run": run_no, "t": r["t"], "truth": truth, "answer": a.current_step, "conf": a.current_confidence,
                "step_right": a.current_step == truth or (truth == "off_sop" and a.current_step in ("idle", "off_sop")),
                "done": a.active_step_done, "done_right": active_end is not None and r["t"] >= active_end,
                "anomaly": a.anomaly, "false_off_sop": a.current_step == "off_sop" and truth != "off_sop"}

    try:
        jobs = [one(n, r) for n in runs
                for r in json.loads(Path(f"runs/scenes-Install_{n}.json").read_text(encoding="utf-8"))]
        calls = [c for c in await asyncio.gather(*jobs) if c]
    finally:
        await client.aclose()

    step, done = pick(calls, "conf", "step_right"), pick(calls, "done", "done_right")
    anomaly = round(min(0.99, max(c["anomaly"] for c in calls) + 0.01), 2)
    print(f"{len(calls)} requests from {len(runs)} training recordings\n")
    print("current step confidence:\n" + table(calls, "conf", "step_right", (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)))
    print("\nactive step done:\n" + table(calls, "done", "done_right", (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)))
    print(f"\nanomaly on clean runs: max {max(c['anomaly'] for c in calls):.2f}")
    if step is not None:
        fo = [c for c in calls if c["false_off_sop"] and c["conf"] >= step]
        print(f"off_sop answers at >= {step} while a step was really under way: {len(fo)} of {len(calls)} calls")
    print(f"\nchosen: step {step}, done {done}, anomaly {anomaly}")
    Path("runs/kev-tuning.json").write_text(json.dumps({"thresholds": {"step": step, "done": done, "anomaly": anomaly},
                                                        "calls": calls}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kev-url", default="http://127.0.0.1:8009")
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    a = ap.parse_args()
    asyncio.run(main(a.kev_url, a.runs))
