"""Score a decision provider on labelled training-format rows (e.g. the validation split), per row kind.

    docker compose run --rm --entrypoint python sopvision scripts/eval_training_rows.py data/training/sop-v1/val.jsonl \
        --provider laya

For model selection only: the frozen benchmark (scripts/bench_decisions.py --frozen) remains the test. Requests go
through the same client the app uses; `anomaly` and `active_step_done` count as yes at the tracker's 0.8 threshold.
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from sopvision.config import list_sops
from sopvision.decide.jev_client import JevUnavailableError
from sopvision.decide.providers import make_client
from sopvision.decide.questions import QuestionSet
from sopvision.settings import Settings


def question_set(wire: dict) -> QuestionSet:
    return QuestionSet(wire["current_step"]["criteria"], "", wire["active_step_done"]["instructions"])


async def run(rows: list[dict], provider: str, accept: float) -> dict:
    client = make_client(Settings(), provider)
    hits: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    ms, errors = [], 0
    try:
        for r in rows:
            t = time.perf_counter()
            try:
                a = await client.ask(r["state"], question_set(r["questions"]))
            except JevUnavailableError:
                errors += 1
                continue
            ms.append((time.perf_counter() - t) * 1000)
            exp, kind = r["expected"], r["kind"]
            for group in (kind, "all"):
                if "current_step" in exp:
                    hits[group]["current_step"].append(a.current_step == exp["current_step"])
                    if a.current_confidence >= accept:
                        hits[group]["current_step_when_acting"].append(a.current_step == exp["current_step"])
                if "active_step_done" in exp:
                    hits[group]["active_step_done"].append((a.active_step_done >= accept) == exp["active_step_done"])
                if "anomaly" in exp:
                    hits[group]["anomaly"].append((a.anomaly >= accept) == exp["anomaly"])
    finally:
        await client.aclose()
    s = sorted(ms)
    return {"provider": provider, "rows": len(rows), "errors": errors,
            "latency_ms": {"p50": round(s[len(s) // 2], 1), "p95": round(s[int(0.95 * len(s))], 1)} if s else None,
            "accuracy": {g: {q: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for q, v in qs.items() if v}
                         for g, qs in sorted(hits.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path)
    ap.add_argument("--provider", default="laya")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    accept = list_sops(Settings().sops_dir)[0].decisions.accept_confidence
    result = asyncio.run(run(rows, args.provider, accept))
    for group, qs in result["accuracy"].items():
        print(f"{group:34} " + "  ".join(f"{q} {v['acc']:.1%} (n={v['n']})" for q, v in qs.items()))
    print(f"errors {result['errors']}, latency {result['latency_ms']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
