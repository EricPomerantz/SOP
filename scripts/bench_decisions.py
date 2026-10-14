"""Replay recorded SOP decision requests through Jev and the local Decider, without the camera pipeline.

Input: run logs holding `decision_comparison` records (each carries the exact state + questions sent), written by
any run with DECISION_MODE=compare — live (`runs/run-*.jsonl`) or `sopvision eval` (`runs/<report>.<video>.jsonl`).

    docker compose run --rm --entrypoint python sopvision scripts/bench_decisions.py runs/eval-compare.*.jsonl
    ... --no-replay      score the answers already in the logs instead of calling the models again
    ... --limit 200      first N requests per log

Experiments on a frozen set (scripts/freeze_benchmark.py; labels come from the set and are never recomputed):

    ... scripts/bench_decisions.py --frozen benchmarks/sop-244 --variant a1_progress
    variants (sopvision.evaluation.variants): baseline, a1_progress, ...; output in runs/experiments/<variant>-<time>

Each request goes to both providers at once; requests run one after another so neither sees contention from the
benchmark itself. Ground truth comes from the dataset's step segments for unedited test recordings
(sopvision.evaluation.decisions); everything else stays unlabelled. Writes <out>.json (summary) and
<out>.jsonl (one record per request, ready for labels added later).
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from sopvision.config import SopConfig, list_sops, load_sop
from sopvision.decide.compare import comparison_record, provider_entry
from sopvision.decide.jev_client import JevUnavailableError
from sopvision.decide.providers import make_client
from sopvision.decide.questions import QuestionSet
from sopvision.decide.types import ProviderResult
from sopvision.evaluation.decisions import label, summarize, summary_text
from sopvision.evaluation.variants import VARIANTS, expected_steps
from sopvision.recordings import load_recordings
from sopvision.settings import Settings
from sopvision.storage.runlog import read_events


async def ask(client, state: dict, qs: QuestionSet) -> ProviderResult:
    start = time.perf_counter()
    try:
        answers = await client.ask(state, qs)
    except (JevUnavailableError, Exception) as e:  # noqa: BLE001 — every failure is a data point here
        return ProviderResult(client.provider, None, f"{type(e).__name__}: {e}", (time.perf_counter() - start) * 1000)
    return ProviderResult(client.provider, answers, None, (time.perf_counter() - start) * 1000)


def load(paths: list[Path], limit: int | None) -> list[tuple[str, dict]]:
    """(source id, record) for every decision_comparison in the logs."""
    out = []
    for path in paths:
        events = read_events(path)
        source = next((e.payload["source"] for e in events if e.type == "run_started"), path.stem)
        records = [e.payload for e in events if e.type == "decision_comparison"]
        out += [(source, r) for r in records[:limit]]
        print(f"{path}: {len(records[:limit])} requests from {source}", flush=True)
    return out


def labeller(settings: Settings, sop: SopConfig):
    """Ground truth from the SOP's recordings (sopvision.recordings), for runs named '<split>:<recording id>'."""
    try:
        recordings = {r.source_id: r for r in load_recordings(sop, settings) if r.valid}
    except Exception as e:  # noqa: BLE001 — no dataset: everything stays unlabelled
        print(f"no ground truth (dataset not readable: {e})", flush=True)
        recordings = {}

    def apply(source: str, record: dict) -> dict:
        # Only unedited recordings carry a timeline; scenario videos are edited, so theirs is unknown.
        rec = recordings.get(source)
        # A run with a deviation in it (off-SOP, repeated or out-of-order steps) leaves anomaly unknown.
        return label(record, rec.segments if rec else None, rec is not None and rec.clean, source)

    return apply


def load_frozen(folder: Path, limit: int | None) -> list[tuple[str, dict]]:
    lines = (folder / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()][:limit]
    print(f"{folder}: {len(records)} frozen requests", flush=True)
    return [(r["source"], r) for r in records]


def single_record(r: dict, state: dict, qs: QuestionSet, result: ProviderResult) -> dict:
    """The comparison record's shape for a run with one provider (no comparison block)."""
    return {"request_id": r["request_id"], "t_request": r["t_request"], "version": r["version"],
            "active_step_id": qs.active_step_id, "request": {"state": state, "questions": asdict(qs)},
            "authoritative": result.provider, result.provider: provider_entry(result), "comparison": None,
            "ground_truth": None}


async def replay(items: list[tuple[str, dict]], settings: Settings, accept: float, variant,
                 providers: list[str]) -> list[tuple[str, dict]]:
    clients = [make_client(settings, p) for p in providers]
    out = []
    try:
        for i, (source, r) in enumerate(items):
            state, qs = variant(r)  # identical request to every provider
            results = await asyncio.gather(*(ask(c, state, qs) for c in clients))
            rec = (comparison_record(r["request_id"], r["t_request"], r["version"], state, qs, *results, accept)
                   if len(results) == 2 else single_record(r, state, qs, results[0]))
            frozen = {"ground_truth": r["ground_truth"], "probe": r.get("probe")} if "tracker" in r else {}
            out.append((source, rec | frozen))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)
    finally:
        for c in clients:
            await c.aclose()
    return out


def decider_info(settings: Settings) -> dict | None:
    """Which model and temperatures answered, for reproducibility."""
    import httpx

    try:
        return httpx.get(f"{settings.decider_url}/health", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        return None


def laya_info(settings: Settings) -> dict:
    """Laya checkpoint, budgets and what the server reports, for reproducibility."""
    import httpx

    info = {"model": settings.laya_model, "max_len": settings.laya_max_len, "head_max_len": settings.laya_head_max_len}
    try:
        info["health"] = httpx.get(f"{settings.laya_url}/health", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        info["health"] = None
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--frozen", type=Path, help="a frozen benchmark folder instead of run logs")
    ap.add_argument("--variant", default="baseline", choices=sorted(VARIANTS), help="request variant (frozen sets)")
    ap.add_argument("--providers", default="jev,decider", help="one or two of jev, decider, laya (the first is reported "
                                                               "as authoritative)")
    ap.add_argument("--no-replay", action="store_true", help="score the recorded answers, call no model")
    ap.add_argument("--limit", type=int, default=None, help="first N requests per log")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sop", type=Path, default=None, help="SOP file (default: the first in sops/)")
    args = ap.parse_args()
    stamp = time.strftime('%Y%m%d-%H%M%S')
    if args.out is None:
        args.out = Path(f"runs/experiments/{args.variant}-{stamp}" if args.frozen else f"runs/bench-{stamp}")
    if args.variant != "baseline" and not args.frozen:
        print("--variant needs --frozen (variants use the tracker context stored in a frozen set)", file=sys.stderr)
        return 2

    settings = Settings()
    sop = load_sop(args.sop) if args.sop else list_sops(settings.sops_dir)[0]
    accept = sop.decisions.accept_confidence
    items = load_frozen(args.frozen, args.limit) if args.frozen else load(args.logs, args.limit)
    if not items:
        print("no decision_comparison records in those logs (run with DECISION_MODE=compare)", file=sys.stderr)
        return 1
    variant = VARIANTS[args.variant]
    expected = {(source, r["request_id"]): expected_steps(r, sop) for source, r in items}  # ids are unique per run only
    if not args.no_replay:
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        if not 1 <= len(providers) <= 2:
            print("--providers takes one or two providers", file=sys.stderr)
            return 2
        items = asyncio.run(replay(items, settings, accept, lambda r: variant(r, sop), providers))
    apply = (lambda source, r: r) if args.frozen else labeller(settings, sop)
    records = [apply(source, r) | {"source": source, "expected_steps": expected[(source, r["request_id"])]} for source, r in items]
    summary = summarize(records, accept)

    # Appended, not with_suffix: names like bench-v2.1 must keep their dots.
    json_path, jsonl_path = Path(f"{args.out}.json"), Path(f"{args.out}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            # Experiments keep the request each provider saw, so a variant can be audited later.
            f.write(json.dumps({k: v for k, v in r.items() if k != "request" or args.frozen}) + "\n")
    json_path.write_text(json.dumps({"logs": [str(p) for p in args.logs], "frozen": str(args.frozen) if args.frozen else None,
                                     "variant": args.variant, "replayed": not args.no_replay, "providers": args.providers,
                                     "decider": decider_info(settings) if "decider" in args.providers else None,
                                     "laya": laya_info(settings) if "laya" in args.providers else None,
                                     "summary": summary}, indent=2), encoding="utf-8")
    for line in summary_text(summary):
        print(line)
    print(f"summary: {json_path}  records: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
