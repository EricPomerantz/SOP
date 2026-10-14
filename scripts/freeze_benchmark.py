"""Freeze recorded Jev/Decider decision requests into a fixed benchmark set (run once; never rewrite a frozen set).

    docker compose run --rm --entrypoint python sopvision scripts/freeze_benchmark.py \
        runs/eval-compare.test-Install_12.jsonl runs/eval-compare.test-Install_13.jsonl --out benchmarks/sop-244

Writes <out>/requests.jsonl, one line per decision_comparison record:
    id, source, request_id, t_request, version, request {state, questions} exactly as sent,
    tracker {done_t: {step: time}}: when each done step was completed, as the tracker knew it at request time
                                     (replayed from the run log's step_update events), for request variants that need it,
    ground_truth (sopvision.evaluation.decisions; None where unknown)
and <out>/manifest.json with checksums of the output and of the source logs. Refuses to overwrite.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_decisions import labeller  # noqa: E402

from sopvision.config import list_sops, load_sop  # noqa: E402
from sopvision.settings import Settings  # noqa: E402
from sopvision.storage.runlog import read_events  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path: Path, apply) -> list[dict]:
    """Every decision request in a run log, with the tracker's completion times at that moment and its label."""
    events = read_events(path)
    source = next(e.payload["source"] for e in events if e.type == "run_started")
    done_t: dict[str, float] = {}
    records = []
    for e in events:  # file order = the order the run processed them
        if e.type == "step_update":
            if e.payload["status"] == "done":
                done_t[e.payload["step_id"]] = e.t
            else:
                done_t.pop(e.payload["step_id"], None)  # part removed: back to pending
        elif e.type == "decision_comparison":
            p = e.payload
            rec = {"id": f"{source}/{p['request_id']}", "source": source, "request_id": p["request_id"],
                   "t_request": p["t_request"], "version": p["version"], "request": p["request"],
                   "tracker": {"done_t": dict(done_t)}, "ground_truth": None}
            rec["ground_truth"] = apply(source, rec)["ground_truth"]
            records.append(rec)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sop", type=Path, default=None, help="SOP file (default: the first in sops/)")
    args = ap.parse_args()
    if (args.out / "requests.jsonl").exists():
        print(f"{args.out} is already frozen; refusing to overwrite", file=sys.stderr)
        return 1
    settings = Settings()
    apply = labeller(settings, load_sop(args.sop) if args.sop else list_sops(settings.sops_dir)[0])
    records = [r for path in args.logs for r in extract(path, apply)]
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "requests.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            # Never sort keys: key and option order are part of the prompt the models read.
            f.write(json.dumps(r) + "\n")
    labelled = sum(bool((r["ground_truth"] or {}).get("current_step")) for r in records)
    manifest = {"requests": len(records), "labelled_current_step": labelled, "sha256": {"requests.jsonl": sha256(out)},
                "sources": {str(p): sha256(p) for p in args.logs},
                "note": "Frozen evaluation set. Never train on it, never rewrite it; runs 12-13 are excluded from training data."}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # The frozen requests must be byte-identical (order included) to what was sent.
    sent = [e.payload["request"] for p in args.logs for e in read_events(p) if e.type == "decision_comparison"]
    frozen = [json.loads(line)["request"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert [json.dumps(r) for r in sent] == [json.dumps(r) for r in frozen], "frozen requests differ from the log"
    print(f"froze {len(records)} requests ({labelled} labelled) into {out}; byte-identical to the logged requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
