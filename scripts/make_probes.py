"""Build a deviation probe set from a frozen benchmark (evaluation only; never train on it).

The frozen recordings are clean, in-order runs, where "answer the step the tracker expects" is almost always right.
Each probe takes a frozen request and edits the scene so that a known deviation is happening, so a model that only
echoes the tracker gets it wrong. The edits come from the SOP file (sopvision.evaluation.deviations):

    out_of_order_work     working on a later, not-yet-allowed step (its part in hand at its slot, or the tool there)
    out_of_order_install  a later step's slot just became installed / torqued      -> anomaly
    idle_bench            nobody at the bench for 20-30 s                          -> idle / off_sop
    done_part_removed     a completed step's part just left its slot               -> anomaly

    docker compose run --rm --entrypoint python sopvision scripts/make_probes.py benchmarks/sop-244 \
        --out benchmarks/probes-d1 [--sop sops/lug_bolts.yaml]

Synthetic edits of real states: a guardrail against echoing, not a substitute for recordings of real deviations. A model
trained on rows from the same generator (scripts/build_training_set.py) must be tested on something else.
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

from sopvision.config import list_sops, load_sop
from sopvision.evaluation.deviations import ELIGIBLE, GENERATORS
from sopvision.settings import Settings

COUNTS = {"out_of_order_work": 40, "out_of_order_install": 30, "idle_bench": 40, "done_part_removed": 30}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frozen", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sop", type=Path, default=None, help="SOP file (default: the first in sops/)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if (args.out / "requests.jsonl").exists():
        print(f"{args.out} already exists; refusing to overwrite", file=sys.stderr)
        return 1
    sop = load_sop(args.sop) if args.sop else list_sops(Settings().sops_dir)[0]
    base = [json.loads(line) for line in (args.frozen / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    base = [r for r in base if (r["ground_truth"] or {}).get("current_step") and r["ground_truth"]["current_step"] != ["idle", "off_sop"]]
    rng = random.Random(args.seed)
    probes = []
    for kind, n in COUNTS.items():
        pool = [r for r in base if ELIGIBLE[kind](r, sop)]
        for src in rng.sample(pool, min(n, len(pool))):
            rec = copy.deepcopy(src)
            truth = GENERATORS[kind](rec, sop)
            rec.update(id=f"{kind}/{src['id']}", request_id=f"{kind}/{src['request_id']}", probe=kind,
                       derived_from=src["id"], ground_truth={"current_step": truth["current_step"],
                                                             "anomaly": truth["anomaly"], "source": f"probe:{kind}"})
            probes.append(rec)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "requests.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in probes:
            f.write(json.dumps(r) + "\n")
    manifest = {"requests": len(probes), "by_probe": {k: sum(p["probe"] == k for p in probes) for k in COUNTS},
                "seed": args.seed, "derived_from": str(args.frozen), "sop": sop.sop,
                "sha256": {"requests.jsonl": hashlib.sha256(out.read_bytes()).hexdigest()},
                "note": "Synthetic deviation probes for evaluation only. Never train on them."}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(probes)} probes -> {out}: {manifest['by_probe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
