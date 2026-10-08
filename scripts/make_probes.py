"""Build the D1 deviation probe set from a frozen benchmark (evaluation only; never train on it).

The frozen recordings are clean, in-order installs, where "answer the step the tracker expects" is right 97% of the
time. Each probe takes a frozen request and changes the scene so that a known deviation is happening, so a model
that only echoes the tracker gets it wrong:

    P1 out_of_order_work     holding a fan, both hands at fan_bay_{k+2} while fan_k is expected  -> current_step fan_{k+2}
    P2 out_of_order_install  fan_bay_{k+2} has just become installed while fan_bay_k is empty      -> anomaly yes
    P3 idle_bench            no hands, nothing held, last activity 20-30 s ago                    -> idle / off_sop
    P4 done_part_removed     a completed fan's bay just went installed -> empty, a hand on it      -> anomaly yes

    docker compose run --rm --entrypoint python sopvision scripts/make_probes.py benchmarks/sop-244 --out benchmarks/probes-d1

Synthetic edits of real states: a guardrail against echoing, not a substitute for recordings of real deviations.
"""

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

COUNTS = {"out_of_order_work": 40, "out_of_order_install": 30, "idle_bench": 40, "done_part_removed": 30}
FANS = [f"fan_{i}" for i in range(1, 7)]
_EVENT_T = re.compile(r"^(\d+(?:\.\d+)?)s ")


def statuses(rec):
    return {s["id"]: s["status"] for s in rec["request"]["state"]["steps"]}


def expected(rec):
    st = statuses(rec)
    return next((s["id"] for s in rec["request"]["state"]["steps"] if st[s["id"]] != "done"), None)


def events(scene_t, *lines):
    return [f"{scene_t - dt:.1f}s {text}" for dt, text in lines]


def out_of_order_work(rec):
    k = int(expected(rec)[-1])
    st = rec["request"]["state"]; sc = st["scene"]; t = sc["t"]; target = f"fan_bay_{k + 2}"
    sc["slots"][f"fan_bay_{k}"] = {"state": "empty", "conf": 0.93}
    sc["slots"][target] = {"state": "occluded", "last": "empty"}
    sc["hands"], sc["held"] = [{"near": target}, {"near": target}], ["fan"]
    st["recent_events"] = st["recent_events"][:6] + events(t, (2.4, "held: fan"), (1.6, f"hand entered {target}"),
                                                           (1.2, f"{target} empty→occluded"), (0.8, f"hand entered {target}"))
    return {"current_step": [f"fan_{k + 2}"], "anomaly": None}


def out_of_order_install(rec):
    k = int(expected(rec)[-1])
    st = rec["request"]["state"]; sc = st["scene"]; t = sc["t"]; target = f"fan_bay_{k + 2}"
    sc["slots"][f"fan_bay_{k}"] = {"state": "empty", "conf": 0.93}
    sc["slots"][target] = {"state": "installed", "conf": 0.95}
    sc["hands"], sc["held"] = [{"near": "chassis"}], []
    st["recent_events"] = st["recent_events"][:6] + events(t, (3.0, f"hand entered {target}"), (1.5, f"{target} occluded→installed"),
                                                           (1.0, "held: nothing"), (0.6, f"hand left {target}"))
    return {"current_step": None, "anomaly": True}


def idle_bench(rec):
    st = rec["request"]["state"]; sc = st["scene"]; t = sc["t"]
    sc["hands"], sc["held"] = [], []
    old = [_EVENT_T.sub("", e) for e in st["recent_events"]]  # same events, but long ago
    st["recent_events"] = [f"{t - 30 + i:.1f}s {e}" for i, e in enumerate(old)] + events(t, (20.0, "hand left chassis"))
    st["recent_events"] = st["recent_events"][-10:]
    return {"current_step": ["idle", "off_sop"], "anomaly": False}


def done_part_removed(rec):
    st = rec["request"]["state"]; sc = st["scene"]; t = sc["t"]
    done = [s for s in FANS if statuses(rec)[s] == "done"]
    bay = f"fan_bay_{done[-1][-1]}"
    sc["slots"][bay] = {"state": "empty", "conf": 0.92}
    sc["hands"], sc["held"] = [{"near": bay}], ["fan"]
    st["recent_events"] = st["recent_events"][:6] + events(t, (2.0, f"hand entered {bay}"), (1.2, f"{bay} occluded→empty"),
                                                           (1.0, "held: fan"), (0.5, f"hand left {bay}"))
    return {"current_step": None, "anomaly": True}


ELIGIBLE = {
    "out_of_order_work": lambda r: expected(r) in FANS[:4],
    "out_of_order_install": lambda r: expected(r) in FANS[:4],
    "idle_bench": lambda r: True,
    "done_part_removed": lambda r: any(statuses(r)[s] == "done" for s in FANS),
}
BUILD = {"out_of_order_work": out_of_order_work, "out_of_order_install": out_of_order_install,
         "idle_bench": idle_bench, "done_part_removed": done_part_removed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frozen", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if (args.out / "requests.jsonl").exists():
        print(f"{args.out} already exists; refusing to overwrite", file=sys.stderr)
        return 1
    base = [json.loads(line) for line in (args.frozen / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    base = [r for r in base if (r["ground_truth"] or {}).get("current_step") and r["ground_truth"]["current_step"] != ["idle", "off_sop"]]
    rng = random.Random(args.seed)
    probes = []
    for kind, n in COUNTS.items():
        pool = [r for r in base if ELIGIBLE[kind](r)]
        for src in rng.sample(pool, min(n, len(pool))):
            rec = copy.deepcopy(src)
            truth = BUILD[kind](rec)
            rec.update(id=f"{kind}/{src['id']}", request_id=f"{kind}/{src['request_id']}", probe=kind,
                       derived_from=src["id"], ground_truth=truth | {"source": f"probe:{kind}"})
            probes.append(rec)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "requests.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in probes:
            f.write(json.dumps(r) + "\n")
    manifest = {"requests": len(probes), "by_probe": {k: sum(p["probe"] == k for p in probes) for k in COUNTS},
                "seed": args.seed, "derived_from": str(args.frozen),
                "sha256": {"requests.jsonl": hashlib.sha256(out.read_bytes()).hexdigest()},
                "note": "Synthetic deviation probes for evaluation only. Never train on them."}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(probes)} probes -> {out}: {manifest['by_probe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
