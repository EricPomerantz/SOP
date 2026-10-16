"""Build a fine-tuning set (Laya's `{state, questions, expected}` rows) from captured decision requests.

    docker compose run --rm --entrypoint python sopvision scripts/build_training_set.py data/training/captures \
        --out data/training/sop-v1

Rows come from scripts/capture_requests.py logs of the train and validation recordings, and are of three kinds:

  in_order   the request as captured. Labels from the recording's step timeline:
             current_step  the step under way (off-SOP stretches: "idle"); unlabelled within 1 s of a step boundary
             active_step_done  whether the asked-about step's segment has ended (unlabelled within 1 s of its end)
             anomaly       false (the recordings are clean installs)
  noise      an in_order row with YOLO-style noise (a slot occluded or unknown, a hand dropped or duplicated, an event
             missing, a confidence jittered); labels unchanged
  deviation  the D1 probe edits (scripts/make_probes.py) applied to training states: out-of-order work, an
             out-of-order install, an idle bench, a done part removed. Without them a model learns to answer the
             step the tracker expects, which is right 97% of the time on clean runs and useless for deviations.

The questions are the exact wire format the app sends (sopvision.decide.questions.wire_questions), so training matches
serving. Leakage checks: no row from the test recordings (Install_12, 13), and no state identical to one in the frozen
benchmark or the probe set (such rows are dropped and listed in the manifest). Refuses to overwrite.
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_decisions import labeller  # noqa: E402
from freeze_benchmark import extract  # noqa: E402

from sopvision.builder.timelines import read_rows  # noqa: E402
from sopvision.config import SopConfig, list_sops, load_sop  # noqa: E402
from sopvision.decide.questions import QuestionSet, wire_questions  # noqa: E402
from sopvision.evaluation.deviations import ELIGIBLE, GENERATORS  # noqa: E402
from sopvision.recordings import load_recordings  # noqa: E402
from sopvision.settings import Settings  # noqa: E402

BOUNDARY = 1.0  # seconds around a step's end where "done" is left unlabelled
NOISE_PER_ROW = 0.5  # noisy copies per labelled in_order row
DEVIATION_SHARE = 0.35  # of the final set


def state_key(state: dict) -> str:
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()


def labels(rec: dict, segments, raw_rows=None) -> dict:
    """`raw_rows`: the recording's steps.csv as written (idle and off_sop apart). With it, off-SOP stretches are
    labelled "off_sop" instead of "idle", so the model learns to say off-SOP (the tracker's off_sop alert)."""
    gt = rec["ground_truth"] or {}
    out = {}
    if gt.get("current_step"):
        out["current_step"] = "idle" if len(gt["current_step"]) > 1 else gt["current_step"][0]
        if raw_rows and out["current_step"] == "idle":
            at = next((step for step, a, b in raw_rows if a <= rec["t_request"] < b), None)
            if at == "off_sop":
                out["current_step"] = "off_sop"
    active = rec["request"]["questions"]["active_step_id"]
    seg_end = max((s.end for s in segments if s.step_id == active), default=None)
    if seg_end is not None and abs(rec["t_request"] - seg_end) > BOUNDARY:
        out["active_step_done"] = rec["t_request"] > seg_end
    if gt.get("anomaly") is False:
        out["anomaly"] = False
    return out


def noisy(rec: dict, rng: random.Random) -> dict:
    rec = copy.deepcopy(rec)
    sc, st = rec["request"]["state"]["scene"], rec["request"]["state"]
    for _ in range(rng.choice((1, 2))):
        kind = rng.choice(("occlude", "unknown", "drop_hand", "dup_hand", "drop_event", "jitter"))
        observed = [k for k, v in sc["slots"].items() if v["state"] in ("installed", "empty")]
        if kind in ("occlude", "unknown") and observed:
            slot = rng.choice(observed)
            sc["slots"][slot] = {"state": "occluded" if kind == "occlude" else "unknown", "last": sc["slots"][slot]["state"]}
        elif kind == "drop_hand" and sc["hands"]:
            sc["hands"].pop(rng.randrange(len(sc["hands"])))
        elif kind == "dup_hand" and sc["hands"]:
            sc["hands"].append(dict(rng.choice(sc["hands"])))
        elif kind == "drop_event" and st["recent_events"]:
            st["recent_events"].pop(rng.randrange(len(st["recent_events"])))
        elif kind == "jitter" and observed:
            slot = rng.choice(observed)
            if sc["slots"][slot].get("conf") is not None:
                sc["slots"][slot]["conf"] = round(min(1.0, max(0.3, sc["slots"][slot]["conf"] + rng.uniform(-0.15, 0.1))), 2)
    return rec


def deviation(rec: dict, kind: str, base_labels: dict, sop: SopConfig) -> tuple[dict, dict]:
    rec = copy.deepcopy(rec)
    truth = GENERATORS[kind](rec, sop)
    out = {}
    if truth["current_step"]:
        out["current_step"] = "idle" if len(truth["current_step"]) > 1 else truth["current_step"][0]
    if truth["anomaly"] is not None:
        out["anomaly"] = truth["anomaly"]
    if truth["active_step_done"] is not None:  # e.g. the expected step's slot was emptied
        out["active_step_done"] = truth["active_step_done"]
    elif "active_step_done" in base_labels:
        out["active_step_done"] = base_labels["active_step_done"]
    return rec, out


def row(rec: dict, expected: dict, kind: str, split: str, n: int) -> dict:
    qs = QuestionSet(**rec["request"]["questions"])
    return {"id": f"{rec['id']}#{kind}{n}", "split": split, "kind": kind, "source": rec["source"],
            "state": rec["request"]["state"], "questions": wire_questions(qs), "expected": expected}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, nargs="*", default=[Path("benchmarks/sop-244"), Path("benchmarks/probes-d1")],
                    help="frozen sets whose states must not appear in training (missing folders are skipped)")
    ap.add_argument("--sop", type=Path, default=None, help="SOP file (default: the first in sops/)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--off-sop-label", action="store_true",
                    help="label off-SOP stretches 'off_sop' (timeline datasets); default: 'idle', as sop-v1 was built")
    args = ap.parse_args()
    if (args.out / "train.jsonl").exists():
        print(f"{args.out} already exists; refusing to overwrite", file=sys.stderr)
        return 1
    settings, rng = Settings(), random.Random(args.seed)
    sop = load_sop(args.sop) if args.sop else list_sops(settings.sops_dir)[0]
    recordings = {r.source_id: r for r in load_recordings(sop, settings)}
    apply = labeller(settings, sop)
    frozen_states = {state_key(json.loads(line)["request"]["state"])
                     for d in args.frozen if (d / "requests.jsonl").exists()
                     for line in (d / "requests.jsonl").read_text(encoding="utf-8").splitlines()}

    rows: dict[str, list[dict]] = {"train": [], "val": []}
    for path in sorted(args.captures.glob("*.jsonl")):
        records = extract(path, apply)
        if not records:
            continue
        source = records[0]["source"]
        recording = recordings.get(source)
        if recording is None or recording.split not in rows:  # test recordings never reach training
            print(f"refusing {path}: {source} is not a train/val recording of {sop.sop}", file=sys.stderr)
            return 1
        split, segments = recording.split, recording.segments
        steps_csv = recording.video.parent / "steps.csv"
        raw = read_rows(steps_csv) if args.off_sop_label and steps_csv.exists() else None
        base = [(r, labels(r, segments, raw)) for r in records]
        base = [(r, lab) for r, lab in base if lab]
        out = rows[split]
        start = len(out)
        out += [row(r, lab, "in_order", split, 0) for r, lab in base]
        stepped = [(r, lab) for r, lab in base if "current_step" in lab]
        for i, (r, lab) in enumerate(rng.sample(stepped, int(len(stepped) * NOISE_PER_ROW))):
            out.append(row(noisy(r, rng), lab, "noise", split, i))
        n_dev = int((len(out) - start) * DEVIATION_SHARE / (1 - DEVIATION_SHARE))  # per recording: each split keeps the mix
        working = [(r, lab) for r, lab in stepped if lab["current_step"] != "idle"]
        for i in range(n_dev):
            kind = list(GENERATORS)[i % len(GENERATORS)]
            pool = [(r, lab) for r, lab in working if ELIGIBLE[kind](r, sop)]
            if pool:
                r, lab = rng.choice(pool)
                d, dlab = deviation(r, kind, lab, sop)
                out.append(row(d, dlab, f"deviation:{kind}", split, i))
        print(f"{path.name}: {len(records)} requests -> {len(out) - start} rows", flush=True)

    # A state identical to a frozen benchmark/probe state (e.g. the empty bench every run opens with) is dropped, so no
    # test input is ever seen in training.
    dropped = {split: [r["id"] for r in items if state_key(r["state"]) in frozen_states] for split, items in rows.items()}
    rows = {split: [r for r in items if state_key(r["state"]) not in frozen_states] for split, items in rows.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "sop": sop.sop, "captures": str(args.captures), "off_sop_label": args.off_sop_label,
                "excluded": "test-split recordings (never captured or used)",
                "dropped_identical_to_frozen": dropped, "splits": {}}
    for split, items in rows.items():
        rng.shuffle(items)
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
        manifest["splits"][split] = {
            "rows": len(items), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sources": sorted({r["source"] for r in items}),
            "by_kind": dict(Counter(r["kind"] for r in items)),
            "current_step": dict(Counter(r["expected"].get("current_step") for r in items if "current_step" in r["expected"])),
            "active_step_done": dict(Counter(str(r["expected"]["active_step_done"]) for r in items if "active_step_done" in r["expected"])),
            "anomaly": dict(Counter(str(r["expected"]["anomaly"]) for r in items if "anomaly" in r["expected"])),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
