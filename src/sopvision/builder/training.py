"""Train, evaluate and report one SOP's own Laya checkpoint: the stages of the builder's `train_laya` job.

    python -m sopvision.builder.training <stage> --sop sops/<file>.yaml --version <n>

Stages, in order (each one a job step, so a failure says where):
  check      the SOP loads; train, val and test runs with step timelines exist; the Laya service is up and new
             enough to serve candidates
  capture    decision requests from the train/val recordings (scripts/capture_requests.py)
  benchmark  the same from the test recordings, frozen as the benchmark (capture --benchmark, freeze_benchmark.py)
  build      the fine-tuning set, off-SOP stretches labelled off_sop, benchmark states left out (build_training_set.py)
  train      laya-train from the shipped Laya, the SOP's rows only
  evaluate   the candidate served by name (POST /admin/register), scored on the benchmark (bench_decisions.py) and
             replayed end to end on the test recordings (replay_recordings.py); the active checkpoint, if any, on the
             same benchmark for comparison
  report     report.json: the numbers the dashboard shows before anyone activates the checkpoint

Everything for version n of SOP <id> is named <id>-v<n>: data/training/captures-<id>-v<n>, runs/capture-<id>-v<n>-test,
benchmarks/<id>-v<n>-test, data/training/<id>-v<n>, data/training/models/<id>-v<n> (+ .eval/ with the results).
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STAGES = ("check", "capture", "benchmark", "build", "train", "evaluate", "report")
MODELS = Path("data/training/models")


@dataclass(frozen=True)
class Paths:
    name: str
    captures: Path
    test_captures: Path
    benchmark: Path
    training_set: Path
    model: Path
    eval: Path


def paths(sop_id: str, version: int) -> Paths:
    name = f"{sop_id}-v{version}"
    return Paths(name, Path(f"data/training/captures-{name}"), Path(f"runs/capture-{name}-test"),
                 Path(f"benchmarks/{name}-test"), Path(f"data/training/{name}"), MODELS / name, MODELS / f"{name}.eval")


def next_version(sop_id: str) -> int:
    taken = [int(p.name.rsplit("-v", 1)[1]) for p in MODELS.glob(f"{sop_id}-v*")
             if p.name.rsplit("-v", 1)[1].split(".")[0].isdigit() and "." not in p.name]
    return max(taken, default=0) + 1


def versions(sop_id: str) -> list[dict]:
    """Trained checkpoints of an SOP with their reports, newest first."""
    out = []
    for p in sorted(MODELS.glob(f"{sop_id}-v*"), reverse=True):
        if p.is_dir() and not p.name.endswith(".eval") and (p / "model.safetensors").exists():
            report = p.with_name(p.name + ".eval") / "report.json"
            out.append({"checkpoint": p.name, "report": json.loads(report.read_text(encoding="utf-8")) if report.exists() else None})
    return sorted(out, key=lambda v: int(v["checkpoint"].rsplit("-v", 1)[1]), reverse=True)


def _run(argv: list[str]) -> None:
    print("$ " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def check(sop_file: Path) -> None:
    import httpx

    from sopvision.config import load_sop
    from sopvision.recordings import load_recordings
    from sopvision.settings import Settings

    settings = Settings()
    sop = load_sop(sop_file)
    recs = load_recordings(sop, settings)
    timed = {s: [r.id for r in recs if r.split == s and r.valid] for s in ("train", "val", "test")}
    missing = [r.id for r in recs if not r.valid]
    print(f"recordings with timelines: " + ", ".join(f"{s} {len(v)}" for s, v in timed.items()), flush=True)
    if missing:
        print(f"without a timeline (left out): {', '.join(missing)}", flush=True)
    short = [s for s, v in timed.items() if not v]
    if short:
        raise SystemExit(f"Need at least one {', '.join(short)} run with a saved step timeline")
    try:
        httpx.get(f"{settings.laya_url}/health", timeout=5).raise_for_status()
        admin = httpx.get(f"{settings.laya_url}/admin/models", timeout=5)
    except httpx.HTTPError as e:
        raise SystemExit(f"The Laya service isn't answering at {settings.laya_url} ({e}): start it "
                         "(./start.sh --model laya) and run this again") from None
    if admin.status_code == 404:
        raise SystemExit("The Laya service is too old to serve per-SOP checkpoints: rebuild it "
                         "(docker compose --profile laya build laya) and restart it")


def capture(sop_file: Path, p: Paths) -> None:
    _run([sys.executable, "scripts/capture_requests.py", "--sop", str(sop_file), "--out", str(p.captures)])


def benchmark(sop_file: Path, p: Paths) -> None:
    if (p.benchmark / "requests.jsonl").exists():
        print(f"{p.benchmark} already frozen", flush=True)
        return
    _run([sys.executable, "scripts/capture_requests.py", "--benchmark", "--sop", str(sop_file), "--out", str(p.test_captures)])
    logs = sorted(str(x) for x in p.test_captures.glob("*.jsonl"))
    if not logs:
        raise SystemExit("The test recordings gave no decision requests")
    _run([sys.executable, "scripts/freeze_benchmark.py", *logs, "--out", str(p.benchmark), "--sop", str(sop_file)])


def build(sop_file: Path, p: Paths) -> None:
    _run([sys.executable, "scripts/build_training_set.py", str(p.captures), "--out", str(p.training_set), "--sop",
          str(sop_file), "--frozen", str(p.benchmark), "--off-sop-label"])


def train(p: Paths, epochs: int) -> None:
    _run(["laya-train", "--data", str(p.training_set / "train.jsonl"), "--out", str(p.model),
          "--base", "convaiinnovations/laya", "--max-len", "2048", "--head-max-len", "512", "--epochs", str(epochs),
          "--device", "cuda", "--seed", "0"])


def evaluate(sop_file: Path, p: Paths) -> None:
    import httpx

    from sopvision.config import load_sop
    from sopvision.decide import laya_registry
    from sopvision.settings import Settings

    settings = Settings()
    sop = load_sop(sop_file)
    r = httpx.post(f"{settings.laya_url}/admin/register", json={"folder": p.model.name}, timeout=120)
    r.raise_for_status()
    model = r.json()["model"]
    p.eval.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"DECISION_MODE": "laya"}
    bench = [sys.executable, "scripts/bench_decisions.py", "--frozen", str(p.benchmark), "--providers", "laya",
             "--sop", str(sop_file)]
    print(f"$ bench {model}", flush=True)
    subprocess.run(bench + ["--laya-model", model, "--out", str(p.eval / "bench")], check=True, env=env)
    active = laya_registry.load(settings.laya_registry).get(sop.sop)
    if active and active != p.model.name:
        print(f"$ bench the active checkpoint {active}", flush=True)
        subprocess.run(bench + ["--laya-model", laya_registry.model_name(active), "--out", str(p.eval / "bench-active")],
                       check=True, env=env)
    _run([sys.executable, "scripts/replay_recordings.py", "--sop", str(sop_file), "--split", "test", "--laya-model", model,
          "--out", str(p.eval / "replay.json")])


def _bench_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    laya = json.loads(path.read_text(encoding="utf-8"))["summary"]["providers"]["laya"]
    cs, off, an = laya["current_step"], laya["off_sop"], laya["anomaly"]
    return {"requests": laya["requests"], "errors": laya["errors"], "current_step_accuracy": cs["accuracy"],
            "labelled": cs["labelled"], "echo_rate": cs["echo_rate"], "accuracy_when_acting": cs["accuracy_when_acting"],
            "off_sop_false_negative_rate": off["false_negative_rate"], "off_sop_false_positive_rate": off["false_positive_rate"],
            "anomaly_false_positive_rate": an["false_positive_rate"], "request_ms_p50": laya["request_ms"]["p50"]}


def report(sop_file: Path, p: Paths) -> dict:
    from sopvision.config import load_sop
    from sopvision.decide import laya_registry
    from sopvision.settings import Settings

    sop = load_sop(sop_file)
    manifest = p.training_set / "manifest.json"
    replay = json.loads((p.eval / "replay.json").read_text(encoding="utf-8")) if (p.eval / "replay.json").exists() else None
    out = {
        "sop": sop.sop, "checkpoint": p.model.name,
        "active": laya_registry.load(Settings().laya_registry).get(sop.sop),
        "training_set": {s: v["rows"] for s, v in json.loads(manifest.read_text())["splits"].items()} if manifest.exists() else None,
        "bench": _bench_summary(p.eval / "bench.json"),
        "bench_active": _bench_summary(p.eval / "bench-active.json"),
        "replay": [{k: r[k] for k in ("run", "done", "not_done", "alerts", "current_step", "idle_or_off_sop")}
                   for r in (replay or {}).get("runs", [])],
    }
    (p.eval / "report.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["bench"], indent=2), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--sop", type=Path, required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    from sopvision.config import load_sop

    p = paths(load_sop(args.sop).sop, args.version)
    if args.stage == "check":
        check(args.sop)
    elif args.stage == "capture":
        capture(args.sop, p)
    elif args.stage == "benchmark":
        benchmark(args.sop, p)
    elif args.stage == "build":
        build(args.sop, p)
    elif args.stage == "train":
        train(p, args.epochs)
    elif args.stage == "evaluate":
        evaluate(args.sop, p)
    else:
        report(args.sop, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
