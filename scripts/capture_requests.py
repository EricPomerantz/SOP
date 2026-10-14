"""Capture decision requests from the training/validation recordings, for building a fine-tuning set.

Each recording is replayed in lockstep through the real detector, scene builder and tracker, exactly as `sopvision eval`
does, but with a stub model that never acts: steps still complete on visual evidence, so the states the stub is asked
about are what a live run produces, without any model's opinion feeding back into them. Every request is written to a
run log (`decision_comparison` events, with step updates), which scripts/build_training_set.py turns into labelled rows.

    docker compose run --rm --entrypoint python sopvision scripts/capture_requests.py --out data/training/captures \
        [--sop sops/lug_bolts.yaml]

Recordings (sopvision.recordings): the train and validation splits with a usable step timeline; a recording's torque log
(torque.csv) is replayed with it. Test-split recordings are only captured with --benchmark, which captures nothing else:
its logs are for scripts/freeze_benchmark.py, never for a training set (build_training_set.py refuses test sources).

    docker compose run --rm --entrypoint python sopvision scripts/capture_requests.py --benchmark \
        --sop sops/lug_bolts.yaml --out runs/capture-lug_bolts-test
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sopvision.config import SopConfig, list_sops, load_sop
from sopvision.decide.types import JevAnswers
from sopvision.pipeline import run_lockstep
from sopvision.recordings import load_recordings
from sopvision.settings import Settings
from sopvision.sop.torque import read_torque_log
from sopvision.sources.file import FileSource
from sopvision.storage.runlog import RunLog


class StubClient:
    """Answers every request with nothing the tracker would act on (confidence 0)."""

    def __init__(self, provider: str):
        self.provider = provider

    async def ask(self, state, qs) -> JevAnswers:
        return JevAnswers("idle", 0.0, {"idle": 1.0}, 0.0, 0.0, 0.0, self.provider)

    async def aclose(self) -> None:
        pass


async def capture(settings: Settings, sop: SopConfig, out: Path, only: set[str] | None = None,
                  benchmark: bool = False) -> None:
    from sopvision.perception.detector import YoloDetector

    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    out.mkdir(parents=True, exist_ok=True)
    for rec in load_recordings(sop, settings):
        if (rec.split == "test") != benchmark or (only and rec.id not in only):
            continue
        if not rec.valid:
            print(f"skip {rec.id}: no usable step timeline, labels would be wrong", flush=True)
            continue
        path = out / f"{rec.split}-{rec.id}.jsonl"
        path.unlink(missing_ok=True)
        detector.reset()
        torque = read_torque_log(rec.torque_log) if rec.torque_log else None
        # The shadow stub makes the run log carry each request as a decision_comparison event.
        res = await run_lockstep(sop, FileSource(rec.video, mode="lockstep"), detector, StubClient("capture"),
                                 runlog=RunLog(path), run_id=f"capture-{rec.id}", source_id=rec.source_id,
                                 shadow=StubClient("capture-shadow"), torque=torque)
        print(f"{rec.source_id}: {len(res.comparisons)} requests -> {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/training/captures"))
    ap.add_argument("--sop", type=Path, default=None, help="SOP file (default: the first in sops/)")
    ap.add_argument("--recording", action="append", help="only these recording ids (repeatable)")
    ap.add_argument("--benchmark", action="store_true", help="capture the test recordings only, for a frozen benchmark")
    args = ap.parse_args()
    settings = Settings()
    sop = load_sop(args.sop) if args.sop else list_sops(settings.sops_dir)[0]
    asyncio.run(capture(settings, sop, args.out, set(args.recording) if args.recording else None,
                        args.benchmark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
