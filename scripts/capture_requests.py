"""Capture decision requests from the training/validation recordings, for building a fine-tuning set.

Each recording is replayed in lockstep through the real detector, scene builder and tracker, exactly as `sopvision eval`
does, but with a stub model that never acts: steps still complete on visual evidence, so the states the stub is asked
about are what a live run produces, without any model's opinion feeding back into them. Every request is written to a
run log (`decision_comparison` events, with step updates), which scripts/build_training_set.py turns into labelled rows.

    docker compose run --rm --entrypoint python sopvision scripts/capture_requests.py --out data/training/captures

Recordings: the train split with valid step timing, and the validation split. The test recordings (Install_12, 13), which
the frozen benchmark comes from, are refused.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sopvision.config import list_sops
from sopvision.dataset import load_index
from sopvision.decide.types import JevAnswers
from sopvision.pipeline import run_lockstep
from sopvision.settings import Settings
from sopvision.sources.file import FileSource
from sopvision.storage.runlog import RunLog

TEST_RUNS = {12, 13}  # the frozen benchmark's recordings: never capture them for training


class StubClient:
    """Answers every request with nothing the tracker would act on (confidence 0)."""

    def __init__(self, provider: str):
        self.provider = provider

    async def ask(self, state, qs) -> JevAnswers:
        return JevAnswers("idle", 0.0, {"idle": 1.0}, 0.0, 0.0, 0.0, self.provider)

    async def aclose(self) -> None:
        pass


async def capture(settings: Settings, out: Path) -> None:
    from sopvision.perception.detector import YoloDetector

    sop = list_sops(settings.sops_dir)[0]
    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    out.mkdir(parents=True, exist_ok=True)
    for run in load_index(settings.sop_data_root):
        if run.run in TEST_RUNS or run.split == "test":
            continue
        if run.split == "train" and not run.timing_valid:
            print(f"skip Install_{run.run}: step timing invalid, labels would be wrong", flush=True)
            continue
        source = f"{run.split}:Install_{run.run}"
        path = out / f"{run.split}-Install_{run.run}.jsonl"
        path.unlink(missing_ok=True)
        detector.reset()
        # The shadow stub makes the run log carry each request as a decision_comparison event.
        res = await run_lockstep(sop, FileSource(run.raw_path, mode="lockstep"), detector, StubClient("capture"),
                                 runlog=RunLog(path), run_id=f"capture-{run.run}", source_id=source,
                                 shadow=StubClient("capture-shadow"))
        print(f"{source}: {len(res.comparisons)} requests -> {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/training/captures"))
    args = ap.parse_args()
    asyncio.run(capture(Settings(), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
