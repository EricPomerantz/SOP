"""Spike: record the exact state sent at every decision call of a lockstep replay, without calling any model.

The scene part of the state depends only on the video, so it matches what Jev/Kev saw in compare_deciders.py at the
same times. Usage: python spikes/dump_scenes.py --run 12 [--out runs/scenes-Install_12.json]
"""

import argparse
import asyncio
import json
from pathlib import Path

from sopvision.config import list_sops
from sopvision.dataset import load_index
from sopvision.decide.types import JevAnswers
from sopvision.perception.detector import YoloDetector
from sopvision.pipeline import run_lockstep
from sopvision.settings import Settings
from sopvision.sources.file import FileSource


class NullClient:
    """Records each request and answers 'idle' with zero confidence, so the tracker never acts on it."""

    def __init__(self):
        self.requests = []

    async def ask(self, state, qs):
        self.requests.append({"t": state["scene"]["t"], "active": qs.active_step_id, "state": state})
        return JevAnswers("idle", 0.0, {"idle": 1.0}, 0.0, 0.0, 0.0)

    async def aclose(self):
        pass


async def main(run_no: int, out: Path) -> None:
    settings = Settings()
    sop = list_sops(settings.sops_dir)[0]
    run = {r.run: r for r in load_index(settings.sop_data_root)}[run_no]
    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    client = NullClient()
    await run_lockstep(sop, FileSource(run.raw_path, mode="lockstep"), detector, client, run_id="dump",
                       source_id=f"test:Install_{run_no}")
    out.write_text(json.dumps(client.requests, ensure_ascii=False), encoding="utf-8")
    print(f"{len(client.requests)} requests -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=12)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    asyncio.run(main(a.run, a.out or Path(f"runs/scenes-Install_{a.run}.json")))
