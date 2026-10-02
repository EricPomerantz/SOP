"""Throwaway: run the real live pipeline on Install_12 for 30 s and report processed fps and dropped frames."""

import asyncio
import time
from pathlib import Path

from sopvision.config import load_sop
from sopvision.decide.jev_client import JevClient
from sopvision.perception.detector import YoloDetector
from sopvision.pipeline import LiveRun, RunContext
from sopvision.server.frames import FrameBuffer
from sopvision.settings import Settings
from sopvision.sources.file import FileSource
from sopvision.sources.registry import SourceSpec
from sopvision.storage.runlog import RunLog

SECONDS = 30


async def main():
    s, sop = Settings(), load_sop(Path("sops/server_fan.yaml"))
    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    client = JevClient(s.typesafe_api_key.get_secret_value(), s.typesafe_base_url, s.typesafe_model)
    msgs = []
    ctx = RunContext("bench", sop, SourceSpec(id="test:Install_12", kind="file", label="bench"),
                     RunLog(Path("runs/bench.jsonl")), msgs.append, FrameBuffer())
    run = LiveRun(ctx, FileSource(s.sop_data_root / "raw" / "Install_12.MP4"), detector, client)
    await run.start()
    await asyncio.sleep(3)  # warm-up (CUDA init, first frames)
    seq0, t0 = ctx.frames.latest()[0], time.perf_counter()
    await asyncio.sleep(SECONDS)
    frames, dt = ctx.frames.latest()[0] - seq0, time.perf_counter() - t0
    await run.stop()
    steps = [m for m in msgs if m["type"] == "step_update" and m["status"] == "done"]
    decisions = [m for m in msgs if m["type"] == "decision"]
    print(f"processed {frames / dt:.1f} fps over {dt:.0f} s; dropped {run.reader.dropped} of the source's frames")
    print(f"{len(decisions)} Jev decisions, median latency {sorted(d['latency_ms'] for d in decisions)[len(decisions) // 2]:.0f} ms"
          if decisions else "no Jev decisions")
    print("steps done:", [m["step_id"] for m in steps])


asyncio.run(main())
