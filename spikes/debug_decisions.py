"""Throwaway: lockstep replay that records every Jev question/answer, for diagnosing eval failures."""

import asyncio
import json
import sys
from pathlib import Path

from sopvision.config import load_sop
from sopvision.dataset import load_index
from sopvision.decide.jev_client import JevClient
from sopvision.perception.detector import YoloDetector
from sopvision.pipeline import run_lockstep
from sopvision.settings import Settings
from sopvision.sources.file import FileSource


class Recording:
    def __init__(self, inner):
        self.inner, self.log = inner, []

    async def ask(self, state, qs):
        a = await self.inner.ask(state, qs)
        self.log.append({"t": state["scene"]["t"], "active": qs.active_step_id, "hands": state["scene"]["hands"],
                         "held": state["scene"]["held"], "recent": state["recent_events"][-3:],
                         "answer": [a.current_step, round(a.current_confidence, 2)],
                         "off_sop_p": round(a.current_probabilities.get("off_sop", 0), 2),
                         "idle_p": round(a.current_probabilities.get("idle", 0), 2), "done": round(a.active_step_done, 2)})
        return a


async def main(runs):
    s, sop = Settings(), load_sop(Path("sops/server_fan.yaml"))
    det = YoloDetector(sop.detector.weights, sop.detector.classes)
    index = {r.run: r for r in load_index(s.sop_data_root)}
    for n in runs:
        client = Recording(JevClient(s.typesafe_api_key.get_secret_value(), s.typesafe_base_url, s.typesafe_model))
        det.reset()
        await run_lockstep(sop, FileSource(index[n].raw_path, mode="lockstep"), det, client)
        Path(f"runs/debug-decisions-{n}.json").write_text(json.dumps(client.log, indent=0))
        print(n, len(client.log), "decisions")


asyncio.run(main([int(a) for a in sys.argv[1:]]))
