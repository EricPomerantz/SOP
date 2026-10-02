"""Phase 0 spike (throwaway): verify the Jev SDK call shape, latency and determinism.

Sends the spec's §6.3 example scene with the §7.3 questions 20 times, prints
latencies and the first response, and saves that response as a test fixture.
"""

import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

STEPS = {
    "fan_1": "Install the first fan by connecting the connector, then pressing the fan in place.",
    "fan_2": "Install the second fan by connecting the connector, then pressing the fan in place.",
    "fan_3": "Install the third fan by connecting the connector, then pressing the fan in place.",
    "fan_4": "Install the fourth fan by connecting the connector, then pressing the fan in place.",
    "fan_5": "Install the fifth fan by connecting the connector, then pressing the fan in place.",
    "fan_6": "Install the sixth fan by connecting the connector, then pressing the fan in place.",
    "psu_1": "Install the first power supply by placing it in its compartment and pressing it in until the latch clicks.",
    "psu_2": "Install the second power supply by placing it in its compartment and pressing it in until the latch clicks.",
    "cover": "Place the cover on top of the server and press the black latch to lock it.",
}

STATE = {
    "sop": "Install six fans, two power supplies and the cover on a 1U server.",
    "steps": [{"id": s, "status": "done" if s in ("fan_1", "fan_2") else "active" if s == "fan_3" else "pending"} for s in STEPS],
    "scene": {
        "version": 812,
        "t": 41.2,
        "chassis": {"visible": True, "conf": 0.99},
        "slots": {
            "fan_bay_1": {"state": "installed", "conf": 0.97},
            "fan_bay_2": {"state": "installed", "conf": 0.95},
            "fan_bay_3": {"state": "occluded", "last": "empty"},
        },
        "hands": [{"near": "fan_bay_3"}],
        "held": ["fan"],
    },
    "recent_events": ["38.2s fan_bay_2 empty→installed", "39.0s hand entered fan_bay_3", "40.1s held: fan"],
}

QUESTIONS = {
    "current_step": Choice(
        instructions="Which step is the operator performing right now?",
        criteria={**STEPS, "idle": "No one is working on the server", "off_sop": "Any action that is not one of the steps above."},
    ),
    "active_step_done": Noul(instructions="Has this step been completed: " + STEPS["fan_3"].rstrip(".") + "?"),
    "anomaly": Noul(instructions="Is anything in the scene inconsistent with the SOP so far?"),
}


def main() -> None:
    load_dotenv()
    client = TypeSafeClient(retry=RetryPolicy(max_retries=0))  # raw latency, no SDK retries
    latencies, answers, first = [], [], None
    for _ in range(20):
        start = time.perf_counter()
        resp = client.system_one(state=STATE, questions=QUESTIONS)
        latencies.append((time.perf_counter() - start) * 1000)
        answers.append(json.dumps(resp.model_dump(mode="json")["answers"], sort_keys=True))
        first = first or resp
    client.close()

    lat = sorted(latencies)
    print("model:", first.model, "usage:", first.usage)
    print("first response:", repr(first))
    print(f"latency ms p50={statistics.median(lat):.0f} p90={lat[int(0.9 * len(lat)) - 1]:.0f} max={lat[-1]:.0f}")
    print("identical answers across 20 calls:", len(set(answers)) == 1, f"({len(set(answers))} distinct)")

    out = Path("tests/decide/fixtures/jev_response.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(first.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
