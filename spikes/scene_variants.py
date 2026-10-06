"""Spike: try scene-description variants on recorded requests (spikes/dump_scenes.py) against Jev and Kev.

No replay: each recorded state is rewritten per variant and sent straight to each backend, and `current_step` is
scored against the ground-truth step at that moment. Usage: python spikes/scene_variants.py [--kev-url ...]
"""

import argparse
import asyncio
import copy
import json
from pathlib import Path

from sopvision.config import list_sops
from sopvision.dataset import load_index
from sopvision.decide.jev_client import JevClient
from sopvision.decide.questions import QuestionSet, build_questions
from sopvision.settings import Settings

ACCEPT = 0.8


def baseline(sop, state, qs):
    return state, qs


def step_bays(sop, state, qs):
    state = copy.deepcopy(state)
    for entry in state["steps"]:
        entry["bay"] = ", ".join(sop.step(entry["id"]).evidence)
    return state, qs


def step_bays_and_options(sop, state, qs):
    state, _ = step_bays(sop, state, qs)
    options = {sid: (f"{text} (bay: {', '.join(sop.step(sid).evidence)})" if sid in {s.id for s in sop.steps} else text)
               for sid, text in qs.current_step_options.items()}
    return state, QuestionSet(options, qs.active_step_id, qs.active_step_question)


def no_hands(sop, state, qs):
    state = copy.deepcopy(state)
    state["scene"].pop("hands", None)
    state["recent_events"] = [e for e in state["recent_events"] if " hand " not in e]
    return state, qs


def no_hands_no_occlusion(sop, state, qs):
    state, qs = no_hands(sop, state, qs)
    for name, view in state["scene"]["slots"].items():
        if view["state"] == "occluded" and view.get("last"):
            state["scene"]["slots"][name] = {"state": view["last"]}
    state["recent_events"] = [e for e in state["recent_events"] if "occluded" not in e]
    return state, qs


def next_expected(sop, state, qs):
    state = copy.deepcopy(state)
    state["next_expected_step"] = next((e["id"] for e in state["steps"] if e["status"] != "done"), None)
    return state, qs


VARIANTS = {"A as today": baseline, "D no hands": no_hands, "E no hands, no occlusion": no_hands_no_occlusion,
            "H + next expected step": next_expected}


async def run_variant(client, sop, requests, variant, limit=4):
    sem = asyncio.Semaphore(limit)

    async def one(r):
        state, qs = variant(sop, r["state"], build_questions(sop, r["active"]))
        async with sem:
            try:
                a = await client.ask(state, qs)
            except Exception:  # noqa: BLE001 — counted as a failure
                return None
        ok = a.current_step == r["truth"] or (r["truth"] == "off_sop" and a.current_step in ("idle", "off_sop"))
        return {"correct": ok, "conf": a.current_confidence, "ms": a.latency_ms}

    return await asyncio.gather(*(one(r) for r in requests))


def line(results) -> str:
    done = [x for x in results if x]
    conf = [x for x in done if x["conf"] >= ACCEPT]
    pct = lambda xs, n: f"{100 * sum(xs) / n:3.0f}%" if n else "  –"
    return (f"correct {pct([x['correct'] for x in done], len(done))}   confident {pct([True] * len(conf), len(done))}"
            f"   correct when confident {pct([x['correct'] for x in conf], len(conf))}   failed {len(results) - len(done)}")


async def main(kev_url: str) -> None:
    settings = Settings()
    sop = list_sops(settings.sops_dir)[0]
    index = {r.run: r for r in load_index(settings.sop_data_root)}
    requests = []
    for n in (12, 13):
        for r in json.loads(Path(f"runs/scenes-Install_{n}.json").read_text(encoding="utf-8")):
            truth = next((s.step_id for s in index[n].segments if s.start <= r["t"] < s.end), None)
            if truth:
                requests.append(r | {"truth": truth})
    backends = {"Jev": lambda: JevClient(settings.typesafe_api_key.get_secret_value(), settings.typesafe_base_url,
                                         settings.typesafe_model),
                "Kev-4B": lambda: JevClient("kev-local", kev_url, "kev-4b")}
    print(f"{len(requests)} recorded requests")
    order = [r["active"] == r["truth"] for r in requests]
    print(f"{'order only':7} {'next unfinished step, no model':28} correct {100 * sum(order) / len(order):3.0f}%")
    for name, make in backends.items():
        client = make()
        try:
            for vname, variant in VARIANTS.items():
                print(f"{name:7} {vname:28} {line(await run_variant(client, sop, requests, variant))}", flush=True)
        finally:
            await client.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kev-url", default="http://127.0.0.1:8009")
    asyncio.run(main(ap.parse_args().kev_url))
