"""Mistakes the recorded footage can't show, scripted at the detection level and run through real Jev."""

import math

import pytest

from sopvision.decide.decider import JevDecider
from sopvision.decide.jev_client import JevClient
from sopvision.perception.detector import Detection
from sopvision.scene.builder import SceneBuilder
from sopvision.settings import Settings
from sopvision.sop.tracker import SopTracker
from sopvision.sop.types import Alert
from tests.helpers import CHASSIS, det_at

pytestmark = pytest.mark.jev
ALL_FANS = {f"fan_bay_{i}": 0.5 for i in range(1, 7)}


async def run_timeline(sop, installed_at: dict[str, float], seconds: float = 15.0) -> list[Alert]:
    """Each slot shows its part from installed_at[slot] onward (empty before; never if absent)."""
    s = Settings()
    builder, tracker, out, last = SceneBuilder(sop), SopTracker(sop), [], {}
    client = JevClient(s.typesafe_api_key.get_secret_value(), s.typesafe_base_url, s.typesafe_model)
    decider = JevDecider(sop, client, "lockstep", on_decision=lambda d: out.extend(tracker.on_decision(d, last["scene"])),
                         on_status=lambda st, t: out.extend(tracker.set_jev_online(st == "online", t)))
    try:
        for i in range(int(seconds * 10)):
            t = i / 10
            dets = [Detection("chassis", CHASSIS, 0.95, None)]
            for name, spec in sop.layout.slots.items():
                if t >= installed_at.get(name, math.inf):
                    dets.append(det_at(spec.filled_by, *spec.at, size=300 if spec.filled_by == "cover" else 40))
                elif spec.empty_as:
                    dets.append(det_at(spec.empty_as, *spec.at))
            scene, events = builder.update(t, dets)
            last["scene"] = scene
            out.extend(tracker.on_scene(scene))
            decider.observe(scene, events, tracker.statuses(), tracker.active_step_id())
            await decider.poll(t)
        out.extend(tracker.end_run(seconds))
    finally:
        await decider.aclose()
        await client.aclose()
    return [a for a in out if isinstance(a, Alert)]


async def test_wrong_order(sop):
    alerts = await run_timeline(sop, {"fan_bay_2": 1.0, "fan_bay_1": 6.0})
    assert "fan_2 completed before fan_1" in [a.message for a in alerts if a.kind == "order_violation"]


async def test_skipped_step(sop):
    alerts = await run_timeline(sop, {"fan_bay_1": 1.0, "fan_bay_2": 2.0, "fan_bay_4": 5.0})
    order = [a for a in alerts if a.kind == "order_violation"]
    assert any(a.step_id == "fan_4" and "fan_3" in a.data["missing"] for a in order)
    (incomplete,) = [a for a in alerts if a.kind == "sop_incomplete"]
    assert "fan_3" in incomplete.data["missing"]


async def test_cover_before_psus(sop):
    alerts = await run_timeline(sop, ALL_FANS | {"cover": 4.0})
    cover = [a for a in alerts if a.kind == "order_violation" and a.step_id == "cover"]
    assert cover and cover[0].data["missing"] == ["psu_1", "psu_2"]
    assert cover[0].message == "cover completed before psu_1, psu_2"
