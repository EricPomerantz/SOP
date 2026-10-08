"""The real local Decider-4B, over HTTP: `docker compose --profile decider up -d decider`, then
`docker compose --profile test run --rm tests -m decider`."""

import pytest

from sopvision.decide.jev_client import JevClient
from sopvision.decide.providers import DECIDER_MODEL, decider_readiness
from sopvision.decide.questions import build_questions, build_state
from sopvision.scene.types import HandView, SceneState, SlotView
from sopvision.settings import Settings

pytestmark = pytest.mark.decider


def test_decider_is_ready_on_the_gpu():
    r = decider_readiness(Settings().decider_url)
    assert r.ready, r.detail


async def test_real_decider_answers_an_sop_decision_like_jev(sop):
    s = Settings()
    scene = SceneState(812, 41.2, True, 0.99,
                       {"fan_bay_1": SlotView("installed", 0.97, "installed"), "fan_bay_2": SlotView("installed", 0.95, "installed"),
                        "fan_bay_3": SlotView("occluded", last="empty")}, [HandView("fan_bay_3")], ["fan"], None)
    statuses = {st.id: "done" if st.id in ("fan_1", "fan_2") else "active" if st.id == "fan_3" else "pending" for st in sop.steps}
    qs = build_questions(sop, "fan_3")
    client = JevClient("local", s.decider_url, DECIDER_MODEL, timeout=s.decider_timeout, provider="decider")
    try:
        a = await client.ask(build_state(sop, statuses, scene, []), qs)
    finally:
        await client.aclose()
    print(f"decider: {a}")
    assert a.provider == "decider"
    assert a.current_step in qs.current_step_options
    assert set(a.current_probabilities) == set(qs.current_step_options)
    assert sum(a.current_probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert 0 <= a.current_confidence <= 1 and 0 <= a.active_step_done <= 1 and 0 <= a.anomaly <= 1
