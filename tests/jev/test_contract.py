import pytest

from sopvision.decide.jev_client import JevClient
from sopvision.decide.questions import build_questions, build_state
from sopvision.scene.types import HandView, SceneState, SlotView
from sopvision.settings import Settings

pytestmark = pytest.mark.jev


async def test_real_jev_answers_have_the_expected_shape(sop):
    s = Settings()
    scene = SceneState(812, 41.2, True, 0.99,
                       {"fan_bay_1": SlotView("installed", 0.97, "installed"), "fan_bay_2": SlotView("installed", 0.95, "installed"),
                        "fan_bay_3": SlotView("occluded", last="empty")}, [HandView("fan_bay_3")], ["fan"], None)
    statuses = {st.id: "done" if st.id in ("fan_1", "fan_2") else "active" if st.id == "fan_3" else "pending" for st in sop.steps}
    qs = build_questions(sop, "fan_3")
    client = JevClient(s.typesafe_api_key.get_secret_value(), s.typesafe_base_url, s.typesafe_model)
    try:
        a = await client.ask(build_state(sop, statuses, scene, []), qs)
    finally:
        await client.aclose()
    assert a.current_step in qs.current_step_options
    assert 0 <= a.current_confidence <= 1 and 0 <= a.active_step_done <= 1 and 0 <= a.anomaly <= 1
    assert set(a.current_probabilities) <= set(qs.current_step_options)
