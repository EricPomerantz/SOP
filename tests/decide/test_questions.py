import json

from sopvision.decide.questions import build_questions, build_state
from sopvision.scene.types import SceneEvent, SceneState, SlotView


def scene(t=41.2):
    return SceneState(7, t, True, 0.99, {"fan_bay_1": SlotView("installed", 0.97, "installed")}, [], [], (0, 0, 10, 10))


def test_options_and_copy(sop):
    qs = build_questions(sop, "fan_1")
    assert list(qs.current_step_options)[-2:] == ["idle", "off_sop"] and len(qs.current_step_options) == 11
    assert qs.current_step_options["idle"] == "No one is working on the server"
    assert qs.current_step_options["off_sop"] == "Any action that is not one of the steps above."
    assert qs.current_step_options["fan_1"] == sop.step("fan_1").text
    assert qs.active_step_id == "fan_1"
    assert qs.active_step_question == ("Has this step been completed: Install the first fan by connecting the "
                                       "connector, then pressing the fan in place?")


def test_state_keeps_last_ten_events(sop):
    events = [SceneEvent(float(i), "hand_entered", zone="fan_bay_1") for i in range(15)]
    st = build_state(sop, {"fan_1": "done"}, scene(), events)
    assert st["sop"] == sop.description and st["steps"] == [{"id": "fan_1", "status": "done"}]
    assert len(st["recent_events"]) == 10 and st["recent_events"][0] == "5.0s hand entered fan_bay_1"
    assert st["scene"] == scene().to_jev_dict()
    assert "image" not in json.dumps(st) and "chassis_box" not in json.dumps(st)


def test_strict_sop_names_the_next_expected_step(sop):
    # The first step in SOP order that isn't done; on recorded test runs this lifted Jev's step accuracy from 60% to 78%
    st = build_state(sop, {"fan_1": "done", "fan_2": "active", "fan_3": "pending"}, scene(), [])
    assert st["next_expected_step"] == "fan_2"
    all_done = build_state(sop, {s.id: "done" for s in sop.steps}, scene(), [])
    assert all_done["next_expected_step"] is None


def test_unordered_sops_have_no_expected_step(sop):
    for order in ("any", "groups"):
        st = build_state(sop.model_copy(update={"order": order}), {"fan_1": "done"}, scene(), [])
        assert "next_expected_step" not in st
