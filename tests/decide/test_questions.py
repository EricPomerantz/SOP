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
