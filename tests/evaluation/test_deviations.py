import copy

from sopvision.evaluation.deviations import ELIGIBLE, GENERATORS, expected

PLACE = [f"place_{k}" for k in range(1, 6)]
TORQUE = ["torque_1", "torque_3", "torque_5", "torque_2", "torque_4"]


def lug_record(lug_sop, done: set[str], t=40.0):
    slots = {}
    for k in range(1, 6):
        placed = f"place_{k}" in done
        slots[f"hole_{k}"] = {"state": "installed" if placed else "empty", "conf": 0.9}
        if f"torque_{k}" in done:
            slots[f"hole_{k}"]["torqued"] = True
    steps = [{"id": s.id, "status": "done" if s.id in done else "pending"} for s in lug_sop.steps]
    return {"request": {"state": {"sop": "x", "steps": steps,
                                  "scene": {"t": t, "slots": slots, "hands": [], "held": [], "tools": []},
                                  "recent_events": [f"{t - 1:.1f}s hand entered hole_1"]},
                        "questions": {"current_step_options": {}, "active_step_id": "", "active_step_question": ""}}}


def test_torque_phase_out_of_order_work_puts_the_wrench_on_a_later_stud(lug_sop):
    rec = lug_record(lug_sop, set(PLACE))
    assert expected(rec, lug_sop) == "torque_1"
    truth = GENERATORS["out_of_order_work"](rec, lug_sop)
    sc = rec["request"]["state"]["scene"]
    assert truth == {"current_step": ["torque_5"], "anomaly": None, "active_step_done": None}
    assert sc["tools"] == [{"tool": "torque_wrench", "near": "hole_5"}] and sc["slots"]["hole_5"]["state"] == "installed"
    assert rec["request"]["state"]["recent_events"][-1].endswith("torque_wrench at hole_5")


def test_torque_phase_out_of_order_install_marks_a_later_hole_torqued(lug_sop):
    rec = lug_record(lug_sop, set(PLACE) | {"torque_1"})
    truth = GENERATORS["out_of_order_install"](rec, lug_sop)
    assert truth["anomaly"] is True
    assert rec["request"]["state"]["scene"]["slots"]["hole_2"]["torqued"] is True  # torque_2 while torque_3 is next


def test_placing_phase_work_ahead_is_torquing_before_all_nuts_are_on(lug_sop):
    rec = lug_record(lug_sop, {"place_1", "place_2"})
    assert expected(rec, lug_sop) == "place_3"  # any order: the first allowed one
    assert GENERATORS["out_of_order_work"](rec, lug_sop)["current_step"] == ["torque_3"]


def test_done_part_removed_takes_a_placed_nut_off(lug_sop):
    rec = lug_record(lug_sop, {"place_1", "place_2"})
    assert ELIGIBLE["done_part_removed"](rec, lug_sop)
    assert GENERATORS["done_part_removed"](rec, lug_sop)["anomaly"] is True
    sc = rec["request"]["state"]["scene"]
    assert sc["slots"]["hole_2"] == {"state": "empty", "conf": 0.92} and sc["held"] == ["lug_bolt_blue"]


def test_idle_clears_hands_and_tools(lug_sop):
    rec = lug_record(lug_sop, set(PLACE))
    rec["request"]["state"]["scene"]["tools"] = [{"tool": "torque_wrench", "near": "hole_1"}]
    assert GENERATORS["idle_bench"](rec, lug_sop)["current_step"] == ["idle", "off_sop"]
    assert rec["request"]["state"]["scene"]["tools"] == []


def test_nothing_ahead_when_everything_is_done(lug_sop):
    rec = lug_record(lug_sop, set(PLACE) | set(TORQUE))
    assert not ELIGIBLE["out_of_order_work"](rec, lug_sop) and not ELIGIBLE["out_of_order_install"](rec, lug_sop)


def test_generators_edit_only_their_copy(lug_sop):
    rec = lug_record(lug_sop, set(PLACE))
    before = copy.deepcopy(rec)
    GENERATORS["out_of_order_work"](copy.deepcopy(rec), lug_sop)
    assert rec == before
