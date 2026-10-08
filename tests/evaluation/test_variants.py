from sopvision.decide.questions import eligible_steps, progress, with_progress
from sopvision.evaluation.variants import VARIANTS, a1_progress, baseline, expected_steps

STEPS = ["fan_1", "fan_2", "fan_3", "fan_4", "fan_5", "fan_6", "psu_1", "psu_2", "cover"]


def record(done: dict[str, float], t: float = 30.0) -> dict:
    statuses = [{"id": s, "status": "done" if s in done else "pending"} for s in STEPS]
    return {"request": {"state": {"sop": "x", "steps": statuses, "scene": {"t": t}, "recent_events": []},
                        "questions": {"current_step_options": {"fan_1": "a"}, "active_step_id": "fan_3",
                                      "active_step_question": "done?"}},
            "tracker": {"done_t": done}}


def test_strict_order_expects_the_first_pending_step(sop):
    assert eligible_steps(sop, {s: "done" if s in ("fan_1", "fan_2") else "pending" for s in STEPS}) == ["fan_3"]


def test_other_orders_list_every_eligible_step(sop):
    anyorder = sop.model_copy(update={"order": "any"})
    assert eligible_steps(anyorder, {s: "done" if s == "fan_1" else "pending" for s in STEPS}) == STEPS[1:]


def test_progress_names_expected_and_last_completed(sop):
    p = progress(sop, {s: "done" if s in ("fan_1", "fan_2") else "pending" for s in STEPS}, {"fan_1": 10.0, "fan_2": 21.4}, 30.0)
    assert p == {"expected_step": "fan_3", "last_completed": {"step": "fan_2", "seconds_ago": 8.6}}
    assert progress(sop, {s: "pending" for s in STEPS}, {}, 5.0) == {"expected_step": "fan_1", "last_completed": None}


def test_a1_only_adds_progress_after_the_steps(sop):
    rec = record({"fan_1": 10.0, "fan_2": 21.4})
    state, qs = a1_progress(rec, sop)
    assert list(state) == ["sop", "steps", "progress", "scene", "recent_events"]
    assert {k: v for k, v in state.items() if k != "progress"} == rec["request"]["state"]
    assert qs == baseline(rec, sop)[1]
    assert expected_steps(rec, sop) == ["fan_3"]


def test_variants_never_modify_the_frozen_record(sop):
    rec = record({"fan_1": 10.0})
    before = repr(rec)
    for variant in VARIANTS.values():
        variant(rec, sop)
    assert repr(rec) == before


def test_with_progress_keeps_order():
    assert list(with_progress({"a": 1, "steps": [], "b": 2}, {"x": 1})) == ["a", "steps", "progress", "b"]


def test_a6_moves_only_the_steps_to_the_end(sop):
    rec = record({"fan_1": 10.0})
    state, qs = VARIANTS["a6_steps_last"](rec, sop)
    assert list(state) == ["sop", "scene", "recent_events", "steps"]
    assert state == rec["request"]["state"] and qs == baseline(rec, sop)[1]  # same content, other order


def test_a7_sorts_only_the_options(sop):
    rec = record({})
    rec["request"]["questions"]["current_step_options"] = {"fan_2": "b", "cover": "c", "fan_1": "a"}
    state, qs = VARIANTS["a7_options_sorted"](rec, sop)
    assert list(qs.current_step_options) == ["cover", "fan_1", "fan_2"]
    assert list(state) == list(rec["request"]["state"]) and qs.active_step_question == "done?"


def test_a8_puts_steps_then_progress_last(sop):
    state, _ = VARIANTS["a8_steps_progress_last"](record({"fan_1": 10.0}), sop)
    assert list(state) == ["sop", "scene", "recent_events", "steps", "progress"]
    assert state["progress"]["expected_step"] == "fan_2"
