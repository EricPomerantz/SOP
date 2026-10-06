from sopvision.decide.types import Decision, JevAnswers
from sopvision.scene.types import SceneState, SlotView
from sopvision.sop.tracker import SopTracker
from sopvision.sop.types import Alert, StepUpdate

FANS = [f"fan_{i}" for i in range(1, 7)]


def decision(t, current="idle", conf=0.9, done=0.0, anomaly=0.0, active="fan_1", version=1):
    return Decision(version, t, active, JevAnswers(current, conf, {current: conf}, done, anomaly, 100.0))


def view(spec: str) -> SlotView:
    touched = spec.endswith("+touched")  # e.g. "installed+touched"
    spec = spec.removesuffix("+touched")
    if ":" in spec:  # e.g. "occluded:installed"
        state, last = spec.split(":")
        return SlotView(state, last=last, touched=True if state == "occluded" else touched)
    return SlotView(spec, 0.9, spec, touched)


def scene_with(t, slots: dict[str, str]) -> SceneState:
    return SceneState(1, t, True, 0.9, {k: view(v) for k, v in slots.items()}, [], [], None)


S0 = scene_with(0.0, {})


def with_timer(sop, seconds):
    return sop.model_copy(update={"decisions": sop.decisions.model_copy(update={"evidence_only_after": seconds})})


def alerts(items, kind=None):
    return [a for a in items if isinstance(a, Alert) and (kind is None or a.kind == kind)]


def test_activate_needs_hold(sop):
    tr = SopTracker(sop)
    tr.on_decision(decision(0.0, "fan_1"), S0)
    assert tr.statuses()["fan_1"] == "pending"
    out = tr.on_decision(decision(1.0, "fan_1"), S0)
    assert tr.statuses()["fan_1"] == "active" and StepUpdate(1.0, "fan_1", "active", None) in out


def test_low_confidence_breaks_the_streak(sop):
    tr = SopTracker(sop)
    tr.on_decision(decision(0.0, "fan_1"), S0)
    tr.on_decision(decision(0.5, "fan_1", conf=0.5), S0)
    tr.on_decision(decision(1.0, "fan_1"), S0)
    assert tr.statuses()["fan_1"] == "pending"


def test_activating_another_step_returns_previous_to_pending(sop):
    tr = SopTracker(sop)
    for t in (0.0, 1.0):
        tr.on_decision(decision(t, "fan_1"), S0)
    for t in (2.0, 3.0):
        tr.on_decision(decision(t, "fan_2"), S0)
    assert (tr.statuses()["fan_1"], tr.statuses()["fan_2"]) == ("pending", "active")
    assert tr.active_step_id() == "fan_2"


def test_complete_requires_jev_and_evidence(sop):
    tr = SopTracker(sop)
    for t in (0.0, 1.0):
        tr.on_decision(decision(t, "fan_1"), S0)
    empty, installed = scene_with(2.0, {"fan_bay_1": "empty"}), scene_with(4.0, {"fan_bay_1": "installed"})
    tr.on_decision(decision(2.0, "fan_1", done=0.9), empty)
    tr.on_decision(decision(3.0, "fan_1", done=0.9), empty)
    assert tr.statuses()["fan_1"] == "active"
    out = tr.on_decision(decision(4.0, "fan_1", done=0.9), installed)
    step = next(s for s in tr.steps() if s.id == "fan_1")
    assert (step.status, step.source, step.done_t, step.duration) == ("done", "jev", 4.0, 3.0)
    assert StepUpdate(4.0, "fan_1", "done", "jev") in out
    assert tr.active_step_id() == "fan_2"


def test_evidence_only_after_three_seconds(sop):
    tr = SopTracker(with_timer(sop, 3.0))
    tr.on_scene(scene_with(10.0, {"fan_bay_1": "installed"}))
    assert tr.on_scene(scene_with(12.9, {"fan_bay_1": "occluded:installed"})) == []
    out = tr.on_scene(scene_with(13.0, {"fan_bay_1": "installed"}))
    step = next(s for s in tr.steps() if s.id == "fan_1")
    assert (step.status, step.source, step.duration) == ("done", "evidence", None)
    assert StepUpdate(13.0, "fan_1", "done", "evidence") in out


def test_evidence_timer_resets_when_evidence_breaks(sop):
    tr = SopTracker(with_timer(sop, 3.0))
    tr.on_scene(scene_with(10.0, {"fan_bay_1": "installed"}))
    tr.on_scene(scene_with(11.0, {"fan_bay_1": "empty"}))
    tr.on_scene(scene_with(12.0, {"fan_bay_1": "installed"}))
    tr.on_scene(scene_with(14.5, {"fan_bay_1": "installed"}))
    assert tr.statuses()["fan_1"] == "pending"


def test_order_violation_strict(sop):
    tr = SopTracker(sop)
    tr.on_scene(scene_with(0.0, {"fan_bay_2": "installed"}))
    out = tr.on_scene(scene_with(3.0, {"fan_bay_2": "installed"}))
    (a,) = alerts(out, "order_violation")
    assert (a.level, a.message, a.step_id, a.data) == ("violation", "fan_2 completed before fan_1", "fan_2", {"missing": ["fan_1"]})


def test_order_modes_any_and_groups(sop):
    tr = SopTracker(sop.model_copy(update={"order": "any"}))
    tr.on_scene(scene_with(0.0, {"fan_bay_2": "installed"}))
    assert alerts(tr.on_scene(scene_with(3.0, {"fan_bay_2": "installed"}))) == []

    grouped = sop.model_copy(update={"order": "groups", "groups": [FANS, ["psu_1", "psu_2"], ["cover"]]})
    tr = SopTracker(grouped)
    tr.on_scene(scene_with(0.0, {"fan_bay_3": "installed", "psu_bay_1": "installed"}))
    out = tr.on_scene(scene_with(3.0, {"fan_bay_3": "installed", "psu_bay_1": "installed"}))
    assert [a.message for a in alerts(out, "order_violation")] == [
        "psu_1 completed before fan_1, fan_2, fan_4, fan_5, fan_6"]


def test_part_removed_needs_observed_empty(sop):
    tr = SopTracker(sop)
    tr.on_scene(scene_with(0.0, {"fan_bay_3": "installed"}))
    tr.on_scene(scene_with(3.0, {"fan_bay_3": "installed"}))
    assert tr.statuses()["fan_3"] == "done"
    for t in (4.0, 6.0, 9.0):
        assert alerts(tr.on_scene(scene_with(t, {"fan_bay_3": "occluded:installed"})), "part_removed") == []
    tr.on_scene(scene_with(10.0, {"fan_bay_3": "empty"}))
    assert alerts(tr.on_scene(scene_with(10.9, {"fan_bay_3": "empty"})), "part_removed") == []
    out = tr.on_scene(scene_with(11.0, {"fan_bay_3": "empty"}))
    (a,) = alerts(out, "part_removed")
    assert (a.level, a.message, a.step_id) == ("violation", "fan_3: part removed after the step was completed", "fan_3")
    assert tr.statuses()["fan_3"] == "pending" and StepUpdate(11.0, "fan_3", "pending", None) in out


def test_off_sop_and_anomaly_once_per_streak(sop):
    tr = SopTracker(sop)
    s = scene_with(0.0, {"fan_bay_1": "empty"})
    fired = []
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        fired += tr.on_decision(decision(t, "off_sop", anomaly=0.9), s)
    assert [a.kind for a in alerts(fired)] == ["off_sop", "anomaly"]
    off, anomaly = alerts(fired)
    assert (off.level, off.message, off.t) == ("warning", "Off-SOP activity", 1.0)
    assert anomaly.message == "Something looks inconsistent with the SOP" and anomaly.data == {"scene": s.to_jev_dict()}
    tr.on_decision(decision(2.5, "idle"), s)
    again = tr.on_decision(decision(3.0, "off_sop"), s) + tr.on_decision(decision(4.0, "off_sop"), s)
    assert [a.kind for a in alerts(again)] == ["off_sop"]


def test_incomplete_lists_missing(sop):
    (a,) = SopTracker(sop).end_run(60.0)
    assert (a.level, a.kind) == ("violation", "sop_incomplete")
    assert a.message == "SOP incomplete: missing fan_1, fan_2, fan_3, fan_4, fan_5, fan_6, psu_1, psu_2, cover"


def test_complete_run_has_no_incomplete_alert(sop):
    tr = SopTracker(sop.model_copy(update={"order": "any"}))
    all_in = {f"fan_bay_{i}": "installed" for i in range(1, 7)} | {"psu_bay_1": "installed", "psu_bay_2": "installed",
                                                                  "cover": "installed"}
    tr.on_scene(scene_with(0.0, all_in))
    tr.on_scene(scene_with(3.0, all_in))
    assert tr.end_run(10.0) == []


def test_jev_offline_alert(sop):
    tr = SopTracker(sop)
    (a,) = tr.set_jev_online(False, 5.0)
    assert (a.t, a.level, a.kind, a.message) == (5.0, "warning", "jev_offline", "Jev offline — using evidence only")
    assert tr.set_jev_online(True, 9.0) == []


def test_active_step_falls_back_to_first_pending(sop):
    tr = SopTracker(sop)
    assert tr.active_step_id() == "fan_1"
    tr.on_scene(scene_with(0.0, {"fan_bay_1": "installed"}))
    tr.on_scene(scene_with(3.0, {"fan_bay_1": "installed"}))
    assert tr.active_step_id() == "fan_2"


ALL_BEFORE_COVER = {f"fan_bay_{i}": "installed" for i in range(1, 7)} | {"psu_bay_1": "installed", "psu_bay_2": "installed"}


def tracker_with_all_but_cover(sop):
    tr = SopTracker(sop)
    tr.on_scene(scene_with(0.0, ALL_BEFORE_COVER))
    tr.on_scene(scene_with(3.0, ALL_BEFORE_COVER))
    assert tr.active_step_id() == "cover"
    return tr


def test_hands_off_step_waits_until_hands_leave(sop):
    tr = tracker_with_all_but_cover(sop)
    for t in (10.0, 12.0, 20.0):  # cover in place but being latched for 10 s
        assert tr.on_scene(scene_with(t, ALL_BEFORE_COVER | {"cover": "installed+touched"})) == []
    tr.on_scene(scene_with(21.0, ALL_BEFORE_COVER | {"cover": "installed"}))
    assert tr.statuses()["cover"] == "pending"
    out = tr.on_scene(scene_with(21.5, ALL_BEFORE_COVER | {"cover": "installed"}))
    assert StepUpdate(21.5, "cover", "done", "evidence") in out


def test_jev_cannot_complete_a_hands_off_step_while_touched(sop):
    tr = tracker_with_all_but_cover(sop)
    touched = scene_with(10.0, ALL_BEFORE_COVER | {"cover": "installed+touched"})
    for t in (10.0, 11.0, 12.0):
        tr.on_decision(decision(t, "cover", done=0.95, active="cover"), touched)
    assert tr.statuses()["cover"] != "done"
    tr.on_decision(decision(13.0, "cover", done=0.95, active="cover"), scene_with(13.0, ALL_BEFORE_COVER | {"cover": "installed"}))
    assert tr.statuses()["cover"] == "done"


def test_ordinary_steps_ignore_touching(sop):
    tr = SopTracker(sop)
    tr.on_scene(scene_with(0.0, {"fan_bay_1": "installed+touched"}))
    tr.on_scene(scene_with(1.0, {"fan_bay_1": "installed+touched"}))
    assert tr.statuses()["fan_1"] == "done"


def thresholds(sop, **kw):
    return sop.model_copy(update={"decisions": sop.decisions.model_copy(update=kw)})


def test_low_step_threshold_activates_without_firing_anomalies(sop):
    # A model whose step confidence runs low (Kev) gets its own threshold; its anomaly scores keep the default
    tr = SopTracker(thresholds(sop, step_confidence=0.3))
    out = []
    for t in (0.0, 1.0):
        out += tr.on_decision(decision(t, "fan_1", conf=0.35, anomaly=0.35), S0)
    assert tr.statuses()["fan_1"] == "active" and alerts(out, "anomaly") == []


def test_anomaly_threshold_is_its_own(sop):
    tr = SopTracker(thresholds(sop, anomaly_confidence=0.95))
    fired = []
    for t in (0.0, 1.0, 2.0):
        fired += tr.on_decision(decision(t, "idle", anomaly=0.9), S0)
    assert alerts(fired, "anomaly") == []


def test_done_threshold_is_its_own(sop):
    tr = SopTracker(thresholds(sop, done_confidence=0.5))
    for t in (0.0, 1.0):
        tr.on_decision(decision(t, "fan_1"), S0)
    installed = scene_with(2.0, {"fan_bay_1": "installed"})
    tr.on_decision(decision(2.0, "fan_1", done=0.6), installed)
    out = tr.on_decision(decision(3.0, "fan_1", done=0.6), installed)
    assert StepUpdate(3.0, "fan_1", "done", "jev") in out


def test_offline_alert_names_the_decision_model(sop):
    (a,) = SopTracker(sop, decider="Kev").set_jev_online(False, 5.0)
    assert (a.kind, a.message) == ("jev_offline", "Kev offline — using evidence only")
