from sopvision.scene.types import SceneState, SlotView, ToolView
from sopvision.sop.torque import AttributedTorque, TorqueAttributor, TorqueResult
from sopvision.sop.tracker import SopTracker

HOLES = {f"hole_{i}" for i in range(1, 6)}


def frame(t, *near, slots=None):
    return SceneState(1, t, True, 0.9, slots or {}, [], [], None, [ToolView("torque_wrench", n) for n in near])


def watch(attr, frames):
    for f in frames:
        attr.observe(f)


# -- attribution ---------------------------------------------------------------------------------------------------
def test_waits_for_the_frames_after_the_result_then_takes_the_majority():
    a = TorqueAttributor(HOLES)
    watch(a, [frame(4.5 + i / 10, "hole_3") for i in range(6)])  # 4.5 .. 5.0
    a.add(TorqueResult(5.0, True, 120.0))
    assert a.resolve(5.0) == []  # the 0.3 s after it hasn't been seen yet
    watch(a, [frame(5.1, "hole_3"), frame(5.2, "hole_3"), frame(5.3, "hole_3")])
    (r,) = a.resolve(5.3)
    assert r.slot == "hole_3" and "hole_3" in r.reason


def test_wrench_moving_between_slots_is_unattributed():
    a = TorqueAttributor(HOLES)
    watch(a, [frame(4.5 + i / 10, "hole_1" if i % 2 else "hole_2") for i in range(9)])
    a.add(TorqueResult(5.0, True))
    (r,) = a.resolve(5.4)
    assert r.slot is None and "moved between" in r.reason


def test_no_wrench_seen_is_unattributed():
    a = TorqueAttributor(HOLES)
    watch(a, [frame(4.5 + i / 10) for i in range(9)])
    a.add(TorqueResult(5.0, True))
    (r,) = a.resolve(5.4)
    assert r.slot is None and "wasn't seen" in r.reason


def test_camera_delay_shifts_the_window():
    """Live: the result is timed at latest frame + delay, so the frames showing it arrive after it was received."""
    a = TorqueAttributor(HOLES)
    watch(a, [frame(10.0 + i / 10, "hole_4") for i in range(4)])  # frames received so far: wrench at hole 4
    a.add(TorqueResult(10.3 + 0.8, True))  # received at 10.3, camera 0.8 s behind
    assert a.resolve(10.3) == []
    watch(a, [frame(10.4 + i / 10, "hole_2") for i in range(12)])  # the frames that show the actual moment
    (r,) = a.resolve(11.5)
    assert r.slot == "hole_2"


def test_flush_resolves_everything():
    a = TorqueAttributor(HOLES)
    a.observe(frame(1.0, "hole_5"))
    a.add(TorqueResult(1.0, True))
    assert [r.slot for r in a.resolve(1.0, flush=True)] == ["hole_5"]


# -- judging -------------------------------------------------------------------------------------------------------
def bolt(state="installed", torqued=False):
    return SlotView(state, 0.9, state if state in ("installed", "empty") else None, False, torqued)


def judged(lug_sop, slot, ok=True, slots=None, reason="wrench at slot"):
    scene = frame(5.0, slots=slots or {f"hole_{i}": bolt() for i in range(1, 6)})
    return SopTracker(lug_sop).on_torque(AttributedTorque(TorqueResult(5.0, ok, 110.0), slot, reason), scene)


def test_accepted_torque_marks_the_slot(lug_sop):
    alerts, slot = judged(lug_sop, "hole_1")
    assert slot == "hole_1" and [(a.level, a.kind, a.message) for a in alerts] == [("info", "torque", "Torque OK on hole_1 (110 Nm)")]


def test_rejected_torques_raise_their_alert_and_mark_nothing(lug_sop):
    cases = {
        "torque_not_reached": judged(lug_sop, "hole_1", ok=False),
        "torque_unattributed": judged(lug_sop, None, reason="the wrench wasn't seen at any slot"),
        "torque_without_part": judged(lug_sop, "hole_2", slots={"hole_2": bolt("empty")}),
        "torque_repeated": judged(lug_sop, "hole_3", slots={"hole_3": bolt(torqued=True)}),
    }
    for kind, (alerts, slot) in cases.items():
        assert slot is None and [a.kind for a in alerts] == [kind], kind


def test_torqued_slot_completes_its_torque_step_at_once(lug_sop):
    tr = SopTracker(lug_sop)
    out = tr.on_scene(frame(3.0, slots={"hole_1": bolt(torqued=True)}))
    done = [u for u in out if getattr(u, "status", None) == "done"]
    assert [(u.step_id, u.source) for u in done] == [("torque_1", "torque")]


def test_torque_out_of_sequence_is_an_order_violation(lug_sop):
    tr = SopTracker(lug_sop)
    all_placed = {f"hole_{i}": bolt() for i in range(1, 6)}
    tr.on_scene(frame(0.0, slots=all_placed))
    tr.on_scene(frame(1.5, slots=all_placed))  # every bolt placed (1 s of evidence)
    out = tr.on_scene(frame(2.0, slots=all_placed | {"hole_3": bolt(torqued=True)}))  # hole 3 before hole 1
    (violation,) = [a for a in out if getattr(a, "kind", None) == "order_violation"]
    assert violation.message == "torque_3 completed before torque_1"


def test_read_torque_log(tmp_path):
    from sopvision.sop.torque import read_torque_log

    p = tmp_path / "torque.csv"
    p.write_text("time_s,ok,torque_nm\n13.0,1,110.5\n10.0,true,\n16.5,0,62\n", encoding="utf-8")
    assert read_torque_log(p) == [TorqueResult(10.0, True, None), TorqueResult(13.0, True, 110.5),
                                  TorqueResult(16.5, False, 62.0)]
