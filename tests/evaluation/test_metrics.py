import pytest

from sopvision.dataset import Segment
from sopvision.evaluation.compare import compare_reports
from sopvision.evaluation.metrics import EvalReport, EvalVideo, check_acceptance, offsop_flagged, step_errors
from sopvision.sop.types import Alert, StepState

STEPS = ["fan_1", "fan_2", "fan_3", "fan_4", "fan_5", "fan_6", "psu_1", "psu_2", "cover"]
SLOTS = ["fan_bay_1", "fan_bay_2", "cover"]
SEGS = [Segment("off_sop", 0, 5), Segment("fan_1", 5, 15), Segment("fan_2", 15, 21), Segment("off_sop", 21, 23)]


def steps_done_at(times: dict[str, float]) -> list[StepState]:
    return [StepState(s, "done", None, times[s], None, "jev") if s in times else StepState(s, "pending", None, None, None, None)
            for s in ["fan_1", "fan_2"]]


def test_step_errors_and_missing_steps():
    errs = step_errors(SEGS, steps_done_at({"fan_1": 15.5}))
    assert errs == {"fan_1": pytest.approx(0.5), "fan_2": None}


def test_offsop_requires_alert_inside_segment():
    def alert(t):
        return Alert(t, "warning", "off_sop", "Off-SOP activity")

    assert offsop_flagged(SEGS, [alert(3.0), alert(24.4)]) == [True, True]
    assert offsop_flagged(SEGS, [alert(24.6)]) == [False, False]
    assert offsop_flagged(SEGS, [Alert(3.0, "warning", "anomaly", "x")]) == [False, False]


def video(vid="test:Install_12", kind="test", shift=0.0, kinds=("off_sop",), corrupt=0.0, done=STEPS, extra=None):
    steps = {s: {"status": "done" if s in done else "pending", "done_t": 10.0 + i + (shift if s == "fan_2" else 0) if s in done else None,
                 "error": 0.5 if s in done else None, "source": "jev"} for i, s in enumerate(STEPS)}
    frames = [{"t": i / 30, "slots": {s: "installed" for s in SLOTS}} for i in range(1000)]
    for i in range(int(corrupt * len(frames) * len(SLOTS))):
        frames[i // len(SLOTS)]["slots"][SLOTS[i % len(SLOTS)]] = "empty"
    alerts = [{"t": 2.0, "level": "warning", "kind": k, "message": "m", "step_id": None, "data": {}} for k in kinds]
    return EvalVideo(id=vid, kind=kind, step_order=[s for s in STEPS if s in done], steps=steps, alerts=alerts + (extra or []),
                     frames=frames, offsop=[True, True] if kind == "test" else [],
                     latency_ms={"p50": 150.0, "p90": 190.0, "max": 400.0}, expected_alerts=[])


def scenario(name, expected, alerts):
    v = video(f"scenario:{name}", "scenario", kinds=(), done=STEPS[:3], extra=alerts)
    return v.model_copy(update={"expected_alerts": expected})


def report(**kw):
    vids = [video("test:Install_12", **kw), video("test:Install_13"),
            scenario("incomplete_run", [{"kind": "sop_incomplete", "missing": ["fan_5"]}],
                     [{"t": 40, "level": "violation", "kind": "sop_incomplete", "message": "m", "step_id": None,
                       "data": {"missing": ["fan_5"]}}]),
            scenario("fan_removed", [{"kind": "part_removed", "step_id": "fan_3"}],
                     [{"t": 30, "level": "violation", "kind": "part_removed", "message": "m", "step_id": "fan_3", "data": {}}])]
    r = EvalReport(env={}, videos=vids, acceptance={}, passed=False)
    acc = check_acceptance(r)
    return r.model_copy(update={"acceptance": acc, "passed": all(acc.values())})


def test_good_report_passes_every_criterion():
    r = report()
    assert r.acceptance == {"c1": True, "c2": True, "c3": True, "c4": True, "c5": True} and r.passed


def test_acceptance_fails_on_missing_step():
    r = report(done=STEPS[:-1])
    assert r.acceptance["c1"] is False and r.acceptance["c2"] is False and not r.passed


def test_acceptance_fails_on_violation_in_a_test_video():
    bad = {"t": 5, "level": "violation", "kind": "order_violation", "message": "m", "step_id": "fan_2", "data": {}}
    assert report(extra=[bad]).acceptance["c4"] is False


def test_scenario_must_raise_expected_alert():
    r = report()
    vids = r.videos[:3] + [r.videos[3].model_copy(update={"alerts": []})]
    assert check_acceptance(r.model_copy(update={"videos": vids}))["c5"] is False


def test_parity_identical_reports_pass():
    assert compare_reports(report(), report()) == []


def test_parity_step_time_tolerance():
    assert compare_reports(report(), report(shift=0.4)) == []
    assert any("fan_2" in m for m in compare_reports(report(), report(shift=0.6)))


def test_parity_slot_agreement_threshold():
    assert any("slot agreement 98.9%" in m for m in compare_reports(report(), report(corrupt=0.011)))


def test_parity_alert_kinds_must_match():
    assert any("alert kinds differ" in m for m in compare_reports(report(), report(kinds=("off_sop", "anomaly"))))


def test_parity_requires_both_to_pass():
    failed = report().model_copy(update={"passed": False})
    assert any("did not pass acceptance" in m for m in compare_reports(report(), failed))
