from pathlib import Path

from sopvision.dataset import ClipInfo, RunIndex
from sopvision.evaluation.metrics import EvalReport
from sopvision.evaluation.run_eval import latency_stats, summary_lines, video_from_result
from sopvision.pipeline import FrameRecord, LockstepResult
from sopvision.sop.types import Alert, StepState


def run12():
    clips = [ClipInfo(Path("a"), 10, 12, 1, 1, 5.0), ClipInfo(Path("b"), 1, 12, 1, 2, 10.0), ClipInfo(Path("c"), 2, 12, 1, 3, 6.0),
             ClipInfo(Path("d"), 10, 12, 2, 4, 2.0)]
    return RunIndex(12, "test", clips, Path("raw"), 23.0)


def test_video_from_result_scores_steps_and_off_sop():
    res = LockstepResult(
        frames=[FrameRecord(0.0, {"fan_bay_1": "empty"}), FrameRecord(0.033, {"fan_bay_1": "installed"})],
        steps=[StepState("fan_2", "done", 15.2, 21.4, 6.2, "evidence"), StepState("fan_1", "done", 5.5, 15.9, 10.4, "jev")],
        alerts=[Alert(3.0, "warning", "off_sop", "Off-SOP activity"),
                Alert(22.5, "violation", "sop_incomplete", "SOP incomplete: missing cover", data={"missing": ["cover"]})],
        latencies_ms=[100.0, 200.0, 300.0], decisions=[(0.5, "idle"), (12.0, "fan_1"), (22.0, "fan_3")])
    v = video_from_result("test:Install_12", "test", res, run12(), [])
    assert v.step_order == ["fan_1", "fan_2"]
    assert v.steps["fan_1"] == {"status": "done", "done_t": 15.9, "error": 0.9, "source": "jev"}
    assert round(v.steps["fan_2"]["error"], 3) == 0.4
    assert v.offsop == [True, False]  # Jev named fan_3 during the closing off-SOP segment
    assert v.offsop_jev_answers == [0, 1]  # the 0.5 s answer is inside the opening segment's grace
    assert v.alerts[1]["data"] == {"missing": ["cover"]}
    assert v.frames[1] == {"t": 0.033, "slots": {"fan_bay_1": "installed"}}
    assert v.latency_ms == {"p50": 200.0, "p90": 300.0, "max": 300.0}


def test_scenario_video_has_no_ground_truth():
    res = LockstepResult([], [StepState("fan_1", "pending", None, None, None, None)], [], [], [])
    v = video_from_result("scenario:fan_removed", "scenario", res, None, [{"kind": "part_removed", "step_id": "fan_3"}])
    assert v.offsop == [] and v.steps["fan_1"]["error"] is None and v.expected_alerts[0]["kind"] == "part_removed"


def test_latency_stats_empty():
    assert latency_stats([]) == {"p50": 0.0, "p90": 0.0, "max": 0.0}


def test_summary_says_how_steps_completed_and_how_much_jev_answered_off_sop():
    res = LockstepResult([], [StepState("fan_1", "done", 5.5, 15.9, 10.4, "evidence"),
                              StepState("fan_2", "done", 15.2, 21.4, 6.2, "jev")], [], [100.0], [(22.0, "idle")])
    v = video_from_result("test:Install_12", "test", res, run12(), [])
    report = EvalReport(env={}, videos=[v, v.model_copy(update={"id": "test:Install_13"})],
                        acceptance={"c3": True}, passed=True)
    lines = summary_lines(report)
    assert "test-run completions: visual evidence 2, Jev 2" in lines
    assert "off-SOP windows: 4; Jev answers judged per window: [0, 1, 0, 1]" in lines
    assert ("note: Jev gave no answer to judge in 2 of 4 off-SOP windows; c3's Jev check is vacuous there"
            in lines)
