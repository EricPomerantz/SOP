from pathlib import Path

from sopvision.dataset import ClipInfo, RunIndex
from sopvision.evaluation.run_eval import latency_stats, video_from_result
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
        latencies_ms=[100.0, 200.0, 300.0])
    v = video_from_result("test:Install_12", "test", res, run12(), [])
    assert v.step_order == ["fan_1", "fan_2"]
    assert v.steps["fan_1"] == {"status": "done", "done_t": 15.9, "error": 0.9, "source": "jev"}
    assert round(v.steps["fan_2"]["error"], 3) == 0.4
    assert v.offsop == [True, False]
    assert v.alerts[1]["data"] == {"missing": ["cover"]}
    assert v.frames[1] == {"t": 0.033, "slots": {"fan_bay_1": "installed"}}
    assert v.latency_ms == {"p50": 200.0, "p90": 300.0, "max": 300.0}


def test_scenario_video_has_no_ground_truth():
    res = LockstepResult([], [StepState("fan_1", "pending", None, None, None, None)], [], [])
    v = video_from_result("scenario:fan_removed", "scenario", res, None, [{"kind": "part_removed", "step_id": "fan_3"}])
    assert v.offsop == [] and v.steps["fan_1"]["error"] is None and v.expected_alerts[0]["kind"] == "part_removed"


def test_latency_stats_empty():
    assert latency_stats([]) == {"p50": 0.0, "p90": 0.0, "max": 0.0}
