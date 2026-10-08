from dataclasses import asdict

import pytest

from sopvision.dataset import Segment
from sopvision.decide.compare import compare_answers, comparison_record
from sopvision.decide.questions import QuestionSet
from sopvision.decide.types import JevAnswers, ProviderResult
from sopvision.evaluation.decisions import expected_steps, label, percentiles, run_number, summarize

SEGMENTS = [Segment("fan_1", 0.0, 10.0), Segment("off_sop", 10.0, 20.0), Segment("fan_2", 20.0, 30.0)]
QS = QuestionSet({"fan_1": "a", "fan_2": "b", "idle": "i", "off_sop": "o"}, "fan_1", "done?")


def answers(step, conf=0.9, done=0.1, anomaly=0.05, provider="jev"):
    return JevAnswers(step, conf, {step: conf, "idle": 1 - conf}, done, anomaly, 100.0, provider)


def record(t, jev, decider, wait=None):
    rec = comparison_record("r", t, 1, {"s": 1}, QS, ProviderResult("jev", jev, None, 180.0),
                            ProviderResult("decider", decider, None if decider else "timeout", 30.0), 0.8)
    return rec | {"wait_ms": wait}


def test_compare_answers_raw_and_actionable():
    c = compare_answers(answers("fan_1", 0.9, done=0.85), answers("fan_1", 0.6, done=0.5, provider="decider"), 0.8)
    assert c["same_current_step"] is True and c["current_confidence_diff"] == pytest.approx(-0.3)
    assert c["same_actionable"] == {"current_step": False, "active_step_done": False, "anomaly": True}
    assert c["current_probabilities_tv"] == pytest.approx(0.3)


def test_failed_provider_has_no_comparison_but_keeps_the_request():
    rec = record(5.0, answers("fan_1"), None)
    assert rec["comparison"] is None and rec["decider"]["error"] == "timeout"
    assert rec["request"] == {"state": {"s": 1}, "questions": asdict(QS)}


def test_expected_steps_skip_boundaries_and_accept_idle_for_off_sop():
    assert expected_steps(SEGMENTS, 5.0) == ["fan_1"]
    assert expected_steps(SEGMENTS, 10.5) is None  # within the grace after a boundary
    assert expected_steps(SEGMENTS, 15.0) == ["idle", "off_sop"]
    assert expected_steps(SEGMENTS, 99.0) is None


def test_run_number_from_source_ids():
    assert run_number("test:Install_12") == 12 and run_number("file:/data/raw/Install_13.MP4") == 13
    assert run_number("webcam:0") is None


def test_labels_only_what_is_known():
    rec = record(5.0, answers("fan_1"), answers("fan_1"))
    assert label(rec, None, False, "live")["ground_truth"] is None
    assert label(rec, SEGMENTS, True, "test:Install_12")["ground_truth"] == {
        "current_step": ["fan_1"], "anomaly": False, "source": "test:Install_12"}


def test_summary_separates_agreement_from_accuracy():
    recs = [label(r, SEGMENTS, True, "t") for r in (
        record(5.0, answers("fan_1"), answers("fan_2", provider="decider"), wait=20.0),  # disagree, jev right
        record(15.0, answers("fan_1"), answers("idle", provider="decider"), wait=40.0),  # disagree, decider right
        record(25.0, answers("fan_1", anomaly=0.9), answers("fan_2", provider="decider")),  # disagree, decider right
        record(26.0, answers("fan_2"), None),  # decider failed
    )]
    s = summarize(recs, 0.8)
    jev, dec = s["providers"]["jev"], s["providers"]["decider"]
    assert s["agreement"]["same_current_step"] == 0.0 and s["agreement"]["both_answered"] == 3
    assert jev["current_step"]["accuracy"] == 0.5 and dec["current_step"]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert (dec["answered"], dec["errors"]) == (3, 1)
    assert jev["off_sop"]["false_negative_rate"] == 1.0 and dec["off_sop"]["false_negative_rate"] == 0.0
    assert jev["anomaly"]["false_positive_rate"] == 0.25 and dec["anomaly"]["false_positive_rate"] == 0.0
    assert jev["e2e_ms"] == {"n": 2, "p50": 200.0, "p95": 220.0, "max": 220.0, "mean": 210.0}
    assert jev["current_step"]["mean_confidence_when_wrong"] == 0.9


def test_percentiles_nearest_rank():
    p = percentiles([float(x) for x in range(1, 101)])
    assert (p["p50"], p["p95"], p["max"]) == (50.0, 95.0, 100.0)
    assert percentiles([]) is None
