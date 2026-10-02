"""Evaluation report model and the spec's acceptance criteria 1–5 (spec §2)."""

import statistics
from typing import Literal

from pydantic import BaseModel

from sopvision.dataset import Segment
from sopvision.sop.types import StepState

VIOLATIONS_FORBIDDEN_ON_TEST = {"order_violation", "part_removed", "sop_incomplete"}
MEDIAN_ERROR_MAX = 2.0
WORST_ERROR_MAX = 4.0
BOUNDARY_GRACE = 1.0


class EvalVideo(BaseModel):
    id: str
    kind: Literal["test", "scenario"]
    step_order: list[str]  # steps in the order they were completed
    steps: dict[str, dict]  # id -> {status, done_t, error, source}, in SOP order
    alerts: list[dict]
    frames: list[dict]  # [{t, slots: {slot: state}}] — every frame, for Docker parity
    offsop: list[bool]  # per ground-truth off-SOP segment: handled cleanly (see offsop_clean)?
    offsop_jev_answers: list[int] = []  # per off-SOP segment: Jev answers offsop_clean judged (0 → its Jev check is vacuous)
    latency_ms: dict[str, float]
    expected_alerts: list[dict] = []  # scenario videos: what they must raise


class EvalReport(BaseModel):
    env: dict
    videos: list[EvalVideo]
    acceptance: dict[str, bool]
    passed: bool


def step_errors(segments: list[Segment], steps: list[StepState]) -> dict[str, float | None]:
    """Completion time minus the end of that step's ground-truth segment (None when never completed)."""
    ends = {s.step_id: s.end for s in segments if s.step_id != "off_sop"}
    return {s.id: (s.done_t - ends[s.id] if s.status == "done" and s.done_t is not None else None)
            for s in steps if s.id in ends}


def offsop_clean(segments: list[Segment], steps: list[StepState], decisions: list[tuple[float, str]],
                 grace: float = BOUNDARY_GRACE) -> list[bool]:
    """Per ground-truth off-SOP segment: no step progressed and Jev called it idle or off-SOP (criterion 3).

    The step that ended where the segment starts may still complete inside it (that's step timing, scored by
    criterion 2), and Jev answers in the segment's first `grace` seconds are about that step, so they're skipped.
    """
    results, previous = [], None
    for seg in segments:
        if seg.step_id != "off_sop":
            previous = seg.step_id
            continue
        no_progress = all(s.id == previous or s.status != "done" or s.done_t is None or not seg.start <= s.done_t <= seg.end
                          for s in steps)
        jev_ok = all(answer in ("idle", "off_sop") for answer in _judged(seg, decisions, grace))
        results.append(no_progress and jev_ok)
    return results


def _judged(seg: Segment, decisions: list[tuple[float, str]], grace: float) -> list[str]:
    return [answer for t, answer in decisions if seg.start + grace <= t <= seg.end]


def offsop_jev_answers(segments: list[Segment], decisions: list[tuple[float, str]],
                       grace: float = BOUNDARY_GRACE) -> list[int]:
    """Per ground-truth off-SOP segment, how many Jev answers criterion 3 actually judged."""
    return [len(_judged(seg, decisions, grace)) for seg in segments if seg.step_id == "off_sop"]


def _has(alerts: list[dict], expected: dict) -> bool:
    for a in alerts:
        if a["kind"] != expected["kind"]:
            continue
        if "step_id" in expected and a.get("step_id") != expected["step_id"]:
            continue
        if "missing" in expected and a.get("data", {}).get("missing") != expected["missing"]:
            continue
        return True
    return False


def check_acceptance(report: EvalReport) -> dict[str, bool]:
    tests = [v for v in report.videos if v.kind == "test"]
    scenarios = [v for v in report.videos if v.kind == "scenario"]
    errors = [s["error"] for v in tests for s in v.steps.values()]
    known = [abs(e) for e in errors if e is not None]
    return {
        "c1": bool(tests) and all(all(s["status"] == "done" for s in v.steps.values()) and v.step_order == list(v.steps)
                                  for v in tests),
        "c2": bool(known) and len(known) == len(errors) and statistics.median(known) <= MEDIAN_ERROR_MAX
              and max(known) <= WORST_ERROR_MAX,
        "c3": bool(tests) and all(v.offsop and all(v.offsop) for v in tests),
        "c4": not any(a["kind"] in VIOLATIONS_FORBIDDEN_ON_TEST for v in tests for a in v.alerts),
        "c5": bool(scenarios) and all(v.expected_alerts and all(_has(v.alerts, e) for e in v.expected_alerts)
                                      for v in scenarios),
    }
