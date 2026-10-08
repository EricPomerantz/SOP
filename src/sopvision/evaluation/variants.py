"""Request variants for decision experiments on a frozen benchmark (scripts/bench_decisions.py --frozen --variant).

Each variant turns one frozen record into the (state, questions) both providers receive. The frozen set itself is
never changed, and every variant goes to Jev and Decider alike, so the comparison stays fair.
"""

from collections.abc import Callable
from dataclasses import replace

from sopvision.config import SopConfig
from sopvision.decide.questions import QuestionSet, eligible_steps, progress, with_progress


def statuses(rec: dict) -> dict[str, str]:
    return {s["id"]: s["status"] for s in rec["request"]["state"]["steps"]}


def expected_steps(rec: dict, sop: SopConfig) -> list[str]:
    """The step(s) the tracker expects next at request time: what an answer would be if it only echoed the tracker."""
    return eligible_steps(sop, statuses(rec))


def baseline(rec: dict, sop: SopConfig) -> tuple[dict, QuestionSet]:
    """The request exactly as it was recorded."""
    return rec["request"]["state"], QuestionSet(**rec["request"]["questions"])


def a1_progress(rec: dict, sop: SopConfig) -> tuple[dict, QuestionSet]:
    """A1: + the step the tracker expects next and the last step it confirmed (with seconds ago). Nothing else changes."""
    state, qs = baseline(rec, sop)
    prog = progress(sop, statuses(rec), rec["tracker"]["done_t"], state["scene"]["t"])
    return with_progress(state, prog), qs


def a6_steps_last(rec: dict, sop: SopConfig) -> tuple[dict, QuestionSet]:
    """A6: the step statuses move to the end of the state, right before the questions. Nothing else changes."""
    state, qs = baseline(rec, sop)
    return {k: v for k, v in state.items() if k != "steps"} | {"steps": state["steps"]}, qs


def a7_options_sorted(rec: dict, sop: SopConfig) -> tuple[dict, QuestionSet]:
    """A7: the current_step options in alphabetical order instead of SOP order. Nothing else changes."""
    state, qs = baseline(rec, sop)
    return state, replace(qs, current_step_options=dict(sorted(qs.current_step_options.items())))


def a8_steps_progress_last(rec: dict, sop: SopConfig) -> tuple[dict, QuestionSet]:
    """A6 + A1: step statuses at the end of the state, followed by the `progress` block."""
    state, qs = a6_steps_last(rec, sop)
    return state | {"progress": progress(sop, statuses(rec), rec["tracker"]["done_t"], state["scene"]["t"])}, qs


VARIANTS: dict[str, Callable[[dict, SopConfig], tuple[dict, QuestionSet]]] = {
    "baseline": baseline,
    "a1_progress": a1_progress,
    "a6_steps_last": a6_steps_last,
    "a7_options_sorted": a7_options_sorted,
    "a8_steps_progress_last": a8_steps_progress_last,
}
