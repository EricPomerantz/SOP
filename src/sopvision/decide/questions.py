"""Questions and state for Jev, generated from the SOP config (spec §7.2–7.3) — no SOP-specific code."""

from dataclasses import dataclass

from sopvision.config import SopConfig
from sopvision.scene.types import SceneEvent, SceneState

CURRENT_STEP_QUESTION = "Which step is the operator performing right now?"
ANOMALY_QUESTION = "Is anything in the scene inconsistent with the SOP so far?"
RECENT_EVENTS = 10


@dataclass(frozen=True)
class QuestionSet:
    current_step_options: dict[str, str]
    active_step_id: str
    active_step_question: str


def build_questions(sop: SopConfig, active_step_id: str) -> QuestionSet:
    options = {s.id: s.text for s in sop.steps} | {"idle": sop.idle, "off_sop": sop.off_sop}
    text = sop.step(active_step_id).text
    return QuestionSet(options, active_step_id, "Has this step been completed: " + text.rstrip(".") + "?")


def build_state(sop: SopConfig, statuses: dict[str, str], scene: SceneState, recent: list[SceneEvent]) -> dict:
    return {
        "sop": sop.description,
        "steps": [{"id": step_id, "status": status} for step_id, status in statuses.items()],
        "scene": scene.to_jev_dict(),
        "recent_events": [e.describe() for e in recent[-RECENT_EVENTS:]],
    }


def eligible_steps(sop: SopConfig, statuses: dict[str, str]) -> list[str]:
    """Steps not done yet whose prerequisites (per the SOP's order rules) are all done: what may happen next."""
    done = {s for s, status in statuses.items() if status == "done"}
    return [s.id for s in sop.steps if s.id not in done and sop.must_precede(s.id) <= done]


def progress(sop: SopConfig, statuses: dict[str, str], done_t: dict[str, float], t: float) -> dict:
    """What the tracker knows about where the run is (experiment A1): the step(s) it expects next and the
    last step it confirmed, with how long ago."""
    eligible = eligible_steps(sop, statuses)
    last = max(((done_t[s], s) for s, status in statuses.items() if status == "done" and s in done_t), default=None)
    out: dict = {"expected_step": eligible[0]} if len(eligible) == 1 else {"expected_steps": eligible}
    out["last_completed"] = None if last is None else {"step": last[1], "seconds_ago": round(t - last[0], 1)}
    return out


def with_progress(state: dict, prog: dict) -> dict:
    """The state with `progress` placed right after the step list."""
    out = {}
    for key, value in state.items():
        out[key] = value
        if key == "steps":
            out["progress"] = prog
    return out


def wire_questions(qs: QuestionSet) -> dict:
    """The questions exactly as JevClient sends them on the wire (training data must match serving)."""
    from typesafe_sdk import Choice, Noul

    return {"current_step": Choice(instructions=CURRENT_STEP_QUESTION, criteria=qs.current_step_options).model_dump(),
            "active_step_done": Noul(instructions=qs.active_step_question).model_dump(),
            "anomaly": Noul(instructions=ANOMALY_QUESTION).model_dump()}
