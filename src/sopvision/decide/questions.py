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
    state = {
        "sop": sop.description,
        "steps": [{"id": step_id, "status": status} for step_id, status in statuses.items()],
        "scene": scene.to_jev_dict(),
        "recent_events": [e.describe() for e in recent[-RECENT_EVENTS:]],
    }
    # Hands are a noisy cue on their own (an operator steadies the chassis with one hand), so a strictly ordered SOP
    # also names the step its progress says comes next; the model can still disagree when the scene contradicts it.
    if sop.order == "strict":
        state["next_expected_step"] = next((s.id for s in sop.steps if statuses.get(s.id) != "done"), None)
    return state
