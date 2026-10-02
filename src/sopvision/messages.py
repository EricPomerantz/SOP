"""WebSocket messages from a run to the dashboard (spec §9.1). Mirrored in web/src/types.ts."""

from typing import Literal

from pydantic import BaseModel


class SceneMsg(BaseModel):
    type: Literal["scene"] = "scene"
    t: float
    scene: dict


class DecisionMsg(BaseModel):
    type: Literal["decision"] = "decision"
    t: float
    current_step: str
    confidence: float
    active_step_id: str
    active_step_done: float
    anomaly: float
    latency_ms: float


class StepUpdateMsg(BaseModel):
    type: Literal["step_update"] = "step_update"
    t: float
    step_id: str
    status: str
    source: str | None
    duration: float | None


class AlertMsg(BaseModel):
    type: Literal["alert"] = "alert"
    t: float
    level: str
    kind: str
    message: str
    step_id: str | None


class RunSummaryMsg(BaseModel):
    type: Literal["run_summary"] = "run_summary"
    t: float
    report: dict


Message = SceneMsg | DecisionMsg | StepUpdateMsg | AlertMsg | RunSummaryMsg


def to_ws(msg: Message) -> dict:
    return msg.model_dump()
