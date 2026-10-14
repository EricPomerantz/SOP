"""Step and alert records produced by the SopTracker."""

from dataclasses import dataclass, field
from typing import Literal

StepStatus = Literal["pending", "active", "done"]
AlertLevel = Literal["info", "warning", "violation"]
AlertKind = Literal["order_violation", "part_removed", "off_sop", "anomaly", "sop_incomplete", "jev_offline",
                    "chassis_lost", "source_lost", "pipeline_error",
                    "torque", "torque_not_reached", "torque_unattributed", "torque_without_part", "torque_repeated"]


@dataclass(frozen=True)
class StepState:
    id: str
    status: StepStatus
    started_t: float | None
    done_t: float | None
    duration: float | None
    source: Literal["jev", "evidence", "torque"] | None


@dataclass(frozen=True)
class StepUpdate:
    t: float
    step_id: str
    status: StepStatus
    source: str | None


@dataclass(frozen=True)
class Alert:
    t: float
    level: AlertLevel
    kind: AlertKind
    message: str
    step_id: str | None = None
    data: dict = field(default_factory=dict)
