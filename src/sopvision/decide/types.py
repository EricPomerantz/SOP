"""Jev's answers and the Decision record the tracker consumes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JevAnswers:
    current_step: str
    current_confidence: float
    current_probabilities: dict[str, float]
    active_step_done: float
    anomaly: float
    latency_ms: float


@dataclass(frozen=True)
class Decision:
    version: int  # SceneState.version the question was asked about
    t_request: float
    active_step_id: str
    answers: JevAnswers
