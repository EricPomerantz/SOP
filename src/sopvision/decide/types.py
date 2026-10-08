"""Answers from a decision model (Jev or the local Decider) and the Decision record the tracker consumes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JevAnswers:
    """One provider's answers to the three SOP questions. Jev and Decider both return this shape."""

    current_step: str
    current_confidence: float
    current_probabilities: dict[str, float]
    active_step_done: float
    anomaly: float
    latency_ms: float  # the successful attempt only
    provider: str = "jev"


@dataclass(frozen=True)
class ProviderResult:
    """What one provider returned for one request: answers, or the error that replaced them."""

    provider: str
    answers: JevAnswers | None
    error: str | None
    request_ms: float  # wall time around ask(), retries and backoff included


@dataclass(frozen=True)
class Decision:
    version: int  # SceneState.version the question was asked about
    t_request: float
    active_step_id: str
    answers: JevAnswers
    request_ms: float | None = None
    ctx_ready: float | None = None  # perf_counter when the call first became due (live mode only)
