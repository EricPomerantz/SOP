"""Jev vs Decider on one request: the side-by-side record written to the run log as `decision_comparison`.

Agreement is not accuracy: the record keeps both answers and a `ground_truth` slot that the benchmark
(scripts/bench_decisions.py) fills from the dataset's step segments when they are known.
"""

from dataclasses import asdict

from sopvision.decide.questions import QuestionSet
from sopvision.decide.types import JevAnswers, ProviderResult


def provider_entry(r: ProviderResult) -> dict:
    return {"provider": r.provider, "answers": None if r.answers is None else asdict(r.answers),
            "error": r.error, "request_ms": round(r.request_ms, 2)}


def compare_answers(a: JevAnswers, b: JevAnswers, accept: float) -> dict:
    """How far apart two providers' answers are, raw and at the tracker's acting threshold (`accept`)."""
    def acted_step(x: JevAnswers) -> str | None:
        return x.current_step if x.current_confidence >= accept else None

    options = set(a.current_probabilities) | set(b.current_probabilities)
    tv = 0.5 * sum(abs(a.current_probabilities.get(o, 0.0) - b.current_probabilities.get(o, 0.0)) for o in options)
    return {
        "same_current_step": a.current_step == b.current_step,
        "current_confidence_diff": round(b.current_confidence - a.current_confidence, 4),
        "current_probabilities_tv": round(tv, 4),  # total variation distance, 0 = identical, 1 = disjoint
        "active_step_done_diff": round(b.active_step_done - a.active_step_done, 4),
        "anomaly_diff": round(b.anomaly - a.anomaly, 4),
        "off_sop_diff": round(b.current_probabilities.get("off_sop", 0.0) - a.current_probabilities.get("off_sop", 0.0), 4),
        # Would the tracker act the same way? (it needs >= accept; see sop/tracker.py)
        "same_actionable": {
            "current_step": acted_step(a) == acted_step(b),
            "active_step_done": (a.active_step_done >= accept) == (b.active_step_done >= accept),
            "anomaly": (a.anomaly >= accept) == (b.anomaly >= accept),
        },
    }


def comparison_record(request_id: str, t_request: float, version: int, state: dict, qs: QuestionSet,
                      primary: ProviderResult, shadow: ProviderResult, accept: float) -> dict:
    both = primary.answers is not None and shadow.answers is not None
    return {
        "request_id": request_id,
        "t_request": t_request,
        "version": version,
        "active_step_id": qs.active_step_id,
        "request": {"state": state, "questions": asdict(qs)},  # everything needed to replay the request
        "authoritative": primary.provider,  # whose answer drove the tracker
        primary.provider: provider_entry(primary),
        shadow.provider: provider_entry(shadow),
        # differences are shadow minus authoritative (Decider minus Jev in compare mode)
        "comparison": compare_answers(primary.answers, shadow.answers, accept) if both else None,
        "ground_truth": None,
    }
