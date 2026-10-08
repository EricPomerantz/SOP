"""Jev vs Decider: ground-truth labels for logged decision requests and the comparison metrics.

Agreement and accuracy are reported separately: two providers agreeing proves neither right.

Ground truth comes from the recordings' step segments (sopvision.dataset), and only where it is clear:
  current_step  the step whose segment holds the request time; off-SOP segments accept "idle" or "off_sop"
                (the recordings' off-SOP stretches are an idle bench). Requests in the first BOUNDARY_GRACE s
                of a segment are left unlabelled: they may still be about the step that just ended.
  anomaly       False on the clean test recordings; unknown elsewhere.
  active_step_done  not labelled yet.
A `ground_truth` of None means unknown, never "wrong".
"""

import math
import re

from sopvision.dataset import Segment
from sopvision.evaluation.metrics import BOUNDARY_GRACE

OFF_SOP_ANSWERS = ("idle", "off_sop")
_RUN = re.compile(r"Install_(\d+)")


def run_number(source_id: str) -> int | None:
    """The recording a run replayed (`test:Install_12`, a file source on Install_12.MP4, …), if any."""
    m = _RUN.search(source_id)
    return int(m.group(1)) if m else None


def expected_steps(segments: list[Segment], t: float, grace: float = BOUNDARY_GRACE) -> list[str] | None:
    for seg in segments:
        if seg.start <= t < seg.end:
            if t - seg.start < grace:
                return None
            return list(OFF_SOP_ANSWERS) if seg.step_id == "off_sop" else [seg.step_id]
    return None


def label(record: dict, segments: list[Segment] | None, clean_run: bool, source: str) -> dict:
    """The record with `ground_truth` filled in from what is known about the recording."""
    current = expected_steps(segments, record["t_request"]) if segments else None
    anomaly = False if clean_run else None
    if current is None and anomaly is None:
        return record
    return record | {"ground_truth": {"current_step": current, "anomaly": anomaly, "source": source}}


def percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    s = sorted(values)

    def pick(q: float) -> float:  # nearest rank
        return round(s[min(len(s), max(1, math.ceil(q * len(s)))) - 1], 2)

    return {"n": len(s), "p50": pick(0.5), "p95": pick(0.95), "max": round(s[-1], 2), "mean": round(sum(s) / len(s), 2)}


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def provider_metrics(records: list[dict], provider: str, accept: float) -> dict:
    entries = [(r, r.get(provider)) for r in records if r.get(provider) is not None]
    answered = [(r, e["answers"]) for r, e in entries if e["answers"] is not None]
    labelled = [(r, a) for r, a in answered if (r.get("ground_truth") or {}).get("current_step")]
    correct = [a["current_step"] in r["ground_truth"]["current_step"] for r, a in labelled]
    wrong_conf = [a["current_confidence"] for (r, a), ok in zip(labelled, correct) if not ok]
    is_off = [set(r["ground_truth"]["current_step"]) == set(OFF_SOP_ANSWERS) for r, _ in labelled]
    said_off = [a["current_step"] in OFF_SOP_ANSWERS for _, a in labelled]
    acting = [ok for (r, a), ok in zip(labelled, correct) if a["current_confidence"] >= accept]
    tracked = [(r, a) for r, a in answered if r.get("expected_steps") is not None]
    clean = [a for r, a in answered if (r.get("ground_truth") or {}).get("anomaly") is False]
    anomalous = [a for r, a in answered if (r.get("ground_truth") or {}).get("anomaly") is True]
    groups = sorted({r["probe"] for r, _ in labelled if r.get("probe")})
    wait = {id(r): r.get("wait_ms") for r, _ in entries}
    e2e = [wait[id(r)] + e["request_ms"] for r, e in entries if e["answers"] is not None and wait[id(r)] is not None]
    return {
        "requests": len(entries),
        "answered": len(answered),
        "errors": sum(e["error"] is not None for _, e in entries),
        "current_step": {
            "labelled": len(labelled),
            "accuracy": _rate(sum(correct), len(labelled)),
            "mean_confidence": _mean([a["current_confidence"] for _, a in answered]),
            "mean_confidence_when_wrong": _mean(wrong_conf),
            "acted_on": _rate(sum(a["current_confidence"] >= accept for _, a in answered), len(answered)),
            # On the labelled requests: how often the tracker would act on the answer, and how often it is right then.
            "acts": _rate(len(acting), len(labelled)),
            "accuracy_when_acting": _rate(sum(acting), len(acting)),
            # How often the answer is just the step the tracker already expects next (high = may be echoing the tracker).
            "echo_rate": _rate(sum(a["current_step"] in r["expected_steps"] for r, a in tracked), len(tracked)),
        },
        "off_sop": {  # positive = idle/off_sop
            "segments_labelled": sum(is_off),
            "false_positive_rate": _rate(sum(s and not o for s, o in zip(said_off, is_off)), sum(not o for o in is_off)),
            "false_negative_rate": _rate(sum(o and not s for s, o in zip(said_off, is_off)), sum(is_off)),
        },
        "anomaly": {  # labelled only where the run is known to be clean
            "labelled_clean": len(clean),
            "false_positive_rate": _rate(sum(a["anomaly"] >= accept for a in clean), len(clean)),
            "mean_probability_when_clean": _mean([a["anomaly"] for a in clean]),
            "labelled_anomalous": len(anomalous),  # deviation probes
            "detection_rate": _rate(sum(a["anomaly"] >= accept for a in anomalous), len(anomalous)),
            "mean_probability_when_anomalous": _mean([a["anomaly"] for a in anomalous]),
        },
        # Deviation probes: step accuracy per probe kind.
        "by_probe": {g: _rate(sum(ok for (r, _), ok in zip(labelled, correct) if r.get("probe") == g),
                              sum(r.get("probe") == g for r, _ in labelled)) for g in groups},
        "request_ms": percentiles([e["request_ms"] for _, e in entries if e["answers"] is not None]),
        "e2e_ms": percentiles(e2e),  # context ready -> answer back (live runs only)
    }


def summarize(records: list[dict], accept: float) -> dict:
    """Per-provider accuracy and latency plus their agreement, over decision_comparison records."""
    if not records:
        return {"requests": 0}
    providers = sorted({k for r in records for k in r if isinstance(r.get(k), dict) and "answers" in r[k]})
    pairs = [r["comparison"] for r in records if r.get("comparison")]
    actionable = {k: _rate(sum(p["same_actionable"][k] for p in pairs), len(pairs))
                  for k in ("current_step", "active_step_done", "anomaly")}
    return {
        "requests": len(records),
        "authoritative": records[0].get("authoritative"),
        "providers": {p: provider_metrics(records, p, accept) for p in providers},
        "agreement": {
            "both_answered": len(pairs),
            "same_current_step": _rate(sum(p["same_current_step"] for p in pairs), len(pairs)),
            "same_actionable": actionable,
            "mean_abs_confidence_diff": _mean([abs(p["current_confidence_diff"]) for p in pairs]),
            "mean_probabilities_tv": _mean([p["current_probabilities_tv"] for p in pairs]),
        },
    }


def summary_text(summary: dict) -> list[str]:
    if not summary.get("requests"):
        return ["no decision comparisons"]
    ag = summary["agreement"]
    lines = [f"requests {summary['requests']}, both answered {ag['both_answered']}, "
             f"same current_step {ag['same_current_step']}, same action {ag['same_actionable']}"]
    for name, m in summary["providers"].items():
        cs, lat = m["current_step"], m["request_ms"] or {}
        lines.append(f"{name}: answered {m['answered']}/{m['requests']} (errors {m['errors']}), "
                     f"step accuracy {cs['accuracy']} on {cs['labelled']} labelled, acts {cs['acts']} "
                     f"(correct when acting {cs['accuracy_when_acting']}), echo {cs['echo_rate']}, conf {cs['mean_confidence']} "
                     f"(when wrong {cs['mean_confidence_when_wrong']}), off-SOP FP/FN {m['off_sop']['false_positive_rate']}/"
                     f"{m['off_sop']['false_negative_rate']}, anomaly FP {m['anomaly']['false_positive_rate']}, "
                     f"request p50/p95/max {lat.get('p50')}/{lat.get('p95')}/{lat.get('max')} ms")
        if m["anomaly"]["labelled_anomalous"]:
            lines.append(f"  {name} anomaly detection {m['anomaly']['detection_rate']} on {m['anomaly']['labelled_anomalous']} "
                         f"(mean p {m['anomaly']['mean_probability_when_anomalous']}); step accuracy by probe {m['by_probe']}")
    return lines
