"""Docker parity: a container eval must match the native eval (spec §2 criterion 7)."""

from sopvision.evaluation.metrics import EvalReport

DONE_TIME_TOLERANCE = 0.5
SLOT_AGREEMENT_MIN = 0.99


def compare_reports(a: EvalReport, b: EvalReport) -> list[str]:
    """Mismatch messages between two eval reports; an empty list means parity."""
    problems = []
    for name, r in (("first", a), ("second", b)):
        if not r.passed:
            problems.append(f"{name} report did not pass acceptance")
    videos_b = {v.id: v for v in b.videos}
    for va in a.videos:
        vb = videos_b.get(va.id)
        if vb is None:
            problems.append(f"{va.id}: missing from second report")
            continue
        if va.step_order != vb.step_order:
            problems.append(f"{va.id}: step order differs: {va.step_order} vs {vb.step_order}")
        kinds_a, kinds_b = {x["kind"] for x in va.alerts}, {x["kind"] for x in vb.alerts}
        if kinds_a != kinds_b:
            problems.append(f"{va.id}: alert kinds differ: {sorted(kinds_a)} vs {sorted(kinds_b)}")
        for step, sa in va.steps.items():
            ta, tb = sa.get("done_t"), vb.steps.get(step, {}).get("done_t")
            if (ta is None) != (tb is None):
                problems.append(f"{va.id}: {step} done in one run only")
            elif ta is not None and abs(ta - tb) > DONE_TIME_TOLERANCE:
                problems.append(f"{va.id}: {step} done at {ta:.2f}s vs {tb:.2f}s (more than {DONE_TIME_TOLERANCE}s apart)")
        pairs = agree = 0
        for fa, fb in zip(va.frames, vb.frames):
            for slot, state in fa["slots"].items():
                pairs += 1
                agree += fb["slots"].get(slot) == state
        if len(va.frames) != len(vb.frames):
            problems.append(f"{va.id}: frame counts differ ({len(va.frames)} vs {len(vb.frames)})")
        if pairs and agree / pairs < SLOT_AGREEMENT_MIN:
            problems.append(f"{va.id}: slot agreement {100 * agree / pairs:.1f}% (needs {100 * SLOT_AGREEMENT_MIN:.0f}%)")
    return problems
