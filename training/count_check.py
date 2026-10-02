"""Expected part counts per frame, derived from where the frame sits in the SOP (spec §6.1 step 3).

Auto-labels that disagree with these counts are queued first for human review.
"""

from dataclasses import dataclass, fields

FAN_ACTIONS = range(1, 7)
PSU_ACTIONS = (7, 8)
COVER_ACTION = 9
FAN_BAYS = 6
PSU_BAYS = 2
HIDDEN_BY_COVER = ("fan", "fan_bay_empty", "psu", "psu_bay_empty")

HIDDEN_BY_HANDS = HIDDEN_BY_COVER  # the same parts a hand can cover

Range = tuple[int, int] | None


@dataclass(frozen=True)
class ExpectedCounts:
    fan: Range
    fan_bay_empty: Range
    psu: Range
    psu_bay_empty: Range
    cover: Range
    chassis: Range


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reasons: list[str]


def _installed(actions_by_position: list[int], position: int, group) -> tuple[int, int]:
    done = sum(1 for a in actions_by_position[: position - 1] if a in group)
    in_progress = actions_by_position[position - 1] in group
    return done, done + (1 if in_progress else 0)


def expected_counts(actions_by_position: list[int], position: int) -> ExpectedCounts:
    fan_lo, fan_hi = _installed(actions_by_position, position, FAN_ACTIONS)
    psu_lo, psu_hi = _installed(actions_by_position, position, PSU_ACTIONS)
    cover_lo, cover_hi = _installed(actions_by_position, position, (COVER_ACTION,))
    if cover_hi > 0:  # cover going on or on: bays below may be hidden
        return ExpectedCounts(None, None, None, None, (cover_lo, cover_hi), (1, 1))
    return ExpectedCounts(
        fan=(fan_lo, fan_hi),
        fan_bay_empty=(FAN_BAYS - fan_hi, FAN_BAYS - fan_lo),
        psu=(psu_lo, psu_hi),
        psu_bay_empty=(PSU_BAYS - psu_hi, PSU_BAYS - psu_lo),
        cover=(0, 0),
        chassis=(1, 1),
    )


def check(counts: dict[str, int], expected: ExpectedCounts, hands: int) -> CheckResult:
    reasons = []
    for f in fields(expected):
        bounds = getattr(expected, f.name)
        if bounds is None:
            continue
        lo, hi = bounds
        n = counts.get(f.name, 0)
        # Each visible hand can hide about one part (never add one), so relax only the lower bound, one per hand.
        floor = lo - hands if f.name in HIDDEN_BY_HANDS else lo
        if n > hi or n < floor:
            reasons.append(f"{f.name}: {n} not in {lo}–{hi}")
    return CheckResult(ok=not reasons, reasons=reasons)
