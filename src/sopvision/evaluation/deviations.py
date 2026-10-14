"""Deviations edited into recorded decision states, driven by the SOP file (slots, evidence, order rules).

Each generator takes a frozen-format record (`request.state`, `request.questions`) and an SOP, edits the scene in place
so that a known deviation is happening, and returns the truth:

    {"current_step": [accepted answers] | None, "anomaly": True | False | None, "active_step_done": bool | None}

None means unknown / unchanged. Used for evaluation probes (scripts/make_probes.py) and training rows
(scripts/build_training_set.py); never feed the same generator's output to both a model's training and its test.

    out_of_order_work     the operator works on a later step that isn't allowed yet: hands holding its part at its
                          slot, or the tool (e.g. the torque wrench) at its slot          -> current_step = that step
    out_of_order_install  a later step's slot has just become installed / torqued          -> anomaly
    idle_bench            nobody at the bench; the last activity 20-30 s ago               -> idle / off_sop
    done_part_removed     a completed step's part has just left its slot, a hand on it     -> anomaly
"""

import re
from collections.abc import Callable

from sopvision.config import SopConfig
from sopvision.decide.questions import eligible_steps

_EVENT_T = re.compile(r"^\d+(?:\.\d+)?s ")


def statuses(rec: dict) -> dict[str, str]:
    return {s["id"]: s["status"] for s in rec["request"]["state"]["steps"]}


def expected(rec: dict, sop: SopConfig) -> str | None:
    eligible = eligible_steps(sop, statuses(rec))
    return eligible[0] if eligible else None


def _evidence(sop: SopConfig, step_id: str) -> tuple[str, str] | None:
    """The step's (slot, wanted state), when it has slot evidence."""
    ev = sop.step(step_id).evidence
    return next(iter(ev.items())) if ev else None


def _ahead(rec: dict, sop: SopConfig) -> str | None:
    """A step that isn't allowed yet: the second one after the expected step in SOP order (else the first)."""
    exp = expected(rec, sop)
    if exp is None:
        return None
    st, allowed = statuses(rec), set(eligible_steps(sop, statuses(rec)))
    order = [s.id for s in sop.steps]
    later = [s for s in order[order.index(exp) + 1:] if st.get(s) != "done" and s not in allowed and _evidence(sop, s)]
    return later[1] if len(later) > 1 else (later[0] if later else None)


def _events(t: float, *lines: tuple[float, str]) -> list[str]:
    return [f"{t - dt:.1f}s {text}" for dt, text in lines]


def _empty_expected_slot(rec: dict, sop: SopConfig) -> bool:
    """Show the expected step's slot as plainly empty (installed-type steps only). Returns whether it did."""
    ev = _evidence(sop, expected(rec, sop))
    if ev is None or ev[1] != "installed":
        return False
    rec["request"]["state"]["scene"]["slots"][ev[0]] = {"state": "empty", "conf": 0.93}
    return True


def out_of_order_work(rec: dict, sop: SopConfig) -> dict:
    target = _ahead(rec, sop)
    slot, want = _evidence(sop, target)
    st = rec["request"]["state"]
    sc, t = st["scene"], st["scene"]["t"]
    emptied = _empty_expected_slot(rec, sop)
    if want == "torqued":
        tool = sop.layout.tools[0]
        sc["slots"][slot] = {"state": "installed", "conf": 0.92}
        sc["tools"], sc["hands"] = [{"tool": tool, "near": slot}], [{"near": slot}]
        st["recent_events"] = st["recent_events"][:7] + _events(t, (1.8, f"hand entered {slot}"), (1.2, f"{tool} at {slot}"))
    else:
        part = sop.layout.slots[slot].fills[0]
        sc["slots"][slot] = {"state": "occluded", "last": "empty"}
        sc["hands"], sc["held"] = [{"near": slot}, {"near": slot}], [part]
        st["recent_events"] = st["recent_events"][:6] + _events(t, (2.4, f"held: {part}"), (1.6, f"hand entered {slot}"),
                                                                (1.2, f"{slot} empty→occluded"), (0.8, f"hand entered {slot}"))
    return {"current_step": [target], "anomaly": None, "active_step_done": False if emptied else None}


def out_of_order_install(rec: dict, sop: SopConfig) -> dict:
    target = _ahead(rec, sop)
    slot, want = _evidence(sop, target)
    st = rec["request"]["state"]
    sc, t = st["scene"], st["scene"]["t"]
    emptied = _empty_expected_slot(rec, sop)
    if want == "torqued":
        tool = sop.layout.tools[0]
        sc["slots"][slot] = {"state": "installed", "conf": 0.92, "torqued": True}
        sc["tools"], sc["hands"] = [], [{"near": "outside"}]
        st["recent_events"] = st["recent_events"][:7] + _events(t, (2.0, f"{tool} at {slot}"), (0.8, f"{tool} left {slot}"),
                                                                (0.5, f"hand left {slot}"))
    else:
        sc["slots"][slot] = {"state": "installed", "conf": 0.95}
        sc["hands"], sc["held"] = [{"near": "chassis"}], []
        st["recent_events"] = st["recent_events"][:6] + _events(t, (3.0, f"hand entered {slot}"), (1.5, f"{slot} occluded→installed"),
                                                                (1.0, "held: nothing"), (0.6, f"hand left {slot}"))
    return {"current_step": None, "anomaly": True, "active_step_done": False if emptied else None}


def idle_bench(rec: dict, sop: SopConfig) -> dict:
    st = rec["request"]["state"]
    sc, t = st["scene"], st["scene"]["t"]
    sc["hands"], sc["held"] = [], []
    if "tools" in sc:
        sc["tools"] = []
    old = [_EVENT_T.sub("", e) for e in st["recent_events"]]  # the same events, long ago
    st["recent_events"] = ([f"{t - 30 + i:.1f}s {e}" for i, e in enumerate(old)] + _events(t, (20.0, "hand left chassis")))[-10:]
    return {"current_step": ["idle", "off_sop"], "anomaly": False, "active_step_done": None}


def _done_installed(rec: dict, sop: SopConfig) -> list[str]:
    st = statuses(rec)
    return [s.id for s in sop.steps if st.get(s.id) == "done" and (_evidence(sop, s.id) or ("", ""))[1] == "installed"]


def done_part_removed(rec: dict, sop: SopConfig) -> dict:
    slot, _ = _evidence(sop, _done_installed(rec, sop)[-1])
    part = sop.layout.slots[slot].fills[0]
    st = rec["request"]["state"]
    sc, t = st["scene"], st["scene"]["t"]
    sc["slots"][slot] = {"state": "empty", "conf": 0.92}
    sc["hands"], sc["held"] = [{"near": slot}], [part]
    st["recent_events"] = st["recent_events"][:6] + _events(t, (2.0, f"hand entered {slot}"), (1.2, f"{slot} occluded→empty"),
                                                            (1.0, f"held: {part}"), (0.5, f"hand left {slot}"))
    return {"current_step": None, "anomaly": True, "active_step_done": None}


GENERATORS: dict[str, Callable[[dict, SopConfig], dict]] = {
    "out_of_order_work": out_of_order_work,
    "out_of_order_install": out_of_order_install,
    "idle_bench": idle_bench,
    "done_part_removed": done_part_removed,
}

ELIGIBLE: dict[str, Callable[[dict, SopConfig], bool]] = {
    "out_of_order_work": lambda r, sop: _ahead(r, sop) is not None,
    "out_of_order_install": lambda r, sop: _ahead(r, sop) is not None,
    "idle_bench": lambda r, sop: True,
    "done_part_removed": lambda r, sop: bool(_done_installed(r, sop)),
}
