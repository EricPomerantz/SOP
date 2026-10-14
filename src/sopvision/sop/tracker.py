"""Applies Jev's judgments and the scene's evidence to the SOP (spec section 8).

Pure and deterministic: every input carries its own time, nothing reads a clock.
"""

from dataclasses import dataclass, replace

from sopvision.config import SopConfig
from sopvision.decide.types import Decision
from sopvision.scene.types import SceneState
from sopvision.sop.torque import AttributedTorque
from sopvision.sop.types import Alert, StepState, StepUpdate

REMOVAL_HOLD = 1.0  # seconds a done step's part must be observed missing


@dataclass
class _Streak:
    key: str | None = None
    start: float = 0.0
    fired: bool = False

    def update(self, key: str | None, t: float) -> float | None:
        """Track a run of identical keys; return how long it has held (None when broken)."""
        if key is None:
            self.key, self.fired = None, False
            return None
        if key != self.key:
            self.key, self.start, self.fired = key, t, False
        return t - self.start


class SopTracker:
    def __init__(self, sop: SopConfig):
        self.sop = sop
        self.order = [s.id for s in sop.steps]
        self._steps = {sid: StepState(sid, "pending", None, None, None, None) for sid in self.order}
        self._active: str | None = None
        self._activate = _Streak()
        self._done = _Streak()
        self._off_sop = _Streak()
        self._anomaly = _Streak()
        self._evidence_since: dict[str, float] = {}
        self._removed_since: dict[str, float] = {}

    # -- queries -------------------------------------------------------------------------------------------
    def steps(self) -> list[StepState]:
        return [self._steps[s] for s in self.order]

    def statuses(self) -> dict[str, str]:
        return {s: self._steps[s].status for s in self.order}

    def active_step_id(self) -> str:
        if self._active and self._steps[self._active].status != "done":
            return self._active
        return next((s for s in self.order if self._steps[s].status != "done"), self.order[-1])

    # -- inputs --------------------------------------------------------------------------------------------
    def on_decision(self, d: Decision, scene: SceneState) -> list[StepUpdate | Alert]:
        out: list[StepUpdate | Alert] = []
        acc, hold, a, t = self.sop.decisions.accept_confidence, self.sop.decisions.hold_seconds, d.answers, d.t_request
        confident = a.current_step if a.current_confidence >= acc else None

        held = self._activate.update(confident if confident in self._steps else None, t)
        if held is not None and held >= hold and confident != self._active and self._steps[confident].status != "done":
            out += self._set_active(confident, t)

        held = self._off_sop.update("off_sop" if confident == "off_sop" else None, t)
        if held is not None and held >= hold and not self._off_sop.fired:
            self._off_sop.fired = True
            out.append(Alert(t, "warning", "off_sop", "Off-SOP activity"))

        held = self._anomaly.update("anomaly" if a.anomaly >= acc else None, t)
        if held is not None and held >= hold and not self._anomaly.fired:
            self._anomaly.fired = True
            out.append(Alert(t, "warning", "anomaly", "Something looks inconsistent with the SOP",
                             data={"scene": scene.to_jev_dict()}))

        held = self._done.update(d.active_step_id if a.active_step_done >= acc else None, t)
        step = self._steps.get(d.active_step_id)
        if (held is not None and held >= hold and step is not None and step.status != "done"
                and self._evidence_holds(d.active_step_id, scene)):
            out += self._complete(d.active_step_id, t, "jev")
        return out

    def on_scene(self, scene: SceneState) -> list[StepUpdate | Alert]:
        out: list[StepUpdate | Alert] = []
        t = scene.t
        for sid in self.order:
            step = self._steps[sid]
            if not self.sop.step(sid).evidence:
                continue
            if step.status != "done":
                if self._evidence_holds(sid, scene):
                    since = self._evidence_since.setdefault(sid, t)
                    wait = (0.0 if self._torque_step(sid)  # a torque result is a discrete, confirmed event
                            else self.sop.decisions.hands_off_seconds if self.sop.step(sid).hands_off
                            else self.sop.decisions.evidence_only_after)
                    if t - since >= wait:
                        out += self._complete(sid, t, "torque" if self._torque_step(sid) else "evidence")
                else:
                    self._evidence_since.pop(sid, None)
            elif self._contradicted(sid, scene):
                since = self._removed_since.setdefault(sid, t)
                if t - since >= REMOVAL_HOLD:
                    out.append(Alert(t, "violation", "part_removed", f"{sid}: part removed after the step was completed",
                                     step_id=sid))
                    self._steps[sid] = StepState(sid, "pending", None, None, None, None)
                    self._removed_since.pop(sid, None)
                    out.append(StepUpdate(t, sid, "pending", None))
            else:
                self._removed_since.pop(sid, None)
        return out

    def on_torque(self, a: AttributedTorque, scene: SceneState) -> tuple[list[Alert], str | None]:
        """Judge an attributed torque result: (alerts, slot to mark torqued or None).

        The step itself completes on the next scene, when the slot shows as torqued (on_scene), so the order check
        is the same one every other step goes through.
        """
        r, slot, t = a.result, a.slot, a.result.t
        where = slot or "an unknown slot"
        value = f" ({r.torque_nm:g} Nm)" if r.torque_nm is not None else ""
        if not r.ok:
            return [Alert(t, "warning", "torque_not_reached", f"Torque not reached on {where}{value}", data={"slot": slot})], None
        if slot is None:
            return [Alert(t, "warning", "torque_unattributed", f"Torque result{value} couldn't be placed on a slot: {a.reason}")], None
        view = scene.slots.get(slot)
        if view is None or view.effective() != "installed":
            return [Alert(t, "warning", "torque_without_part", f"Torque on {slot}{value}, but no part is seen there",
                          data={"slot": slot})], None
        if view.torqued:
            return [Alert(t, "warning", "torque_repeated", f"{slot} torqued again{value}", data={"slot": slot})], None
        return [Alert(t, "info", "torque", f"Torque OK on {slot}{value}", data={"slot": slot})], slot

    def set_jev_online(self, online: bool, t: float, model: str = "Jev") -> list[Alert]:
        """The deciding model (Jev, or a local one) went offline. The kind stays `jev_offline` for every model."""
        return [] if online else [Alert(t, "warning", "jev_offline", f"{model} offline — using evidence only")]

    def end_run(self, t: float) -> list[Alert]:
        missing = [s for s in self.order if self._steps[s].status != "done"]
        if not missing:
            return []
        return [Alert(t, "violation", "sop_incomplete", f"SOP incomplete: missing {', '.join(missing)}",
                      data={"missing": missing})]

    # -- rules ---------------------------------------------------------------------------------------------
    def _evidence_holds(self, sid: str, scene: SceneState) -> bool:
        """Every evidence slot holds the required part — and, for hands_off steps, nobody is touching it."""
        step = self.sop.step(sid)
        return all(slot in scene.slots and (scene.slots[slot].torqued if want == "torqued" else
                                            scene.slots[slot].effective() == want
                                            and not (step.hands_off and scene.slots[slot].touched))
                   for slot, want in step.evidence.items())

    def _torque_step(self, sid: str) -> bool:
        evidence = self.sop.step(sid).evidence
        return bool(evidence) and all(want == "torqued" for want in evidence.values())

    def _contradicted(self, sid: str, scene: SceneState) -> bool:
        """An evidence slot is *observed* (not occluded/unknown) in the opposite state."""
        for slot, want in self.sop.step(sid).evidence.items():
            view = scene.slots.get(slot)
            if view is None or view.state not in ("installed", "empty"):
                continue
            if want == "torqued" and view.state == "empty":  # the torqued part is gone
                return True
            if want != "torqued" and view.state != want:
                return True
        return False

    def _set_active(self, sid: str, t: float) -> list[StepUpdate]:
        out = []
        if self._active and self._active != sid and self._steps[self._active].status == "active":
            self._steps[self._active] = replace(self._steps[self._active], status="pending")
            out.append(StepUpdate(t, self._active, "pending", None))
        step = self._steps[sid]
        self._steps[sid] = replace(step, status="active", started_t=step.started_t if step.started_t is not None else t)
        self._active = sid
        out.append(StepUpdate(t, sid, "active", None))
        return out

    def _complete(self, sid: str, t: float, source: str) -> list[StepUpdate | Alert]:
        step = self._steps[sid]
        duration = t - step.started_t if step.started_t is not None else None
        self._steps[sid] = StepState(sid, "done", step.started_t, t, duration, source)
        self._evidence_since.pop(sid, None)
        if self._active == sid:
            self._active = None
        out: list[StepUpdate | Alert] = [StepUpdate(t, sid, "done", source)]
        must = self.sop.must_precede(sid)
        missing = [s for s in self.order if s in must and self._steps[s].status != "done"]
        if missing:
            out.append(Alert(t, "violation", "order_violation", f"{sid} completed before {', '.join(missing)}",
                             step_id=sid, data={"missing": missing}))
        return out
