"""SceneState: the JSON-able facts sent to Jev (spec §6.3), and the events that change them."""

from dataclasses import dataclass, field
from typing import Literal

SlotState = Literal["installed", "empty", "occluded", "unknown"]
Observed = Literal["installed", "empty"]
EventKind = Literal["slot_changed", "hand_entered", "hand_left", "held_changed", "chassis_lost", "chassis_found"]


@dataclass(frozen=True)
class SlotView:
    state: SlotState
    conf: float | None = None
    last: Observed | None = None

    def effective(self) -> Observed | None:
        """What the slot holds: the observed state, or the last observed one while hidden."""
        return self.state if self.state in ("installed", "empty") else self.last

    def to_jev_dict(self) -> dict:
        if self.state in ("installed", "empty"):
            return {"state": self.state, "conf": None if self.conf is None else round(self.conf, 2)}
        return {"state": self.state} | ({"last": self.last} if self.last else {})


@dataclass(frozen=True)
class HandView:
    near: str  # a slot name, "chassis" or "outside"


@dataclass(frozen=True)
class SceneState:
    version: int
    t: float
    chassis_visible: bool
    chassis_conf: float | None
    slots: dict[str, SlotView]
    hands: list[HandView]
    held: list[str]
    chassis_box: tuple | None = None

    def to_jev_dict(self) -> dict:
        return {
            "version": self.version,
            "t": round(self.t, 2),
            "chassis": {"visible": self.chassis_visible,
                        "conf": None if self.chassis_conf is None else round(self.chassis_conf, 2)},
            "slots": {name: view.to_jev_dict() for name, view in self.slots.items()},
            "hands": [{"near": h.near} for h in self.hands],
            "held": list(self.held),
        }


@dataclass(frozen=True)
class SceneEvent:
    t: float
    kind: EventKind
    slot: str | None = None
    from_state: str | None = None
    to_state: str | None = None
    zone: str | None = None
    items: list[str] | None = field(default=None)

    def describe(self) -> str:
        prefix = f"{self.t:.1f}s "
        if self.kind == "slot_changed":
            return prefix + f"{self.slot} {self.from_state}→{self.to_state}"
        if self.kind == "hand_entered":
            return prefix + f"hand entered {self.zone}"
        if self.kind == "hand_left":
            return prefix + f"hand left {self.zone}"
        if self.kind == "held_changed":
            return prefix + "held: " + (", ".join(self.items) if self.items else "nothing")
        return prefix + self.kind.replace("_", " ")
