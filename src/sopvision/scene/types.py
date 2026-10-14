"""SceneState: the JSON-able facts sent to Jev (spec §6.3), and the events that change them."""

from dataclasses import dataclass, field
from typing import Literal

SlotState = Literal["installed", "empty", "occluded", "unknown"]
Observed = Literal["installed", "empty"]
EventKind = Literal["slot_changed", "hand_entered", "hand_left", "held_changed", "chassis_lost", "chassis_found",
                    "tool_entered", "tool_left"]


@dataclass(frozen=True)
class SlotView:
    state: SlotState
    conf: float | None = None
    last: Observed | None = None
    touched: bool = False  # a hand overlaps the part (or hides the slot)
    torqued: bool = False  # a torque-OK result was attributed to this slot (and the part hasn't left since)
    # For the video overlay only (never sent to a model): the part class seen in the slot since it was installed
    # (majority over its frames, e.g. lug_bolt_gold), and this frame's matched detection box in pixels.
    part: str | None = None
    box: tuple[float, float, float, float] | None = None

    def effective(self) -> Observed | None:
        """What the slot holds: the observed state, or the last observed one while hidden."""
        return self.state if self.state in ("installed", "empty") else self.last

    def to_jev_dict(self) -> dict:
        # `torqued` only appears when true, so SOPs without torque evidence keep exactly the state they had.
        extra = {"torqued": True} if self.torqued else {}
        if self.state in ("installed", "empty"):
            return {"state": self.state, "conf": None if self.conf is None else round(self.conf, 2)} | extra
        return {"state": self.state} | ({"last": self.last} if self.last else {}) | extra


@dataclass(frozen=True)
class HandView:
    near: str  # a slot name, "chassis" or "outside"


@dataclass(frozen=True)
class ToolView:
    tool: str  # detector class, e.g. torque_wrench
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
    tools: list[ToolView] | None = None  # None for SOPs without tools, so their state is unchanged

    def to_jev_dict(self) -> dict:
        tools = {} if self.tools is None else {"tools": [{"tool": v.tool, "near": v.near} for v in self.tools]}
        return {
            "version": self.version,
            "t": round(self.t, 2),
            "chassis": {"visible": self.chassis_visible,
                        "conf": None if self.chassis_conf is None else round(self.chassis_conf, 2)},
            "slots": {name: view.to_jev_dict() for name, view in self.slots.items()},
            "hands": [{"near": h.near} for h in self.hands],
            "held": list(self.held),
        } | tools


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
        if self.kind in ("tool_entered", "tool_left"):
            return prefix + f"{self.items[0] if self.items else 'tool'} {'at' if self.kind == 'tool_entered' else 'left'} {self.zone}"
        if self.kind == "held_changed":
            return prefix + "held: " + (", ".join(self.items) if self.items else "nothing")
        return prefix + self.kind.replace("_", " ")
