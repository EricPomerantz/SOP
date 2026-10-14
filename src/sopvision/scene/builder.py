"""Detections → a stable SceneState plus change events (spec §6.2)."""

import math
from collections import Counter
from dataclasses import dataclass, field

from sopvision.config import SopConfig
from sopvision.perception.detector import Detection
from sopvision.scene.geometry import ChassisAnchor, box_distance_norm, center_norm, match_slots, to_norm
from sopvision.scene.types import HandView, Observed, SceneEvent, SceneState, SlotView, ToolView

HELD_TOUCH = 0.02  # normalised gap between a part and a hand that still counts as holding it


@dataclass
class _SlotTrack:
    observed: Observed | None = None
    conf: float | None = None
    pending: Observed | None = None
    since: float = 0.0
    parts: Counter = field(default_factory=Counter)  # part classes seen in the slot while installed


def _gap_norm(a, b, chassis_box) -> float:
    ax1, ay1 = to_norm(a[:2], chassis_box)
    ax2, ay2 = to_norm(a[2:], chassis_box)
    bx1, by1 = to_norm(b[:2], chassis_box)
    bx2, by2 = to_norm(b[2:], chassis_box)
    dx = max(0.0, bx1 - ax2, ax1 - bx2)
    dy = max(0.0, by1 - ay2, ay1 - by2)
    return math.hypot(dx, dy)


def _intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class SceneBuilder:
    def __init__(self, sop: SopConfig, settle_seconds: float = 0.5):
        self.sop, self.settle = sop, settle_seconds
        self.removal = max(settle_seconds, sop.layout.removal_seconds or 0.0)  # installed -> empty takes this long
        self.anchor_class = sop.layout.anchor
        self.part_classes = {c for spec in sop.layout.slots.values() for c in spec.fills}  # what can be held
        self.tool_classes = set(sop.layout.tools)
        self.torqued: set[str] = set()  # slots with a torque-OK result attributed (mark_torqued)
        self._tool_zones: list[tuple[str, str]] = []
        self.anchor = ChassisAnchor(alpha=sop.layout.anchor_follow)
        self.tracks = {name: _SlotTrack() for name in sop.layout.slots}
        self.version = 0
        self.last_detections: list[Detection] = []
        self._zones: list[str] = []
        self._held: list[str] = []

    def update(self, t: float, detections: list[Detection]) -> tuple[SceneState, list[SceneEvent]]:
        self.version += 1
        self.last_detections = detections
        events: list[SceneEvent] = []
        chassis = max((d for d in detections if d.cls == self.anchor_class), key=lambda d: d.conf, default=None)
        anchor = self.anchor.update(t, chassis.xyxy if chassis else None)
        if anchor.lost_event:
            events.append(SceneEvent(t, "chassis_lost"))
        if anchor.found_event:
            events.append(SceneEvent(t, "chassis_found"))

        if not anchor.visible:
            slots = {name: SlotView("unknown", last=tr.observed, torqued=name in self.torqued)
                     for name, tr in self.tracks.items()}
            hands, held, tools = [], [], []
        else:
            box = anchor.box
            hand_boxes = [d.xyxy for d in detections if d.cls == "hand"]
            tools = [ToolView(d.cls, self._zone(d.xyxy, box)) for d in detections if d.cls in self.tool_classes]
            parts = [d for d in detections if d.cls not in (self.anchor_class, "hand") and d.cls not in self.tool_classes]
            matched = match_slots(parts, self.sop, box)
            slots = {name: self._slot_view(t, name, matched.get(name), hand_boxes, box, events)
                     for name in self.sop.layout.slots}
            hands = [HandView(self._zone(hb, box)) for hb in hand_boxes]
            matched_ids = {id(d) for d in matched.values()}
            held = sorted({d.cls for d in parts if d.cls in self.part_classes and id(d) not in matched_ids
                           and any(_gap_norm(d.xyxy, hb, box) <= HELD_TOUCH for hb in hand_boxes)})

        zones = [h.near for h in hands]
        before, after = Counter(self._zones), Counter(zones)
        events += [SceneEvent(t, "hand_entered", zone=z) for z in (after - before).elements()]
        events += [SceneEvent(t, "hand_left", zone=z) for z in (before - after).elements()]
        if held != self._held:
            events.append(SceneEvent(t, "held_changed", items=held))
        tool_zones = [(v.tool, v.near) for v in tools]
        before_t, after_t = Counter(self._tool_zones), Counter(tool_zones)
        events += [SceneEvent(t, "tool_entered", zone=z, items=[tool]) for tool, z in (after_t - before_t).elements()]
        events += [SceneEvent(t, "tool_left", zone=z, items=[tool]) for tool, z in (before_t - after_t).elements()]
        self._zones, self._held, self._tool_zones = zones, held, tool_zones
        return SceneState(self.version, t, anchor.visible, chassis.conf if chassis else None, slots, hands, held,
                          anchor.box, tools if self.tool_classes else None), events

    def mark_torqued(self, slot: str) -> None:
        """A torque-OK result was attributed to `slot`; cleared again if its part is seen leaving."""
        self.torqued.add(slot)

    def _slot_view(self, t, name, det, hand_boxes, box, events) -> SlotView:
        spec, track = self.sop.layout.slots[name], self.tracks[name]
        radius = self.sop.slot_radius(name)
        reading: Observed | None = None
        if det is not None:
            reading = "installed" if det.cls in spec.fills else "empty"
        elif any(box_distance_norm(spec.at, hb, box) <= radius for hb in hand_boxes):
            return SlotView("occluded", last=track.observed, touched=True, torqued=name in self.torqued,
                            part=self._part(track))
        elif spec.empty_as is None:
            reading = "empty"  # e.g. the cover: nothing there and nothing in the way means not installed
        else:
            return SlotView("unknown", last=track.observed, torqued=name in self.torqued, part=self._part(track))

        if reading == "installed":
            track.parts[det.cls] += 1
        if reading == track.observed:
            track.pending = None
            track.conf = det.conf if det else None
        elif reading != track.pending:
            track.pending, track.since = reading, t
        wait = self.removal if (track.observed, track.pending) == ("installed", "empty") else self.settle
        if track.pending is not None and t - track.since >= wait:
            events.append(SceneEvent(t, "slot_changed", slot=name, from_state=track.observed or "unknown",
                                     to_state=track.pending))
            if track.pending == "empty":
                self.torqued.discard(name)  # the part left: a new one has to be torqued again
                track.parts.clear()
            track.observed, track.pending = track.pending, None
            track.conf = det.conf if det else None
        touched = det is not None and any(_intersects(det.xyxy, hb) for hb in hand_boxes)
        box = tuple(det.xyxy) if det is not None else None
        if track.observed is None:
            return SlotView("unknown", touched=touched, box=box)
        return SlotView(track.observed, track.conf, track.observed, touched, name in self.torqued,
                        self._part(track), box)

    @staticmethod
    def _part(track: _SlotTrack) -> str | None:
        return track.parts.most_common(1)[0][0] if track.observed == "installed" and track.parts else None

    def _zone(self, hand_box, box) -> str:
        hx, hy = center_norm(Detection("hand", hand_box, 0.0, None), box)
        # Proximity uses the layout's default radius for every slot: a per-slot override (the cover's 0.25) is
        # there to match a big part's offset centre, and doubling it would put "near cover" over the whole server.
        reach = 2 * self.sop.layout.match_radius
        candidates = [(math.hypot(spec.at[0] - hx, spec.at[1] - hy), name) for name, spec in self.sop.layout.slots.items()
                      if box_distance_norm(spec.at, hand_box, box) <= reach]
        if candidates:
            return min(candidates)[1]
        return "chassis" if _intersects(hand_box, box) else "outside"
