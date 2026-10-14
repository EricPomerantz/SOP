"""Torque results from the wrench, and which slot each one belongs to.

The wrench reports *that* a tightening reached torque, not *where*. The camera shows where the wrench head is, but
an IP camera delivers frames late, so a result is held until the frames showing the moment it happened have arrived,
then the slots the wrench was seen at around that moment vote. A clear majority wins; anything less is reported as
unattributed rather than guessed.

Times are scene times (the video timeline). A result's `t` is when it happened on that timeline: for a recorded
torque log that is its logged time; live, it is the latest frame's time plus the camera's measured delay.
"""

import csv
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from sopvision.scene.types import SceneState


@dataclass(frozen=True)
class TorqueResult:
    t: float
    ok: bool  # reached the target torque
    torque_nm: float | None = None


@dataclass(frozen=True)
class AttributedTorque:
    result: TorqueResult
    slot: str | None  # None: couldn't tell which slot
    reason: str  # how the slot was decided, or why it couldn't be


class TorqueAttributor:
    def __init__(self, slots: set[str], before: float = 0.6, after: float = 0.3, majority: float = 0.6,
                 history: float = 10.0):
        self.slots = slots
        self.before, self.after, self.majority, self.history = before, after, majority, history
        self._seen: deque[tuple[float, list[str]]] = deque()  # (scene t, slots a tool was at)
        self._pending: list[TorqueResult] = []

    def observe(self, scene: SceneState) -> None:
        self._seen.append((scene.t, [v.near for v in scene.tools or [] if v.near in self.slots]))
        while self._seen and self._seen[0][0] < scene.t - self.history:
            self._seen.popleft()

    def add(self, result: TorqueResult) -> None:
        self._pending.append(result)

    def resolve(self, now: float, flush: bool = False) -> list[AttributedTorque]:
        """Results whose window has been fully seen (all of them when `flush`), attributed."""
        ready = [r for r in self._pending if flush or r.t + self.after <= now]
        self._pending = [r for r in self._pending if r not in ready]
        return [self._attribute(r) for r in sorted(ready, key=lambda r: r.t)]

    def _attribute(self, r: TorqueResult) -> AttributedTorque:
        frames = [near for t, near in self._seen if r.t - self.before <= t <= r.t + self.after]
        votes = Counter(slot for near in frames for slot in set(near))
        at_slot = sum(1 for near in frames if near)
        if not frames:
            return AttributedTorque(r, None, "no video frames around the torque result")
        if not votes:
            return AttributedTorque(r, None, "the wrench wasn't seen at any slot")
        slot, n = votes.most_common(1)[0]
        if n / at_slot < self.majority:
            return AttributedTorque(r, None, f"the wrench moved between slots ({dict(votes)})")
        return AttributedTorque(r, slot, f"wrench at {slot} in {n}/{len(frames)} frames")


def read_torque_log(path) -> list[TorqueResult]:
    """A recorded torque log: CSV with columns time_s, ok, torque_nm (times on the video's timeline)."""
    rows = csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines())
    out = []
    for row in rows:
        ok = str(row["ok"]).strip().lower() in ("1", "true", "yes", "ok")
        nm = row.get("torque_nm")
        out.append(TorqueResult(float(row["time_s"]), ok, float(nm) if nm not in (None, "") else None))
    return sorted(out, key=lambda r: r.t)
