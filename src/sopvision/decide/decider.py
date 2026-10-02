"""When to ask Jev, and delivering its answers (spec §7.1, §7.4).

Calls happen on scene changes (throttled to one per 0.25 s) and every 2 s while hands are visible.
Live mode keeps at most one call in flight and never blocks the caller; lockstep mode (eval)
awaits each call and applies the answer a fixed 0.5 s later in video time, so results don't
depend on how fast the machine or the network is.
"""

import asyncio
import contextlib
import math
from collections import deque
from collections.abc import Callable
from typing import Literal

from sopvision.config import SopConfig
from sopvision.decide.jev_client import JevUnavailableError
from sopvision.decide.questions import RECENT_EVENTS, build_questions, build_state
from sopvision.decide.types import Decision
from sopvision.scene.types import SceneEvent, SceneState

OFFLINE_AFTER = 3  # consecutive failed calls


class JevDecider:
    def __init__(self, sop: SopConfig, client, mode: Literal["live", "lockstep"],
                 on_decision: Callable[[Decision], None],
                 on_status: Callable[[Literal["online", "offline"], float], None],
                 throttle: float = 0.25, heartbeat: float = 2.0, lockstep_latency: float = 0.5):
        self.sop, self.client, self.mode = sop, client, mode
        self.on_decision, self.on_status = on_decision, on_status
        self.throttle, self.heartbeat, self.lockstep_latency = throttle, heartbeat, lockstep_latency
        self._scene: SceneState | None = None
        self._recent: deque[SceneEvent] = deque(maxlen=RECENT_EVENTS)
        self._unsent = 0
        self._statuses: dict[str, str] = {}
        self._active: str | None = None
        self._last_call = -math.inf
        self._inflight: asyncio.Task | None = None
        self._queue: list[tuple[float, Decision]] = []
        self._last_version = -1
        self._failures = 0
        self._offline = False
        self._closed = False

    def observe(self, scene: SceneState, events: list[SceneEvent], statuses: dict[str, str], active_step_id: str) -> None:
        self._scene = scene
        self._recent.extend(events)
        self._unsent += len(events)
        self._statuses = dict(statuses)
        self._active = active_step_id

    def _call_due(self, now: float) -> bool:
        if self._scene is None or self._active is None:
            return False
        since = now - self._last_call
        return (self._unsent > 0 and since >= self.throttle) or (bool(self._scene.hands) and since >= self.heartbeat)

    async def poll(self, now: float) -> None:
        if self._closed:
            return
        busy = self._inflight is not None and not self._inflight.done()
        if self._call_due(now) and not (self.mode == "live" and busy):
            snapshot = (self._scene, list(self._recent), self._statuses, self._active)
            self._unsent, self._last_call = 0, now
            if self.mode == "lockstep":
                decision = await self._call(now, snapshot)
                if decision:
                    self._queue.append((now + self.lockstep_latency, decision))
            else:
                self._inflight = asyncio.create_task(self._call_and_deliver(now, snapshot))
        if self.mode == "lockstep":
            due = sorted((q for q in self._queue if q[0] <= now + 1e-9), key=lambda q: q[0])
            self._queue = [q for q in self._queue if q[0] > now + 1e-9]
            for _, decision in due:
                self._deliver(decision)

    async def _call(self, now: float, snapshot) -> Decision | None:
        scene, recent, statuses, active = snapshot
        try:
            answers = await self.client.ask(build_state(self.sop, statuses, scene, recent),
                                            build_questions(self.sop, active))
        except JevUnavailableError:
            self._failures += 1
            if self._failures >= OFFLINE_AFTER and not self._offline:
                self._offline = True
                self.on_status("offline", now)
            return None
        self._failures = 0
        if self._offline:
            self._offline = False
            self.on_status("online", now)
        return Decision(scene.version, now, active, answers)

    async def _call_and_deliver(self, now: float, snapshot) -> None:
        decision = await self._call(now, snapshot)
        if decision:
            self._deliver(decision)

    def _deliver(self, decision: Decision) -> None:
        if self._closed or decision.version < self._last_version:
            return
        self._last_version = decision.version
        self.on_decision(decision)

    async def aclose(self) -> None:
        self._closed = True
        if self._inflight is not None and not self._inflight.done():
            self._inflight.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._inflight
