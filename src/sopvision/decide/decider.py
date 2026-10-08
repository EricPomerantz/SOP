"""When to ask Jev, and delivering its answers (spec §7.1, §7.4).

Calls happen on scene changes (throttled to one per 0.25 s) and every 2 s while hands are visible.
Live mode keeps at most one call in flight and never blocks the caller; lockstep mode (eval)
awaits each call and applies the answer a fixed 0.5 s later in video time, so results don't
depend on how fast the machine or the network is.

`client` is the authoritative provider (Jev or the local Decider). An optional `shadow` provider gets
the very same state and questions at the same moment; its answer only goes into a comparison record
and never reaches the tracker, and its failures never count against the authoritative provider.
Live, the authoritative answer is delivered as soon as it arrives (the shadow can't delay it), and the
shadow has its own single in-flight slot: a request that finds it busy is recorded as skipped.
"""

import asyncio
import contextlib
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sopvision.config import SopConfig
from sopvision.decide.compare import comparison_record
from sopvision.decide.jev_client import JevUnavailableError
from sopvision.decide.questions import RECENT_EVENTS, QuestionSet, build_questions, build_state
from sopvision.decide.types import Decision, ProviderResult
from sopvision.scene.types import SceneEvent, SceneState

OFFLINE_AFTER = 3  # consecutive failed calls


@dataclass(frozen=True)
class _Request:
    id: str
    t: float  # video time
    version: int
    state: dict
    qs: QuestionSet
    ctx_ready: float | None  # clock() when the call first became due (live only)
    started: float  # clock() when the request was built and sent


class JevDecider:
    def __init__(self, sop: SopConfig, client, mode: Literal["live", "lockstep"],
                 on_decision: Callable[[Decision], None],
                 on_status: Callable[[Literal["online", "offline"], float], None],
                 throttle: float = 0.25, heartbeat: float = 2.0, lockstep_latency: float = 0.5,
                 shadow=None, on_comparison: Callable[[dict], None] | None = None, clock=time.perf_counter):
        self.sop, self.client, self.mode = sop, client, mode
        self.on_decision, self.on_status = on_decision, on_status
        self.throttle, self.heartbeat, self.lockstep_latency = throttle, heartbeat, lockstep_latency
        self.shadow, self.on_comparison, self._clock = shadow, on_comparison or (lambda record: None), clock
        self._scene: SceneState | None = None
        self._recent: deque[SceneEvent] = deque(maxlen=RECENT_EVENTS)
        self._unsent = 0
        self._statuses: dict[str, str] = {}
        self._active: str | None = None
        self._last_call = -math.inf
        self._inflight: asyncio.Task | None = None
        self._shadow_inflight: asyncio.Task | None = None
        self._finishing: set[asyncio.Task] = set()  # comparisons waiting on a slower shadow
        self._queue: list[tuple[float, Decision]] = []
        self._last_version = -1
        self._failures = 0
        self._offline = False
        self._closed = False
        self._due_since: float | None = None
        self._seq = 0

    def observe(self, scene: SceneState, events: list[SceneEvent], statuses: dict[str, str], active_step_id: str) -> None:
        self._scene = scene
        self._recent.extend(events)
        self._unsent += len(events)
        self._statuses = dict(statuses)
        self._active = active_step_id
        self._mark_due(scene.t)

    def _call_due(self, now: float) -> bool:
        if self._scene is None or self._active is None:
            return False
        since = now - self._last_call
        return (self._unsent > 0 and since >= self.throttle) or (bool(self._scene.hands) and since >= self.heartbeat)

    def _mark_due(self, now: float) -> None:
        """Remember when enough context first existed for a call: the start of end-to-end latency."""
        if self.mode == "live" and self._due_since is None and self._call_due(now):
            self._due_since = self._clock()

    async def poll(self, now: float) -> None:
        if self._closed:
            return
        self._mark_due(now)
        busy = self._inflight is not None and not self._inflight.done()
        if self._call_due(now) and not (self.mode == "live" and busy):
            request = self._request(now)
            self._unsent, self._last_call = 0, now
            if self.mode == "lockstep":
                decision = await self._call(request)
                if decision:
                    self._queue.append((now + self.lockstep_latency, decision))
            else:
                self._inflight = asyncio.create_task(self._call_and_deliver(request))
        if self.mode == "lockstep":
            due = sorted((q for q in self._queue if q[0] <= now + 1e-9), key=lambda q: q[0])
            self._queue = [q for q in self._queue if q[0] > now + 1e-9]
            for _, decision in due:
                self._deliver(decision)

    def _request(self, now: float) -> _Request:
        """One state and question set, built once and sent unchanged to every provider."""
        scene, active = self._scene, self._active
        self._seq += 1
        ctx_ready, self._due_since = self._due_since, None
        return _Request(f"{scene.version}-{self._seq}", now, scene.version,
                        build_state(self.sop, self._statuses, scene, list(self._recent)),
                        build_questions(self.sop, active), ctx_ready, self._clock())

    async def _ask(self, client, request: _Request, shadow: bool) -> ProviderResult:
        provider = getattr(client, "provider", "jev")
        start = self._clock()
        try:
            answers = await client.ask(request.state, request.qs)
        except JevUnavailableError as e:
            return ProviderResult(provider, None, str(e), (self._clock() - start) * 1000)
        except Exception as e:  # noqa: BLE001 — a shadow failure of any kind is recorded, never raised
            if not shadow:
                raise
            return ProviderResult(provider, None, f"{type(e).__name__}: {e}", (self._clock() - start) * 1000)
        return ProviderResult(provider, answers, None, (self._clock() - start) * 1000)

    def _settle(self, request: _Request, result: ProviderResult) -> Decision | None:
        """The authoritative result: track offline/online, and turn answers into a Decision."""
        if result.answers is None:
            self._failures += 1
            if self._failures >= OFFLINE_AFTER and not self._offline:
                self._offline = True
                self.on_status("offline", request.t)
            return None
        self._failures = 0
        if self._offline:
            self._offline = False
            self.on_status("online", request.t)
        return Decision(request.version, request.t, request.qs.active_step_id, result.answers, result.request_ms,
                        request.ctx_ready)

    async def _call(self, request: _Request) -> Decision | None:
        if self.shadow is None:
            result = await self._ask(self.client, request, shadow=False)
        else:
            result, shadow = await asyncio.gather(self._ask(self.client, request, shadow=False),
                                                  self._ask(self.shadow, request, shadow=True))
            self._compare(request, result, shadow)
        return self._settle(request, result)

    async def _call_and_deliver(self, request: _Request) -> None:
        shadow_task, skipped = None, False
        if self.shadow is not None:
            if self._shadow_inflight is not None and not self._shadow_inflight.done():
                skipped = True
            else:
                shadow_task = self._shadow_inflight = asyncio.create_task(self._ask(self.shadow, request, shadow=True))
        result = await self._ask(self.client, request, shadow=False)
        decision = self._settle(request, result)
        if decision:
            self._deliver(decision)
        if skipped:
            self._compare(request, result, ProviderResult(getattr(self.shadow, "provider", "shadow"), None,
                                                          "skipped: previous shadow request still running", 0.0))
        elif shadow_task is not None:
            # Don't hold the authoritative slot while the shadow finishes.
            task = asyncio.create_task(self._finish_comparison(request, result, shadow_task))
            self._finishing.add(task)
            task.add_done_callback(self._finishing.discard)

    async def _finish_comparison(self, request: _Request, result: ProviderResult, shadow_task: asyncio.Task) -> None:
        self._compare(request, result, await shadow_task)

    def _compare(self, request: _Request, primary: ProviderResult, shadow: ProviderResult) -> None:
        if self._closed:
            return
        record = comparison_record(request.id, request.t, request.version, request.state, request.qs, primary, shadow,
                                   self.sop.decisions.accept_confidence)
        # From context-ready to the request going out; + each provider's request_ms = its end-to-end time.
        record["wait_ms"] = None if request.ctx_ready is None else round((request.started - request.ctx_ready) * 1000, 2)
        self.on_comparison(record)

    def _deliver(self, decision: Decision) -> None:
        if self._closed or decision.version < self._last_version:
            return
        self._last_version = decision.version
        self.on_decision(decision)

    async def aclose(self) -> None:
        self._closed = True
        for task in [self._inflight, self._shadow_inflight, *self._finishing]:
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
