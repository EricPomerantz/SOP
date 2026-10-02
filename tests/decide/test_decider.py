import asyncio

import pytest

from sopvision.decide.decider import JevDecider
from sopvision.decide.jev_client import JevUnavailableError
from sopvision.decide.types import JevAnswers
from sopvision.scene.types import HandView, SceneEvent, SceneState

ANSWERS = JevAnswers("fan_1", 0.9, {"fan_1": 0.9}, 0.1, 0.05, 120.0)


class FakeClient:
    def __init__(self, delay=0.0, fail=0):
        self.delay, self.fail = delay, fail
        self.call_times: list[float] = []

    async def ask(self, state, qs):
        self.call_times.append(state["scene"]["t"])
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            self.fail -= 1
            raise JevUnavailableError("down")
        return ANSWERS


def make(sop, mode, delay=0.0, fail=0):
    client = FakeClient(delay, fail)
    delivered, statuses = [], []
    d = JevDecider(sop, client, mode, on_decision=delivered.append, on_status=lambda s, t: statuses.append((s, t)))
    d.delivered, d.statuses = delivered, statuses  # test handles
    return d, client


def scene(t, hands=0, version=None):
    return SceneState(version if version is not None else int(t * 100), t, True, 0.9, {},
                      [HandView("fan_bay_1")] * hands, [], None)


def ev(t):
    return SceneEvent(t, "hand_entered", zone="fan_bay_1")


async def test_throttles_event_bursts(sop):
    d, client = make(sop, "lockstep")
    for i in range(21):
        t = i * 0.05
        d.observe(scene(t, hands=0), [ev(t)], {}, "fan_1")
        await d.poll(t)
    assert client.call_times == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


async def test_heartbeat_while_hands_visible(sop):
    d, client = make(sop, "lockstep")
    for i in range(61):
        t = i * 0.1
        d.observe(scene(t, hands=1), [], {}, "fan_1")
        await d.poll(t)
    assert client.call_times == pytest.approx([0.0, 2.0, 4.0, 6.0])


async def test_no_calls_when_idle(sop):
    d, client = make(sop, "lockstep")
    for i in range(61):
        d.observe(scene(i * 0.1), [], {}, "fan_1")
        await d.poll(i * 0.1)
    assert client.call_times == []


async def test_lockstep_delivers_after_latency(sop):
    d, _ = make(sop, "lockstep")
    d.observe(scene(1.0, hands=1), [], {"fan_1": "active"}, "fan_1")
    await d.poll(1.0)
    await d.poll(1.49)
    assert d.delivered == []
    await d.poll(1.5)
    assert [(x.t_request, x.active_step_id, x.answers) for x in d.delivered] == [(1.0, "fan_1", ANSWERS)]


async def test_offline_then_online(sop):
    d, _ = make(sop, "lockstep", fail=3)
    for t in (0.0, 0.25, 0.5, 0.75):
        d.observe(scene(t), [ev(t)], {}, "fan_1")
        await d.poll(t)
    assert d.statuses == [("offline", 0.5), ("online", 0.75)]


async def test_stale_version_dropped(sop):
    d, _ = make(sop, "lockstep")
    d.observe(scene(0.0, version=7), [ev(0.0)], {}, "fan_1")
    await d.poll(0.0)
    d.observe(scene(0.3, version=5), [ev(0.3)], {}, "fan_1")
    await d.poll(0.3)
    await d.poll(1.0)
    assert [x.version for x in d.delivered] == [7]


async def test_live_keeps_one_call_in_flight(sop):
    d, client = make(sop, "live", delay=0.2)
    d.observe(scene(0.0), [ev(0.0)], {}, "fan_1")
    await d.poll(0.0)
    await asyncio.sleep(0)  # let the background call start
    d.observe(scene(0.3), [ev(0.3)], {}, "fan_1")
    await d.poll(0.3)  # first call still running → skipped
    assert client.call_times == [0.0]
    await asyncio.sleep(0.3)
    assert [x.t_request for x in d.delivered] == [0.0]
    await d.aclose()


async def test_nothing_after_close(sop):  # Review Focus 4
    d, _ = make(sop, "live", delay=0.2)
    d.observe(scene(0, hands=1), [], {}, "fan_1")
    await d.poll(0)
    await d.aclose()
    await asyncio.sleep(0.3)
    assert d.delivered == []
    await d.poll(5.0)  # polling a closed decider does nothing
    assert d.delivered == []
