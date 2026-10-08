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


# -- compare mode: an authoritative client plus a shadow --------------------------------------------------
DECIDER_ANSWERS = JevAnswers("fan_2", 0.7, {"fan_2": 0.7, "fan_1": 0.3}, 0.9, 0.1, 30.0, "decider")


class FakeProvider(FakeClient):
    def __init__(self, provider, answers, delay=0.0, fail=0, error=None):
        super().__init__(delay, fail)
        self.provider, self.answers, self.error = provider, answers, error
        self.requests = []

    async def ask(self, state, qs):
        self.requests.append((state, qs))
        if self.error:
            raise self.error
        await super().ask(state, qs)
        return self.answers


def make_compare(sop, mode, primary_delay=0.0, shadow_delay=0.0, shadow_fail=0, shadow_error=None):
    jev = FakeProvider("jev", ANSWERS, primary_delay)
    decider = FakeProvider("decider", DECIDER_ANSWERS, shadow_delay, shadow_fail, shadow_error)
    delivered, statuses, records = [], [], []
    d = JevDecider(sop, jev, mode, on_decision=delivered.append, on_status=lambda s, t: statuses.append((s, t)),
                   shadow=decider, on_comparison=records.append)
    return d, jev, decider, delivered, statuses, records


async def test_compare_sends_both_providers_the_identical_request(sop):
    d, jev, decider, delivered, _, records = make_compare(sop, "lockstep")
    d.observe(scene(1.0, hands=1), [], {"fan_1": "active"}, "fan_1")
    await d.poll(1.0)
    await d.poll(1.5)
    assert jev.requests == decider.requests and jev.requests[0][0] is decider.requests[0][0]  # same objects
    assert [x.answers for x in delivered] == [ANSWERS]  # only the authoritative answer reaches the tracker
    (r,) = records
    assert r["authoritative"] == "jev"
    assert r["jev"]["answers"]["current_step"] == "fan_1" and r["decider"]["answers"]["current_step"] == "fan_2"
    assert r["comparison"]["same_current_step"] is False
    assert r["request"]["state"] == jev.requests[0][0] and r["request"]["questions"]["active_step_id"] == "fan_1"
    assert r["ground_truth"] is None


async def test_shadow_failure_is_recorded_and_never_changes_the_decision(sop):
    for err in (JevUnavailableError("down"), RuntimeError("CUDA out of memory")):
        d, _, _, delivered, statuses, records = make_compare(sop, "lockstep", shadow_error=err)
        for t in (0.0, 0.25, 0.5, 0.75):
            d.observe(scene(t), [ev(t)], {}, "fan_1")
            await d.poll(t)
        await d.poll(2.0)
        assert [x.answers for x in delivered] == [ANSWERS] * 4
        assert statuses == []  # shadow errors never take the authoritative provider offline
        assert all(r["decider"]["error"] and r["decider"]["answers"] is None and r["comparison"] is None for r in records)
        assert "CUDA out of memory" in records[0]["decider"]["error"] or "down" in records[0]["decider"]["error"]


async def test_lockstep_compare_calls_run_in_parallel(sop):
    d, *_ = make_compare(sop, "lockstep", primary_delay=0.2, shadow_delay=0.2)
    d.observe(scene(0.0), [ev(0.0)], {}, "fan_1")
    start = asyncio.get_running_loop().time()
    await d.poll(0.0)
    assert asyncio.get_running_loop().time() - start < 0.35  # not 0.4: both calls overlap


async def test_live_slow_shadow_never_delays_the_authoritative_answer(sop):
    d, _, _, delivered, _, records = make_compare(sop, "live", primary_delay=0.05, shadow_delay=0.3)
    d.observe(scene(0.0), [ev(0.0)], {}, "fan_1")
    await d.poll(0.0)
    await asyncio.sleep(0.1)
    assert [x.answers for x in delivered] == [ANSWERS] and records == []
    d.observe(scene(0.3), [ev(0.3)], {}, "fan_1")
    await d.poll(0.3)  # authoritative slot is free again although the shadow is still running
    await asyncio.sleep(0.1)
    assert len(delivered) == 2
    assert records[-1]["decider"]["error"].startswith("skipped")  # shadow was busy for the second request
    await asyncio.sleep(0.2)
    assert [r["decider"]["answers"] is not None for r in records] == [False, True]  # skip logged first, then the slow one
    await d.aclose()


async def test_live_records_when_context_was_ready(sop):
    clock = iter([10.0, 10.0, 10.0, 10.2, 10.5, 10.6]).__next__
    d, _ = make(sop, "live")
    d._clock = clock
    d.observe(scene(0.0), [ev(0.0)], {}, "fan_1")  # due now → ctx_ready 10.0
    await d.poll(0.0)
    await asyncio.sleep(0.01)
    (decision,) = d.delivered
    assert decision.ctx_ready == 10.0 and decision.request_ms is not None
    await d.aclose()


async def test_lockstep_has_no_end_to_end_timestamp(sop):
    d, _ = make(sop, "lockstep")
    d.observe(scene(1.0), [ev(1.0)], {}, "fan_1")
    await d.poll(1.0)
    await d.poll(2.0)
    assert d.delivered[0].ctx_ready is None


async def test_decider_as_authoritative_uses_the_offline_path(sop):
    decider = FakeProvider("decider", DECIDER_ANSWERS, fail=3)
    delivered, statuses = [], []
    d = JevDecider(sop, decider, "lockstep", on_decision=delivered.append, on_status=lambda s, t: statuses.append((s, t)))
    for t in (0.0, 0.25, 0.5, 0.75):
        d.observe(scene(t), [ev(t)], {}, "fan_1")
        await d.poll(t)
    await d.poll(2.0)
    assert statuses == [("offline", 0.5), ("online", 0.75)]
    assert [x.answers.provider for x in delivered] == ["decider"]


async def test_nothing_recorded_after_close(sop):
    d, _, _, delivered, _, records = make_compare(sop, "live", shadow_delay=0.2)
    d.observe(scene(0.0), [ev(0.0)], {}, "fan_1")
    await d.poll(0.0)
    await asyncio.sleep(0.05)
    await d.aclose()
    await asyncio.sleep(0.25)
    assert records == []
