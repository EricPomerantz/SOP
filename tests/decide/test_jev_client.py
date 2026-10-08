import asyncio
import json
from pathlib import Path

import pytest
from typesafe_sdk import Choice, Noul, SystemOneResponse

from sopvision.decide.jev_client import JevClient, JevUnavailableError
from sopvision.decide.questions import QuestionSet

QS = QuestionSet({"fan_3": "Install the third fan.", "idle": "Idle", "off_sop": "Other"}, "fan_3",
                 "Has this step been completed: Install the third fan?")
RECORDED = json.loads(Path("tests/decide/fixtures/jev_response.json").read_text(encoding="utf-8"))


class ApiError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


class ReplaySdk:
    """Returns the response recorded from the real API in Phase 0, parsed by the SDK's own model."""

    def __init__(self, errors=(), delay=0.0):
        self.errors = list(errors)
        self.delay = delay
        self.calls = []

    async def system_one(self, *, state, questions):
        self.calls.append(questions)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.errors:
            raise self.errors.pop(0)
        return SystemOneResponse.model_validate(RECORDED)


async def test_recorded_response_maps_fields():  # Review Focus 1
    a = await JevClient("k", sdk=ReplaySdk()).ask({}, QS)
    rec = RECORDED["answers"]
    assert a.current_step == rec["current_step"]["choice"] == "fan_3"
    assert a.current_confidence == rec["current_step"]["confidence"]
    assert a.current_probabilities == rec["current_step"]["probabilities"]
    assert a.active_step_done == rec["active_step_done"]["noul"] and a.anomaly == rec["anomaly"]["noul"]
    assert a.latency_ms >= 0


async def test_sends_typed_questions():
    sdk = ReplaySdk()
    await JevClient("k", sdk=sdk).ask({}, QS)
    q = sdk.calls[0]
    assert set(q) == {"current_step", "active_step_done", "anomaly"}
    assert isinstance(q["current_step"], Choice) and q["current_step"].criteria == QS.current_step_options
    assert q["current_step"].instructions == "Which step is the operator performing right now?"
    assert isinstance(q["active_step_done"], Noul) and q["active_step_done"].instructions == QS.active_step_question
    assert q["anomaly"].instructions == "Is anything in the scene inconsistent with the SOP so far?"


async def test_retries_429_then_succeeds():
    sdk, sleeps = ReplaySdk(errors=[ApiError(429), ApiError(429)]), []

    async def rec(s):
        sleeps.append(s)

    await JevClient("k", sdk=sdk, sleep=rec).ask({}, QS)
    assert len(sdk.calls) == 3 and sleeps == [0.25, 0.5]


async def test_gives_up_after_retries():
    sdk = ReplaySdk(errors=[ApiError(500), ApiError(502), ApiError(503)])
    with pytest.raises(JevUnavailableError):
        await JevClient("k", sdk=sdk, sleep=lambda s: asyncio.sleep(0)).ask({}, QS)
    assert len(sdk.calls) == 3


async def test_auth_error_is_not_retried():
    sdk = ReplaySdk(errors=[ApiError(401)])
    with pytest.raises(JevUnavailableError, match="401"):
        await JevClient("k", sdk=sdk, sleep=lambda s: asyncio.sleep(0)).ask({}, QS)
    assert len(sdk.calls) == 1


async def test_slow_call_times_out_and_is_retried():
    sdk = ReplaySdk(delay=0.05)
    with pytest.raises(JevUnavailableError, match="timed out"):
        await JevClient("k", sdk=sdk, timeout=0.01, sleep=lambda s: asyncio.sleep(0)).ask({}, QS)
    assert len(sdk.calls) == 3


async def test_wire_questions_match_what_the_client_sends():
    from sopvision.decide.questions import wire_questions

    sdk = ReplaySdk()
    await JevClient("k", sdk=sdk).ask({}, QS)
    sent = {k: v.model_dump() for k, v in sdk.calls[0].items()}
    assert sent == wire_questions(QS)
