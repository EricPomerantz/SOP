import httpx
import pytest
from pydantic import SecretStr, ValidationError
from typesafe_sdk import SystemOneResponse

from sopvision.config import ConfigError
from sopvision.decide.jev_client import JevClient, JevUnavailableError
from sopvision.decide.providers import (DeciderReadiness, clients_for, decider_readiness, laya_readiness, local_provider,
                                       make_client)
from sopvision.decide.questions import QuestionSet
from sopvision.server.runs import check_decision_providers
from sopvision.settings import Settings

QS = QuestionSet({"fan_3": "Install the third fan.", "idle": "Idle", "off_sop": "Other"}, "fan_3",
                 "Has this step been completed: Install the third fan?")
# What Decider's /v1/systemone returns (decider/systemone.py format_answer): TypeSafe's fields plus extras.
DECIDER_RESPONSE = {
    "model": "decider-4b-v2.1",
    "answers": {
        "current_step": {"type": "choice", "choice": "fan_3", "confidence": 0.82, "x_p_max": 0.88, "certainty": 0.6,
                         "probabilities": {"fan_3": 0.88, "idle": 0.07, "off_sop": 0.05}},
        "active_step_done": {"type": "noul", "noul": 0.31},
        "anomaly": {"type": "noul", "noul": 0.04},
    },
    "usage": {"input_tokens": 812, "output_tokens": 0},
}


def settings(mode="jev", key="sk-test", **kw):
    return Settings(decision_mode=mode, typesafe_api_key=SecretStr(key), **kw)


class Sdk:
    def __init__(self, response):
        self.response = response

    async def system_one(self, *, state, questions):
        return SystemOneResponse.model_validate(self.response)


# -- configuration ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode,shadow_choice,expected", [("jev", "decider", ("jev", None)), ("decider", "decider", ("decider", None)),
                                                          ("laya", "decider", ("laya", None)),
                                                          ("compare", "decider", ("jev", "decider")),
                                                          ("compare", "laya", ("jev", "laya"))])
def test_mode_selects_the_providers(mode, shadow_choice, expected):
    primary, shadow = clients_for(settings(mode, compare_shadow=shadow_choice))
    assert (primary.provider, shadow and shadow.provider) == expected


def test_mode_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("DECISION_MODE", "compare")
    monkeypatch.setenv("DECIDER_URL", "http://decider:8001")
    s = Settings()
    assert (s.decision_mode, s.decider_url) == ("compare", "http://decider:8001")


def test_default_mode_is_jev(monkeypatch):
    monkeypatch.delenv("DECISION_MODE", raising=False)
    assert Settings(_env_file=None).decision_mode == "jev"


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("DECISION_MODE", "both")
    with pytest.raises(ValidationError):
        Settings()


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown decision provider"):
        make_client(settings(), "gpt")


# -- normalisation: both providers come back as the same JevAnswers shape -------------------------------------
async def test_decider_response_normalises_like_jev():
    a = await JevClient("local", sdk=Sdk(DECIDER_RESPONSE), provider="decider").ask({}, QS)
    assert (a.provider, a.current_step, a.current_confidence, a.active_step_done, a.anomaly) == ("decider", "fan_3", 0.82, 0.31, 0.04)
    assert a.current_probabilities == {"fan_3": 0.88, "idle": 0.07, "off_sop": 0.05}


async def test_invalid_response_is_unavailable_not_a_crash():
    broken = DECIDER_RESPONSE | {"answers": {"current_step": DECIDER_RESPONSE["answers"]["current_step"]}}
    with pytest.raises(JevUnavailableError, match="invalid response"):
        await JevClient("local", sdk=Sdk(broken), provider="decider").ask({}, QS)


# -- readiness: a live process is not enough ----------------------------------------------------------------
def http(health=None, probe=None, down=False):
    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("refused")
        if request.url.path == "/health":
            return httpx.Response(200, json=health)
        return probe if isinstance(probe, httpx.Response) else httpx.Response(200, json=probe)

    return httpx.Client(transport=httpx.MockTransport(handler))


GOOD_PROBE = {"model": "decider-4b", "answers": {"go": {"type": "choice", "choice": "stop", "confidence": 0.9,
                                                       "probabilities": {"go": 0.05, "stop": 0.95}}},
              "usage": {"input_tokens": 30, "output_tokens": 0}}


def test_not_running():
    r = decider_readiness("http://decider:8001", http=http(down=True))
    assert not r.ready and not r.running


def test_running_but_model_not_loaded_is_not_ready():
    r = decider_readiness("http://decider:8001", http=http(health={"ok": False, "device": "cuda"}))
    assert (r.running, r.model_loaded, r.ready) == (True, False, False)


def test_loaded_but_inference_failing_is_not_ready():
    r = decider_readiness("http://d", http=http(health={"ok": True, "device": "cuda"}, probe=httpx.Response(500)))
    assert (r.model_loaded, r.inference, r.ready) == (True, False, False)


def test_invalid_probe_answer_is_not_ready():
    bad = GOOD_PROBE | {"answers": {"go": {"type": "choice", "choice": "maybe", "probabilities": {"maybe": 1.0}}}}
    r = decider_readiness("http://d", http=http(health={"ok": True, "device": "cuda"}, probe=bad))
    assert not r.inference and not r.ready


def test_cpu_inference_is_not_gpu_ready():
    r = decider_readiness("http://d", http=http(health={"ok": True, "device": "cpu"}, probe=GOOD_PROBE))
    assert (r.inference, r.gpu, r.ready) == (True, False, False)


def test_ready_only_after_a_real_answer_on_the_gpu():
    r = decider_readiness("http://d", http=http(health={"ok": True, "device": "cuda:0"}, probe=GOOD_PROBE))
    assert r.ready and r.device == "cuda:0"


# -- startup checks per mode --------------------------------------------------------------------------------
READY = DeciderReadiness(True, True, True, True, "cuda", "ready")
DOWN = DeciderReadiness(False, False, False, False, None, "Decider not reachable")


def never(settings, provider):
    raise AssertionError("Decider must not be contacted in jev mode")


def test_jev_mode_needs_the_key_and_never_contacts_decider():
    check_decision_providers(settings("jev"), readiness=never)
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        check_decision_providers(settings("jev", key=""), readiness=never)


def test_decider_mode_needs_no_key_but_a_ready_decider():
    check_decision_providers(settings("decider", key=""), readiness=lambda s, p: READY)
    with pytest.raises(ConfigError, match="local decider service isn't ready"):
        check_decision_providers(settings("decider", key=""), readiness=lambda s, p: DOWN)


def test_compare_mode_starts_without_decider_and_keeps_jev(caplog):
    check_decision_providers(settings("compare"), readiness=lambda s, p: DOWN)
    assert "Jev still decides" in caplog.text
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        check_decision_providers(settings("compare", key=""), readiness=lambda s, p: READY)


class RecordingSdk(Sdk):
    def __init__(self, response):
        super().__init__(response)
        self.kwargs = None

    async def system_one(self, *, state, questions, **kwargs):
        self.kwargs = kwargs
        return await super().system_one(state=state, questions=questions)


def test_laya_client_names_checkpoint_and_budgets():
    c = make_client(settings(laya_model="typed-decisions", laya_max_len=2048, laya_head_max_len=512), "laya")
    assert (c.provider, c.extra_body) == ("laya", {"max_len": 2048, "head_max_len": 512})
    assert make_client(settings(), "laya").extra_body is None  # shipped budgets: send nothing extra


async def test_extra_body_reaches_the_sdk_only_when_set():
    sdk = RecordingSdk(DECIDER_RESPONSE)
    await JevClient("local", sdk=sdk, provider="laya", extra_body={"max_len": 2048}).ask({}, QS)
    assert sdk.kwargs == {"extra_body": {"max_len": 2048}}
    await JevClient("local", sdk=sdk).ask({}, QS)
    assert sdk.kwargs == {}


def test_laya_mode_needs_no_key_but_a_ready_laya():
    seen = []
    check_decision_providers(settings("laya", key=""), readiness=lambda s, p: seen.append(p) or READY)
    assert seen == ["laya"]
    with pytest.raises(ConfigError, match="DECISION_MODE=laya .*--profile laya up -d laya"):
        check_decision_providers(settings("laya", key=""), readiness=lambda s, p: DOWN)


def test_compare_checks_the_chosen_shadow():
    assert local_provider(settings("compare", compare_shadow="laya")) == "laya"
    assert local_provider(settings("jev")) is None


def test_laya_readiness_reads_laya_health():
    up = {"status": "ok", "loaded": ["english"], "device": "cuda", "checkpoint_devices": {"english": "cuda"}}
    assert laya_readiness("http://l", http=http(health=up, probe=GOOD_PROBE)).ready
    nothing = up | {"loaded": []}
    assert not laya_readiness("http://l", http=http(health=nothing, probe=GOOD_PROBE)).model_loaded
    cpu = up | {"checkpoint_devices": {"english": "cpu"}}
    r = laya_readiness("http://l", http=http(health=cpu, probe=GOOD_PROBE))
    assert (r.inference, r.gpu, r.ready) == (True, False, False)


def test_laya_asks_for_the_sops_own_checkpoint_when_the_registry_names_one(tmp_path):
    from sopvision.decide.laya_registry import load, set_checkpoint

    registry = tmp_path / "registry.json"
    s = settings("laya", laya_registry=registry)
    assert make_client(s, "laya", "lug_bolts_torque").model == "english"  # no registry yet: the shared checkpoint
    set_checkpoint(registry, "lug_bolts_torque", "data/training/models/Lug-v3")
    assert load(registry) == {"lug_bolts_torque": "Lug-v3"}
    assert make_client(s, "laya", "lug_bolts_torque").model == "ckpt-lug-v3"
    assert make_client(s, "laya", "server_fan_install").model == "english"
    assert make_client(s, "laya", "lug_bolts_torque", laya_model="ckpt-candidate").model == "ckpt-candidate"
    assert clients_for(settings("compare", compare_shadow="laya", laya_registry=registry), "lug_bolts_torque")[1].model \
        == "ckpt-lug-v3"
    set_checkpoint(registry, "lug_bolts_torque", None)
    assert make_client(s, "laya", "lug_bolts_torque").model == "english"
