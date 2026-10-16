"""Which model answers SOP decisions (DECISION_MODE), and whether the local Decider is ready to answer.

    jev      Jev decides (unchanged behaviour)
    decider  the local Decider decides
    laya     the local Laya service decides (e.g. a fine-tuned checkpoint served with LAYA_CHECKPOINT)
    compare  Jev decides; the COMPARE_SHADOW model (decider or laya) gets the same requests in the shadow and both
             answers are logged side by side
"""

from dataclasses import dataclass

from sopvision.decide.jev_client import JevClient
from sopvision.settings import Settings

DECIDER_MODEL = "decider-4b"
DISPLAY_NAMES = {"jev": "Jev", "decider": "Decider", "laya": "Laya"}


def display_name(provider: str) -> str:
    """How the dashboard and alerts name a decision model."""
    return DISPLAY_NAMES.get(provider, provider)


def authoritative_provider(settings: Settings) -> str:
    """The model whose answers drive the tracker (compare mode: Jev)."""
    return settings.decision_mode if settings.decision_mode in ("decider", "laya") else "jev"


# The smallest real decision: proves the model is loaded and answers, not just that the process is up.
PROBE = {"state": {"light": "red"},
         "questions": {"go": {"type": "choice", "instructions": "Should the car go or stop?",
                              "criteria": {"go": None, "stop": None}}}}


def make_client(settings: Settings, provider: str, sop_id: str | None = None, laya_model: str | None = None) -> JevClient:
    """A client for `provider`. Laya answers with the SOP's own checkpoint when the registry names one
    (decide/laya_registry.py), or `laya_model` when given (a candidate checkpoint under evaluation)."""
    if provider == "jev":
        return JevClient(settings.typesafe_api_key.get_secret_value(), settings.typesafe_base_url, settings.typesafe_model)
    if provider == "decider":
        # The SDK insists on a key; the local server ignores it.
        return JevClient("local", settings.decider_url, DECIDER_MODEL, timeout=settings.decider_timeout, provider="decider")
    if provider == "laya":
        budgets = {"max_len": settings.laya_max_len, "head_max_len": settings.laya_head_max_len}
        from sopvision.decide.laya_registry import laya_model_for

        model = laya_model or laya_model_for(settings, sop_id)
        return JevClient("local", settings.laya_url, model, timeout=settings.decider_timeout, provider="laya",
                         extra_body={k: v for k, v in budgets.items() if v is not None} or None)
    raise ValueError(f"unknown decision provider {provider!r}")


def clients_for(settings: Settings, sop_id: str | None = None,
                laya_model: str | None = None) -> tuple[JevClient, JevClient | None]:
    """(authoritative client, shadow client or None) for the configured DECISION_MODE, for one SOP's requests."""
    mode = settings.decision_mode
    if mode == "compare":
        return make_client(settings, "jev"), make_client(settings, settings.compare_shadow, sop_id, laya_model)
    return make_client(settings, mode, sop_id, laya_model), None


def uses_jev(settings: Settings) -> bool:
    return settings.decision_mode in ("jev", "compare")


def local_provider(settings: Settings) -> str | None:
    """The local model this mode calls (decider or laya), if any."""
    if settings.decision_mode == "compare":
        return settings.compare_shadow
    return settings.decision_mode if settings.decision_mode in ("decider", "laya") else None


@dataclass(frozen=True)
class DeciderReadiness:
    running: bool  # /health answered
    model_loaded: bool  # /health ok: weights loaded, CUDA graphs captured, batcher alive
    gpu: bool  # serving on a CUDA device
    inference: bool  # a real /v1/systemone probe returned valid probabilities
    device: str | None
    detail: str

    @property
    def ready(self) -> bool:
        return self.running and self.model_loaded and self.gpu and self.inference


def _decider_health(h: dict) -> tuple[bool, str | None]:
    """Decider's /health: ok once weights are loaded, CUDA graphs captured and the batcher is alive."""
    return bool(h.get("ok")), h.get("device")


def _laya_health(h: dict) -> tuple[bool, str | None]:
    """laya-serve's /health: status ok and a checkpoint resident; where the routed (english) one computes."""
    devices = h.get("checkpoint_devices") or {}
    return h.get("status") == "ok" and bool(h.get("loaded")), devices.get("english") or h.get("device")


def decider_readiness(url: str, timeout: float = 5.0, http=None) -> DeciderReadiness:
    """Ready means the model answered a real request on the GPU; a live process alone is not enough."""
    return _readiness(url, _decider_health, "Decider", timeout, http)


def laya_readiness(url: str, timeout: float = 5.0, http=None) -> DeciderReadiness:
    return _readiness(url, _laya_health, "Laya", timeout, http)


def local_readiness(settings: Settings, provider: str) -> DeciderReadiness:
    if provider == "laya":
        return laya_readiness(settings.laya_url)
    return decider_readiness(settings.decider_url)


def _readiness(url: str, parse_health, name: str, timeout: float, http) -> DeciderReadiness:
    import httpx

    client = http or httpx.Client(timeout=timeout)
    try:
        try:
            health = client.get(f"{url.rstrip('/')}/health")
            health.raise_for_status()
            h = health.json()
        except (httpx.HTTPError, ValueError) as e:
            return DeciderReadiness(False, False, False, False, None, f"{name} not reachable at {url}: {e}")
        loaded, device = parse_health(h)
        gpu = str(device or "").startswith("cuda")
        if not loaded:
            return DeciderReadiness(True, False, gpu, False, device, f"{name} is running but the model is not loaded yet")
        try:
            resp = client.post(f"{url.rstrip('/')}/v1/systemone", json=PROBE)
            resp.raise_for_status()
            answer = resp.json()["answers"]["go"]
            probs = answer["probabilities"]
            valid = answer["choice"] in ("go", "stop") and set(probs) == {"go", "stop"} and abs(sum(probs.values()) - 1) < 1e-3
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            return DeciderReadiness(True, True, gpu, False, device, f"{name} inference probe failed: {e}")
        if not valid:
            return DeciderReadiness(True, True, gpu, False, device, f"{name} probe returned an invalid answer: {answer}")
        if not gpu:
            return DeciderReadiness(True, True, False, True, device, f"{name} answers but runs on {device}, not a GPU")
        return DeciderReadiness(True, True, True, True, device, f"{name} ready on {device}")
    finally:
        if http is None:
            client.close()
