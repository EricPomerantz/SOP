import pytest

from sopvision.config import ConfigError
from sopvision.settings import Settings


def test_api_key_never_in_repr(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-secret")
    s = Settings()
    assert "sk-secret" not in repr(s) and "sk-secret" not in str(s)
    assert s.typesafe_api_key.get_secret_value() == "sk-secret"


def test_missing_data_root_names_the_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("SOP_DATA_ROOT", str(tmp_path / "nope"))
    with pytest.raises(ConfigError, match="SOP_DATA_ROOT points to .*nope, which doesn't exist. Set it in .env."):
        Settings().check_data_root()


def test_rtsp_urls_are_comma_separated(monkeypatch):
    monkeypatch.setenv("SOPVISION_RTSP_URLS", "rtsp://a/1, rtsp://b/2")
    assert Settings().sopvision_rtsp_urls == ["rtsp://a/1", "rtsp://b/2"]


def test_gateway_settings_read_sdk_env_names(monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://ai-gateway.vercel.sh/typesafe")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "typesafe-ai/jev")
    s = Settings()
    assert (s.typesafe_base_url, s.typesafe_model) == ("https://ai-gateway.vercel.sh/typesafe", "typesafe-ai/jev")
