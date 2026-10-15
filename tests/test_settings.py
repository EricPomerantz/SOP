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
    with pytest.raises(ConfigError, match=r"SOP_DATA_ROOT points to .*nope, which doesn't exist. Set SOP_DATASET_DIR"):
        Settings().check_data_root()


def test_rtsp_urls_are_comma_separated(monkeypatch):
    monkeypatch.setenv("SOPVISION_RTSP_URLS", "rtsp://a/1, rtsp://b/2")
    assert Settings().sopvision_rtsp_urls == ["rtsp://a/1", "rtsp://b/2"]


def test_gateway_settings_read_sdk_env_names(monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://ai-gateway.vercel.sh/typesafe")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "typesafe-ai/jev")
    s = Settings()
    assert (s.typesafe_base_url, s.typesafe_model) == ("https://ai-gateway.vercel.sh/typesafe", "typesafe-ai/jev")


def test_data_root_that_is_not_a_dataset_is_named(monkeypatch, tmp_path):
    monkeypatch.setenv("SOP_DATA_ROOT", str(tmp_path))  # exists, but no raw/ and test/
    with pytest.raises(ConfigError, match=r"isn't a recordings dataset \(needs raw/ and test/ folders\)\. Set SOP_DATASET_DIR"):
        Settings().check_data_root()


def test_defaults_match_compose_network_and_mounts(monkeypatch):
    for variable in ("SOP_DATA_ROOT", "DATASETS_DIR", "DECIDER_URL", "LAYA_URL"):
        monkeypatch.delenv(variable, raising=False)
    settings = Settings(_env_file=None)
    assert settings.decider_url == "http://decider:8001"
    assert settings.laya_url == "http://laya:8000"
    assert str(settings.sop_data_root) == "/data/server_fan"
    assert str(settings.datasets_dir) == "/datasets"


def test_valid_dataset_layout_passes(monkeypatch, tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "test").mkdir()
    monkeypatch.setenv("SOP_DATA_ROOT", str(tmp_path))
    Settings().check_data_root()
