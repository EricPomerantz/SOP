import pytest

from sopvision.settings import Settings
from sopvision.sources.file import FileSource
from sopvision.sources.registry import SourceSpec, SourceUnavailableError, list_sources, open_source

WEBCAM_IN_CONTAINER = "Webcam isn't available in the container — run natively or use an RTSP source."


def test_webcam_not_listed_or_openable_in_container(settings_in_container):
    assert all(s.kind != "webcam" for s in list_sources(settings_in_container))
    with pytest.raises(SourceUnavailableError, match=WEBCAM_IN_CONTAINER):
        open_source(SourceSpec(id="webcam:0", kind="webcam", label="Webcam", device=0), settings_in_container)


def test_lists_scenarios_and_rtsp(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "fan_removed.mp4").touch()
    s = Settings(scenarios_dir=scenarios, sop_data_root=tmp_path / "none", sopvision_rtsp_urls=["rtsp://cam/1"],
                 sopvision_in_container=False)
    by_id = {spec.id: spec for spec in list_sources(s)}
    assert {"scenario:fan_removed", "rtsp:0", "webcam:0"} <= set(by_id)
    assert by_id["scenario:fan_removed"].label == "Simulated: fan removed"
    assert by_id["rtsp:0"].url == "rtsp://cam/1"


def test_lists_test_runs_from_raw_videos(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for run in (12, 13):
        (raw / f"Install_{run}.MP4").touch()
    s = Settings(sop_data_root=tmp_path / "data", scenarios_dir=tmp_path / "none")
    by_id = {spec.id: spec for spec in list_sources(s)}
    assert by_id["test:Install_12"].path == raw / "Install_12.MP4" and by_id["test:Install_13"].kind == "file"


def test_open_file_source_uses_mode(tiny_video):
    src = open_source(SourceSpec(id="scenario:x", kind="file", label="x", path=tiny_video), Settings(), mode="lockstep")
    assert isinstance(src, FileSource) and src.mode == "lockstep"
