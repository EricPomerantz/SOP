import pytest
from pydantic import ValidationError

from sopvision.settings import Settings
from sopvision.sources.file import FileSource
from sopvision.sources.registry import SourceSpec, list_sources, open_source


def test_only_file_and_rtsp_sources_are_supported():
    assert all(source.kind in {"file", "rtsp"} for source in list_sources(Settings()))
    with pytest.raises(ValidationError, match="kind"):
        SourceSpec.model_validate({"id": "webcam:0", "kind": "webcam", "label": "Webcam"})


def test_lists_scenarios_and_rtsp(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "fan_removed.mp4").touch()
    s = Settings(scenarios_dir=scenarios, sop_data_root=tmp_path / "none", sopvision_rtsp_urls=["rtsp://cam/1"])
    by_id = {spec.id: spec for spec in list_sources(s)}
    assert {"scenario:fan_removed", "rtsp:0"} <= set(by_id)
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


def test_lists_timeline_test_recordings_with_their_torque_logs(tmp_path):
    root = tmp_path / "datasets" / "lug_bolts"
    for run in ("correct1", "dev2", "train1"):
        (root / "runs" / run).mkdir(parents=True)
        (root / "runs" / run / "video.mp4").touch()
    (root / "runs" / "dev2" / "torque.csv").write_text("time_s,ok,torque_nm\n3.0,1,\n")
    (root / "splits.yaml").write_text("train: [train1]\nval: []\ntest: [correct1, dev2, missing]\n")
    s = Settings(datasets_dir=tmp_path / "datasets", sop_data_root=tmp_path / "none", scenarios_dir=tmp_path / "none")
    by_id = {spec.id: spec for spec in list_sources(s)}
    assert {i for i in by_id if i.startswith("recording:")} == {"recording:lug_bolts:correct1", "recording:lug_bolts:dev2"}
    assert by_id["recording:lug_bolts:dev2"].torque_log == root / "runs" / "dev2" / "torque.csv"
    assert by_id["recording:lug_bolts:correct1"].torque_log is None
    assert by_id["recording:lug_bolts:dev2"].label == "lug_bolts: dev2 (test recording)"
