import asyncio
import shutil
from pathlib import Path

import pytest
import yaml
import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from sopvision.config import ConfigError
from sopvision.doctor import GpuReport
from sopvision.pipeline import RunCore
from sopvision.server.app import create_app
from sopvision.server.runs import RunManager, startup_checks
from sopvision.settings import Settings
from sopvision.sop.types import StepUpdate

GPU = GpuReport(True, "NVIDIA GeForce RTX 4080", "x")


class FakeRun:
    """Behaves like LiveRun: logs through a real RunCore and emits a scripted step update on start."""

    def __init__(self, ctx, spec):
        self.ctx, self.spec = ctx, spec
        self.core = RunCore(ctx.sop, ctx.run_id, spec.id, ctx.runlog, ctx.broadcast)
        self.done = asyncio.Event()
        self.paused = False

    async def start(self):
        self.statuses = {st.id: "pending" for st in self.ctx.sop.steps} | {"fan_1": "active"}
        self.core.emit([StepUpdate(1.0, "fan_1", "active", None)])

    async def stop(self):
        if not self.done.is_set():
            self.report = self.core.finish()
            self.done.set()
        return self.report

    def snapshot(self):
        return {"steps": [{"id": k, "status": v} for k, v in self.statuses.items()], "alerts": []}

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_speed(self, speed):
        pass


def make_client(tmp_path, **overrides):
    data_root = tmp_path / "data"
    (data_root / "raw").mkdir(parents=True)
    (data_root / "raw" / "Install_12.MP4").touch()
    settings = Settings(sop_data_root=data_root, runs_dir=tmp_path / "runs", scenarios_dir=tmp_path / "scenarios",
                        **overrides)
    from sopvision.config import list_sops

    manager = RunManager(settings, {s.sop: s for s in list_sops(Path("sops"))}, run_factory=FakeRun)
    return TestClient(create_app(settings, manager))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path, sopvision_rtsp_urls=["rtsp://cam/1"]) as c:
        yield c


def start(c, source="test:Install_12"):
    return c.post("/runs", json={"sop": "server_fan_install", "source_id": source})


def test_sops_list_steps_with_labels(client):
    sop = next(s for s in client.get("/sops").json() if s["id"] == "server_fan_install")
    assert sop["steps"][0] == {
        "id": "fan_1", "label": "Fan 1", "text": "Install the first fan by connecting the connector, then pressing the fan in place."}


def test_sources_exclude_webcams(client):
    sources = client.get("/sources").json()
    assert sources and all(s["kind"] != "webcam" for s in sources)


def test_legacy_webcam_request_has_migration_hint(client):
    response = start(client, "webcam:0")
    assert response.status_code == 400
    assert response.json()["detail"] == "USB webcams are no longer supported. Use an RTSP source or a video file."


def test_docker_smoke_accepts_multiple_sops_and_keeps_active_run(client):
    from scripts.docker_smoke import check_server

    run_id = start(client).json()["run_id"]
    sops, sources = check_server(client)
    assert sops == 2 and sources >= 2
    assert client.get("/runs/current").json()["run_id"] == run_id
    assert client.post(f"/runs/{run_id}/stop").status_code == 200


def test_docker_smoke_checks_http_failures():
    from scripts.docker_smoke import check_server

    with httpx.Client(base_url="http://sopvision:8000", transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"detail": "not ready"}))) as client:
        with pytest.raises(httpx.HTTPStatusError):
            check_server(client)


@pytest.mark.parametrize("arguments, expected", [([], "http://sopvision:8000"),
                                                (["--url", "http://gateway:9000"], "http://gateway:9000")])
def test_torque_simulator_uses_container_url_by_default(monkeypatch, arguments, expected):
    import sys

    from scripts import simulate_torque

    urls = []
    monkeypatch.setattr(sys, "argv", ["simulate_torque.py", "--every", "1", "--count", "0", *arguments])
    monkeypatch.setattr(simulate_torque, "current_run", lambda url: urls.append(url) or "test-run")
    assert simulate_torque.main() == 0
    assert urls == [expected]


def test_unknown_source_is_404(client):
    assert start(client, "test:Install_99").status_code == 404


def test_one_run_at_a_time(client):
    assert start(client).status_code == 201
    r = start(client)
    assert r.status_code == 409 and r.json()["detail"] == "A run is already in progress. Stop it first."


def test_ws_streams_step_updates(client):
    with client.websocket_connect("/ws") as ws:
        run_id = start(client).json()["run_id"]
        msg = ws.receive_json()
        assert (msg["type"], msg["step_id"], msg["status"]) == ("step_update", "fan_1", "active")
        client.post(f"/runs/{run_id}/stop")
        kinds = [ws.receive_json()["type"] for _ in range(2)]
        assert "run_summary" in kinds


def test_current_snapshot_and_report_csv(client):
    assert client.get("/runs/current").status_code == 204
    run_id = start(client).json()["run_id"]
    snap = client.get("/runs/current").json()
    assert snap["run_id"] == run_id and snap["source"] == "test:Install_12"
    assert next(s for s in snap["steps"] if s["id"] == "fan_1")["status"] == "active"
    assert client.post(f"/runs/{run_id}/stop").status_code == 200
    r = client.get(f"/runs/{run_id}/report?format=csv")
    assert r.headers["content-type"].startswith("text/csv")
    assert r.text.splitlines()[0] == "row_type,id,t,status,duration_s,message"
    assert client.get(f"/runs/{run_id}/report").json()["run_id"] == run_id


def test_pause_works_for_files_and_is_rejected_for_rtsp(client):
    run_id = start(client).json()["run_id"]
    assert client.post(f"/runs/{run_id}/control", json={"action": "pause"}).status_code == 200
    client.post(f"/runs/{run_id}/stop")
    rtsp_id = start(client, "rtsp:0").json()["run_id"]
    r = client.post(f"/runs/{rtsp_id}/control", json={"action": "pause"})
    assert r.status_code == 400 and r.json()["detail"] == "Only recorded videos can be paused."


@pytest.fixture
def tmp_repo(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "test").mkdir()
    sops = tmp_path / "sops"
    sops.mkdir()
    weights = tmp_path / "models" / "server_fan" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    cfg = yaml.safe_load(Path("sops/server_fan.yaml").read_text(encoding="utf-8"))
    cfg["detector"]["weights"] = str(weights)
    (sops / "server_fan.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    settings = Settings(sop_data_root=tmp_path / "data", sops_dir=sops, typesafe_api_key=SecretStr("k"))
    return settings, weights


def test_startup_checks_pass_and_return_sops(tmp_repo):
    settings, _ = tmp_repo
    assert [s.sop for s in startup_checks(settings, GPU)] == ["server_fan_install"]


def test_startup_checks_name_the_missing_api_key(tmp_repo):
    settings, _ = tmp_repo
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY is not set. Add it to .env."):
        startup_checks(settings.model_copy(update={"typesafe_api_key": SecretStr("")}), GPU)


def test_startup_checks_name_missing_weights(tmp_repo):
    settings, weights = tmp_repo
    weights.unlink()
    with pytest.raises(ConfigError, match="Weights not found: .*best.pt"):
        startup_checks(settings, GPU)


def test_back_to_back_runs_get_their_own_ids_and_logs(client):
    first = start(client).json()["run_id"]
    client.post(f"/runs/{first}/stop")
    second = start(client).json()["run_id"]
    client.post(f"/runs/{second}/stop")
    assert first != second
    for run_id in (first, second):
        assert client.get(f"/runs/{run_id}/report").json()["run_id"] == run_id


def test_sops_carry_a_display_name(client):
    names = {s["id"]: s["name"] for s in client.get("/sops").json()}
    assert names["server_fan_install"] == "Server fan install" and names["lug_bolts_torque"] == "Wheel lug bolts"


def test_run_info_names_the_deciding_model(tmp_path):
    with make_client(tmp_path, decision_mode="laya") as c:
        info = start(c).json()
        assert info["decision_model"] == "laya"
        assert c.get("/runs/current").json()["decision_model"] == "laya"
    with make_client(tmp_path / "b", decision_mode="compare") as c:
        assert start(c).json()["decision_model"] == "jev"  # compare: Jev drives the tracker


def test_config_names_the_deciding_model_before_any_run(tmp_path):
    with make_client(tmp_path, decision_mode="laya") as c:
        assert c.get("/config").json() == {"decision_model": "laya", "decision_mode": "laya"}


def test_torque_results_need_a_run_and_a_sop_with_a_torque_tool(client):
    assert client.post("/runs/nope/torque", json={"ok": True}).status_code == 404
    run_id = start(client).json()["run_id"]
    r = client.post(f"/runs/{run_id}/torque", json={"ok": True, "torque_nm": 110})
    assert r.status_code == 409 and "no torque tool" in r.json()["detail"]


def test_latency_clock_and_measure_guards(client):
    assert client.get("/latency/clock").json()["now_ms"] > 1.7e12
    assert client.post("/latency/measure", json={"source_id": "test:Install_12"}).status_code == 400  # a video
    assert client.post("/latency/measure", json={"source_id": "rtsp:9"}).status_code == 404
    start(client)
    assert client.post("/latency/measure", json={"source_id": "rtsp:0"}).status_code == 409  # a run is going


def test_recording_lists_timeline_sops_and_guards_its_inputs(client):
    sops = client.get("/recording/sops").json()
    assert [s["id"] for s in sops] == ["lug_bolts_torque"] and sops[0]["torque"] is True
    assert client.get("/recording").json() is None
    body = {"sop_file": sops[0]["file"], "source_id": "test:Install_12"}
    assert client.post("/recording/start", json=body).status_code == 400  # a video file, not a camera
    assert client.post("/recording/start", json=body | {"sop_file": "sops/server_fan.yaml"}).status_code == 404
    assert client.post("/recording/torque", json={"ok": True}).status_code == 404  # nothing recording
    assert client.post("/recording/stop").status_code == 404
    assert client.post("/recording/discard").status_code == 404


def test_broken_sop_files_are_reported_not_hidden(tmp_path):
    drafts = tmp_path / "sops" / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "broken.yaml").write_text(
        Path("sops/lug_bolts.yaml").read_text(encoding="utf-8").replace("anchor_follow: 0.7", "anchor_follw: 0.7"),
        encoding="utf-8")
    with make_client(tmp_path, sops_dir=tmp_path / "sops") as c:
        assert c.get("/recording/sops").json() == []
        (err,) = c.get("/recording/sop-errors").json()
        assert err["file"].endswith("broken.yaml") and "anchor_follw" in err["error"]


def test_jobs_are_queued_by_kind_and_can_be_cancelled(tmp_path):
    with make_client(tmp_path, jobs_dir=tmp_path / "jobs") as c:
        r = c.post("/jobs", json={"kind": "draft_timelines", "params": {"sop_file": "lug_bolts.yaml", "runs": ["correct1"]}})
        assert r.status_code == 201, r.text
        job = r.json()
        assert job["state"] == "queued" and job["steps"][0]["argv"][1:4] == ["-m", "sopvision.builder.timelines", "--sop"]
        assert job["steps"][0]["argv"][-2:] == ["--run", "correct1"]
        assert [j["id"] for j in c.get("/jobs").json()] == [job["id"]]
        assert c.post(f"/jobs/{job['id']}/cancel").json()["cancel_requested"] is True
        assert c.get(f"/jobs/{job['id']}").json()["log"] == ""


def test_jobs_refuse_unknown_kinds_and_sop_files_outside_the_sops_folder(tmp_path):
    with make_client(tmp_path, jobs_dir=tmp_path / "jobs") as c:
        assert c.post("/jobs", json={"kind": "shell", "params": {}}).status_code == 404
        for name in ("../.env", "/etc/passwd", "drafts/../../x.yaml", "missing.yaml"):
            assert c.post("/jobs", json={"kind": "draft_timelines", "params": {"sop_file": name}}).status_code == 400
        assert c.get("/jobs/../x").status_code == 404


def test_an_sop_built_in_the_ui_is_published_into_the_top_bar(tmp_path, monkeypatch):
    from sopvision.builder import detector as det

    monkeypatch.setattr(det, "MODELS", tmp_path / "models")
    monkeypatch.setattr(det, "_ultralytics_names",
                        lambda p: ("detect", ["hub", "hole_empty", "bolt", "wrench", "hand"]))
    (tmp_path / "sops").mkdir()
    with make_client(tmp_path, sops_dir=tmp_path / "sops", datasets_dir=tmp_path / "datasets",
                     laya_registry=tmp_path / "registry.json") as c:
        assert c.post("/builder/sops", json={"id": "wheel", "name": "Wheel"}).status_code == 201
        r = c.put("/builder/sops/wheel/detector", content=b"weights", headers={"Content-Type": "application/octet-stream"})
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["detector"]["classes"][0] == "hub" and r.json()["status"]["detector"] is True
        data["detector"]["roles"] = {"bolt": "part", "hole_empty": "empty"}
        data["layout"] = {"anchor": "hub", "match_radius": 0.15, "tools": ["wrench"],
                          "slots": {"hole_1": {"at": [0.6, 0.15], "filled_by": "bolt", "empty_as": "hole_empty"}}}
        assert c.put("/builder/sops/wheel", json={"data": data}).json()["status"]["roles"] is True
        data |= c.post("/builder/sops/wheel/generate-steps").json()
        saved = c.put("/builder/sops/wheel", json={"data": data}).json()
        assert saved["status"]["errors"] == [] and saved["status"]["loaded"] is False
        assert "wheel_torque" not in [s["id"] for s in c.get("/sops").json()]
        published = c.post("/builder/sops/wheel/publish")
        assert published.status_code == 200, published.text
        assert published.json()["draft"] is False and published.json()["status"]["loaded"] is True
        assert "wheel" in [s["id"] for s in c.get("/sops").json()]
        assert [e["id"] for e in c.get("/builder/sops").json()] == ["wheel"]
        assert c.put("/builder/sops/wheel", json={"data": data}).status_code == 409  # published: edit first
        assert c.post("/builder/sops/wheel/edit").json()["draft"] is True


def test_the_builder_refuses_non_detection_uploads(tmp_path, monkeypatch):
    from sopvision.builder import detector as det

    monkeypatch.setattr(det, "MODELS", tmp_path / "models")
    monkeypatch.setattr(det, "_ultralytics_names", lambda p: ("classify", ["a"]))
    (tmp_path / "sops").mkdir()
    with make_client(tmp_path, sops_dir=tmp_path / "sops") as c:
        c.post("/builder/sops", json={"id": "wheel"})
        r = c.put("/builder/sops/wheel/detector", content=b"x", headers={"Content-Type": "application/octet-stream"})
        assert r.status_code == 400 and "classify model" in r.json()["detail"]
        assert list((tmp_path / "models").glob("*")) == []


def test_autofit_jobs_take_the_builders_file_path_and_results_are_served(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    shutil.copytree(Path(__file__).parents[2] / "sops", tmp_path / "sops", ignore=shutil.ignore_patterns("drafts"))
    with make_client(tmp_path, sops_dir=Path("sops"), jobs_dir=tmp_path / "jobs") as c:
        job = c.post("/jobs", json={"kind": "autofit", "params": {"sop_file": "sops/lug_bolts.yaml", "threshold": 0.1}})
        assert job.status_code == 201, job.text
        assert job.json()["steps"][0]["argv"][2:] == ["sopvision.builder.autofit", "--sop", "sops/lug_bolts.yaml",
                                                      "--threshold", "0.1"]
        assert c.get("/builder/sops/lug_bolts_torque/autofit").json() is None
        out = tmp_path / "data/builder/lug_bolts_torque"
        (out / "frames").mkdir(parents=True)
        (out / "autofit.json").write_text(json.dumps({"slots": []}))
        (out / "frames" / "r1.jpg").write_bytes(b"\xff\xd8")
        assert c.get("/builder/sops/lug_bolts_torque/autofit").json() == {"slots": []}
        assert c.get("/builder/sops/lug_bolts_torque/frames/r1.jpg").content == b"\xff\xd8"
        assert c.get("/builder/sops/nope/autofit").status_code == 404
        assert c.post("/builder/sops/nope/generate-steps").status_code == 404


def test_a_runs_step_timeline_is_read_and_saved_only_when_contiguous(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from sopvision.builder import drafts

    (tmp_path / "sops").mkdir()
    settings_kw = dict(sops_dir=tmp_path / "sops", datasets_dir=tmp_path / "datasets")
    with make_client(tmp_path, **settings_kw) as c:
        c.post("/builder/sops", json={"id": "wheel"})
        sf = drafts.current(Settings(**settings_kw), "wheel")
        drafts.write(sf, dict(sf.data, steps=[{"id": "place_1", "text": "x", "label": "Bolt 1"}]))
        folder = tmp_path / "datasets/wheel/runs/r1"
        folder.mkdir(parents=True)
        writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(20):
            writer.write(np.zeros((48, 64, 3), np.uint8))
        writer.release()
        (folder / "torque.csv").write_text("time_s,ok,torque_nm\n1.5,true,\n")
        t = c.get("/builder/sops/wheel/runs/r1/timeline").json()
        assert (t["duration"], t["steps"], t["preview"], t["labels"]) == (2.0, None, False, {"place_1": "Bolt 1"})
        assert t["torque"] == [{"time_s": 1.5, "ok": True}]
        bad = c.put("/builder/sops/wheel/runs/r1/timeline", json={"rows": [["place_1", 0, 1], ["idle", 1.2, 2]]})
        assert bad.status_code == 400 and "row before" in bad.json()["detail"]
        ok = c.put("/builder/sops/wheel/runs/r1/timeline", json={"rows": [["place_1", 0, 1], ["idle", 1, 2]]})
        assert ok.status_code == 200 and ok.json()["steps"] == [["place_1", 0.0, 1.0], ["idle", 1.0, 2.0]]
        assert (folder / "steps.csv").read_text().splitlines()[0] == "step_id,start_s,end_s"
        assert c.get("/builder/sops/wheel/runs/..%2F..%2Fx/timeline").status_code == 404
        assert c.get("/builder/sops/wheel/runs/r1/preview.webm").status_code == 404


def test_training_is_queued_as_stages_and_activation_points_the_sop_at_a_checkpoint(tmp_path, monkeypatch):
    from sopvision.builder import training

    monkeypatch.setattr(training, "MODELS", tmp_path / "models")
    (tmp_path / "models" / "lug_bolts_torque-v1").mkdir(parents=True)
    (tmp_path / "models" / "lug_bolts_torque-v1" / "model.safetensors").write_bytes(b"")
    registry = tmp_path / "registry.json"
    with make_client(tmp_path, jobs_dir=tmp_path / "jobs", laya_registry=registry, laya_url="http://127.0.0.1:9") as c:
        job = c.post("/jobs", json={"kind": "train_laya", "params": {"sop_file": "lug_bolts.yaml", "epochs": 3}}).json()
        assert [s["name"] for s in job["steps"]] == list(training.STAGES)
        assert job["steps"][4]["argv"][-4:] == ["--version", "2", "--epochs", "3"]
        assert job["result"]["checkpoint"] == "lug_bolts_torque-v2"
        r = c.post("/builder/sops/lug_bolts_torque/activate", json={"checkpoint": "lug_bolts_torque-v1"}).json()
        assert r["active"] == "lug_bolts_torque-v1" and "didn't reload" in r["warning"]
        assert [v["checkpoint"] for v in r["versions"]] == ["lug_bolts_torque-v1"]
        assert c.post("/builder/sops/lug_bolts_torque/activate", json={"checkpoint": "nope"}).status_code == 404
        assert c.post("/builder/sops/lug_bolts_torque/activate", json={"checkpoint": None}).json()["active"] is None


def test_an_sop_being_edited_is_offered_for_recording_once_as_its_draft(tmp_path):
    (tmp_path / "sops" / "drafts").mkdir(parents=True)
    shutil.copy("sops/lug_bolts.yaml", tmp_path / "sops" / "lug_bolts.yaml")
    shutil.copy("sops/lug_bolts.yaml", tmp_path / "sops" / "drafts" / "lug_bolts.yaml")
    with make_client(tmp_path, sops_dir=tmp_path / "sops") as c:
        sops = c.get("/recording/sops").json()
        assert [(s["id"], Path(s["file"]).parent.name) for s in sops] == [("lug_bolts_torque", "drafts")]
