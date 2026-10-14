import asyncio
import shutil
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import SecretStr

from sopvision.config import ConfigError
from sopvision.doctor import GpuReport
from sopvision.pipeline import RunCore
from sopvision.server.app import create_app
from sopvision.server.runs import RunManager, startup_checks
from sopvision.settings import Settings
from sopvision.sop.types import StepUpdate

WEBCAM_IN_CONTAINER = "Webcam isn't available in the container — run natively or use an RTSP source."
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


@pytest.fixture
def client_in_container(tmp_path):
    with make_client(tmp_path, sopvision_in_container=True) as c:
        yield c


def start(c, source="test:Install_12"):
    return c.post("/runs", json={"sop": "server_fan_install", "source_id": source})


def test_sops_list_steps_with_labels(client):
    sop = next(s for s in client.get("/sops").json() if s["id"] == "server_fan_install")
    assert sop["steps"][0] == {
        "id": "fan_1", "label": "Fan 1", "text": "Install the first fan by connecting the connector, then pressing the fan in place."}


def test_sources_hide_webcam_in_container(client_in_container):
    sources = client_in_container.get("/sources").json()
    assert sources and all(s["kind"] != "webcam" for s in sources)


def test_webcam_rejected_in_container(client_in_container):
    r = start(client_in_container, "webcam:0")
    assert r.status_code == 400 and r.json()["detail"] == WEBCAM_IN_CONTAINER


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
