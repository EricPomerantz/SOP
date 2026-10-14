import csv
import json
import time

import numpy as np
import pytest
import yaml

from sopvision.config import ConfigError
from sopvision.server.frames import FrameBuffer
from sopvision.server.recording import RecordingSession, add_to_split
from sopvision.settings import Settings
from sopvision.sources.base import Frame
from sopvision.sources.recorder import Recorder
from sopvision.sources.registry import SourceSpec


class FakeWriter:
    def __init__(self):
        self.frames = []

    def write(self, image):
        self.frames.append(image)

    def release(self):
        pass


def test_recorder_writes_on_a_real_time_clock_repeating_frames(tmp_path):
    writer = FakeWriter()
    r = Recorder(tmp_path / "v.mp4", fps=50, delay_s=0.4, open_writer=lambda p, size: writer)
    r.start_thread()
    r.offer(np.zeros((4, 6, 3), np.uint8))  # one frame only: the clock keeps writing it
    time.sleep(0.5)
    press = r.press(True)
    summary = r.stop()
    assert 20 <= summary["frames"] <= 27 and summary["received"] == 1  # ~0.5 s at 50 fps, not 1 frame
    assert len(writer.frames) == summary["frames"]
    assert 0.85 <= press.time_s <= 0.95  # 0.5 s into the video + the camera's 0.4 s delay


def test_press_before_any_frame_is_refused(tmp_path):
    r = Recorder(tmp_path / "v.mp4", open_writer=lambda p, size: FakeWriter())
    assert r.press(True) is None


def test_torque_csv_is_in_the_pipeline_format(tmp_path):
    from sopvision.sop.torque import read_torque_log

    r = Recorder(tmp_path / "v.mp4", open_writer=lambda p, size: FakeWriter(), clock=lambda: 10.0)
    r.start = 4.0
    r.press(True)
    r.press(False, 62.0)
    r.write_torque_csv(tmp_path / "torque.csv")
    assert [(t.t, t.ok, t.torque_nm) for t in read_torque_log(tmp_path / "torque.csv")] == [(6.0, True, None), (6.0, False, 62.0)]


def test_add_to_split_keeps_order_and_creates_the_file(tmp_path):
    f = tmp_path / "splits.yaml"
    add_to_split(f, "train", "r01")
    add_to_split(f, "test", "r02")
    add_to_split(f, "train", "r01")
    assert yaml.safe_load(f.read_text()) == {"train": ["r01"], "val": [], "test": ["r02"]}


class FakeCamera:
    label = "cam"

    def __init__(self):
        self.closed = False

    def frames(self):
        i = 0
        while not self.closed:
            time.sleep(0.02)
            yield Frame(i * 0.02, i, np.zeros((36, 64, 3), np.uint8))
            i += 1

    def close(self):
        self.closed = True


def session(lug_sop, tmp_path, name="r01", **kw):
    settings = Settings(datasets_dir=tmp_path, runs_dir=tmp_path / "runs-app")
    spec = SourceSpec(id="rtsp:0", kind="rtsp", label="cam")
    return RecordingSession(settings, lug_sop, "sops/lug_bolts.yaml", spec, kw.pop("split", "train"), name,
                            FrameBuffer(), fps=25, source=FakeCamera(), **kw)


def test_recording_session_writes_a_pipeline_ready_run(lug_sop, tmp_path):
    from sopvision.recordings import load_recordings

    s = session(lug_sop, tmp_path)
    s.start()
    time.sleep(0.4)
    assert s.torque(True) is not None
    meta = s.stop()
    folder = tmp_path / "lug_bolts" / "runs" / "r01"
    assert (folder / "video.mp4").stat().st_size > 0 and meta["frames"] >= 5
    rows = list(csv.DictReader((folder / "torque.csv").read_text().splitlines()))
    assert len(rows) == 1 and rows[0]["ok"] == "1"
    assert json.loads((folder / "meta.json").read_text())["split"] == "train"
    (rec,) = load_recordings(lug_sop, Settings(datasets_dir=tmp_path))
    assert (rec.id, rec.split, rec.valid, rec.torque_log.name) == ("r01", "train", False, "torque.csv")  # steps.csv next


def test_recording_refuses_to_overwrite_or_bad_names(lug_sop, tmp_path):
    (tmp_path / "lug_bolts" / "runs" / "r01").mkdir(parents=True)
    with pytest.raises(ConfigError, match="already exists"):
        session(lug_sop, tmp_path)
    with pytest.raises(ConfigError, match="letters, digits"):
        session(lug_sop, tmp_path, name="../escape")
    with pytest.raises(ConfigError, match="split"):
        session(lug_sop, tmp_path, name="r02", split="holdout")


def test_recording_refuses_clip_format_sops(sop, tmp_path):
    with pytest.raises(ConfigError, match="timeline format only"):
        session(sop, tmp_path)


def test_discard_deletes_the_run_and_leaves_the_splits_alone(lug_sop, tmp_path):
    keep = session(lug_sop, tmp_path, name="r01")
    keep.start()
    time.sleep(0.2)
    keep.stop()
    splits = (tmp_path / "lug_bolts" / "splits.yaml").read_text()
    s = session(lug_sop, tmp_path, name="r02-oops")
    s.start()
    time.sleep(0.3)
    assert (tmp_path / "lug_bolts" / "runs" / "r02-oops").exists()  # the video is being written
    assert s.discard()["discarded"] is True
    assert not (tmp_path / "lug_bolts" / "runs" / "r02-oops").exists()
    assert (tmp_path / "lug_bolts" / "runs" / "r01" / "video.mp4").exists()
    assert (tmp_path / "lug_bolts" / "splits.yaml").read_text() == splits
