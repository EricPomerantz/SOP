import json
from pathlib import Path

import cv2
import numpy as np

from sopvision.dataset import ClipInfo, RunIndex
from training.sample_frames import extract, plan_samples

SEQ = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def fake_run(durations=(10.0, 4.0), split="train", run=4, paths=None):
    clips = [ClipInfo(Path(paths[i]) if paths else Path(f"c{i}.mp4"), SEQ[i], run, 1, i + 1, d) for i, d in enumerate(durations)]
    return RunIndex(run, split, clips, Path("raw.MP4"), sum(durations))


def test_sample_times_and_boundaries():
    got = sorted((s.position, round(s.t_in_clip, 2), s.tag) for s in plan_samples([fake_run()]))
    assert got == [(1, 1.0, "p10"), (1, 3.5, "p35"), (1, 6.5, "p65"), (1, 9.0, "p90"), (1, 9.5, "bm"),
                   (2, 0.4, "p10"), (2, 0.5, "bp"), (2, 1.4, "p35"), (2, 2.6, "p65"), (2, 3.6, "p90")]


def test_test_runs_never_sampled():
    assert plan_samples([fake_run(split="test")]) == []


def test_extract_writes_jpgs_and_manifest(tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    run = fake_run(durations=(2.0,), split="val", run=11, paths=[video])
    manifest = extract(plan_samples([run]), tmp_path / "frames")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4 and rows[0]["split"] == "val" and rows[0]["actions_by_position"] == [10]
    assert rows[0]["image"] == "val/Install_11_01_p10.jpg"
    assert (tmp_path / "frames" / rows[0]["image"]).is_file()
