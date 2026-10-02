import json

import yaml

from training.yolo_dataset import build_dataset

CLASSES = ["chassis", "fan", "fan_bay_empty", "psu", "psu_bay_empty", "cover", "hand"]


def test_build_dataset_copies_labelled_frames_and_skips_unlabelled(tmp_path):
    frames, labels = tmp_path / "frames", tmp_path / "labels"
    rows = [{"image": "train/Install_4_01_p10.jpg", "split": "train"},
            {"image": "train/Install_4_02_p10.jpg", "split": "train"},   # alignment failed: no label file
            {"image": "val/Install_11_01_p10.jpg", "split": "val"}]
    for r in rows:
        (frames / r["image"]).parent.mkdir(parents=True, exist_ok=True)
        (frames / r["image"]).write_bytes(b"jpg")
    for r in (rows[0], rows[2]):
        (labels / r["image"]).with_suffix(".txt").parent.mkdir(parents=True, exist_ok=True)
        (labels / r["image"]).with_suffix(".txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (frames / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    counts = build_dataset(frames / "manifest.jsonl", labels, CLASSES, tmp_path / "yolo")

    assert counts == {"train": 1, "val": 1, "skipped": 1}
    assert (tmp_path / "yolo/labels/train/Install_4_01_p10.txt").read_text(encoding="utf-8") == "1 0.5 0.5 0.1 0.1\n"
    assert not (tmp_path / "yolo/images/train/Install_4_02_p10.jpg").exists()
    data = yaml.safe_load((tmp_path / "yolo/dataset.yaml").read_text(encoding="utf-8"))
    assert list(data["names"].values()) == CLASSES and data["val"] == "images/val"
