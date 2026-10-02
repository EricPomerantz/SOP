import json

import yaml

from training.labelstudio import export_tasks, import_export, label_config, ls_to_yolo, yolo_to_ls

CLASSES = ["chassis", "fan", "fan_bay_empty", "psu", "psu_bay_empty", "cover", "hand"]
LINE_FAN = "1 0.500000 0.400000 0.100000 0.200000"
LINE_HAND = "6 0.200000 0.700000 0.100000 0.100000"


def make_frames(tmp_path, images=("train/Install_4_01_p10.jpg", "train/Install_4_02_p10.jpg", "val/Install_11_01_p10.jpg"),
                flagged=("train/Install_4_02_p10.jpg",)):
    frames = tmp_path / "data" / "frames"
    labels = tmp_path / "data" / "labels_auto"
    rows, flags = [], []
    for image in images:
        (frames / image).parent.mkdir(parents=True, exist_ok=True)
        (frames / image).write_bytes(b"jpg")
        (labels / image).with_suffix(".txt").parent.mkdir(parents=True, exist_ok=True)
        (labels / image).with_suffix(".txt").write_text(LINE_FAN + "\n", encoding="utf-8")
        split = image.split("/")[0]
        rows.append({"image": image, "run": 11 if split == "val" else 4, "split": split, "position": 1, "action": 10,
                     "actions_by_position": [10]})
        flags.append({"image": image, "ok": image not in flagged, "reasons": [] if image not in flagged else ["fan: 4 not in 2–3"]})
    manifest = frames / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    flags_path = labels / "flags.jsonl"
    flags_path.write_text("".join(json.dumps(f) + "\n" for f in flags), encoding="utf-8")
    return manifest, labels, flags_path


def task(key, annotations=(), prediction=(LINE_FAN,)):
    return {"id": 1, "data": {"image": f"/data/local-files/?d=frames/{key}", "key": key},
            "annotations": list(annotations),
            "predictions": [{"model_version": "auto", "result": [yolo_to_ls(line, CLASSES) for line in prediction]}]}


def annotation(lines, cancelled=False, updated="2026-10-02T10:00:00Z"):
    return {"id": 9, "was_cancelled": cancelled, "updated_at": updated, "result": [yolo_to_ls(line, CLASSES) for line in lines]}


def test_percent_round_trip():
    assert ls_to_yolo(yolo_to_ls(LINE_FAN, CLASSES), CLASSES) == LINE_FAN


def test_ls_result_shape():
    r = yolo_to_ls(LINE_FAN, CLASSES)
    assert r["type"] == "rectanglelabels" and r["from_name"] == "label" and r["to_name"] == "image"
    assert r["value"]["rectanglelabels"] == ["fan"] and abs(r["value"]["x"] - 45.0) < 1e-9 and abs(r["value"]["height"] - 20.0) < 1e-9


def test_label_config_lists_every_class():
    xml = label_config(CLASSES)
    assert '<Image name="image" value="$image"/>' in xml and all(f'<Label value="{c}"/>' in xml for c in CLASSES)


def test_flagged_tasks_first(tmp_path):
    manifest, labels, flags = make_frames(tmp_path)
    out = tmp_path / "tasks.json"
    assert export_tasks(manifest, labels, flags, out, doc_root=tmp_path / "data", model_version="gdino") == 3
    tasks = json.loads(out.read_text(encoding="utf-8"))
    assert [t["data"]["key"] for t in tasks] == ["train/Install_4_02_p10.jpg", "train/Install_4_01_p10.jpg", "val/Install_11_01_p10.jpg"]
    assert tasks[0]["data"]["image"] == "/data/local-files/?d=frames/train/Install_4_02_p10.jpg"
    assert tasks[0]["data"]["flagged"] is True and tasks[0]["predictions"][0]["model_version"] == "gdino"


def test_export_without_predictions_and_limit(tmp_path):
    manifest, labels, flags = make_frames(tmp_path)
    out = tmp_path / "tasks.json"
    assert export_tasks(manifest, labels, flags, out, doc_root=tmp_path / "data", model_version="x", limit=2, with_predictions=False) == 2
    assert all("predictions" not in t for t in json.loads(out.read_text(encoding="utf-8")))


def test_annotation_beats_prediction(tmp_path):
    manifest, _, _ = make_frames(tmp_path)
    export = tmp_path / "export.json"
    export.write_text(json.dumps([task("train/Install_4_01_p10.jpg", [annotation([LINE_HAND])])]), encoding="utf-8")
    stats = import_export(export, manifest, CLASSES, out_dir=tmp_path / "yolo")
    assert stats.annotated == 1 and stats.prediction_only == 0
    assert (tmp_path / "yolo/labels/train/Install_4_01_p10.txt").read_text(encoding="utf-8") == LINE_HAND + "\n"


def test_latest_non_cancelled_annotation_wins(tmp_path):
    manifest, _, _ = make_frames(tmp_path)
    export = tmp_path / "export.json"
    anns = [annotation([LINE_HAND], updated="2026-10-02T11:00:00Z"), annotation([LINE_FAN], updated="2026-10-02T10:00:00Z"),
            annotation([], cancelled=True, updated="2026-10-02T12:00:00Z")]
    export.write_text(json.dumps([task("train/Install_4_01_p10.jpg", anns)]), encoding="utf-8")
    import_export(export, manifest, CLASSES, out_dir=tmp_path / "yolo")
    assert (tmp_path / "yolo/labels/train/Install_4_01_p10.txt").read_text(encoding="utf-8") == LINE_HAND + "\n"


def test_cancelled_annotation_ignored(tmp_path):
    manifest, _, _ = make_frames(tmp_path)
    export = tmp_path / "export.json"
    export.write_text(json.dumps([task("train/Install_4_01_p10.jpg", [annotation([LINE_HAND], cancelled=True)])]), encoding="utf-8")
    stats = import_export(export, manifest, CLASSES, out_dir=tmp_path / "yolo")
    assert stats.prediction_only == 1
    assert (tmp_path / "yolo/labels/train/Install_4_01_p10.txt").read_text(encoding="utf-8") == LINE_FAN + "\n"


def test_dataset_yaml_names_and_split(tmp_path):
    manifest, _, _ = make_frames(tmp_path)
    export = tmp_path / "export.json"
    export.write_text(json.dumps([task("val/Install_11_01_p10.jpg")]), encoding="utf-8")
    import_export(export, manifest, CLASSES, out_dir=tmp_path / "yolo")
    data = yaml.safe_load((tmp_path / "yolo/dataset.yaml").read_text(encoding="utf-8"))
    assert list(data["names"].values()) == CLASSES and data["train"] == "images/train" and data["val"] == "images/val"
    assert (tmp_path / "yolo/images/val/Install_11_01_p10.jpg").is_file()
