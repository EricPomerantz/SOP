"""Label Studio round trip for reviewing auto-labels (spec §6.1 step 4).

Usage:
  uv run python -m training.labelstudio config                         # labelling interface XML
  uv run python -m training.labelstudio export --out data/labelstudio/tasks.json
  uv run python -m training.labelstudio import --export data/labelstudio/export.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sopvision.config import load_sop
from training.yolo_dataset import write_dataset_yaml, write_item

LOCAL_FILES_PREFIX = "/data/local-files/?d="


def label_config(classes: list[str]) -> str:
    labels = "\n".join(f'    <Label value="{c}"/>' for c in classes)
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def yolo_to_ls(line: str, classes: list[str]) -> dict:
    idx, cx, cy, w, h = line.split()
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {"x": (cx - w / 2) * 100, "y": (cy - h / 2) * 100, "width": w * 100, "height": h * 100,
                  "rotation": 0, "rectanglelabels": [classes[int(idx)]]},
    }


def ls_to_yolo(result: dict, classes: list[str]) -> str:
    v = result["value"]
    w, h = v["width"] / 100, v["height"] / 100
    cx, cy = v["x"] / 100 + w / 2, v["y"] / 100 + h / 2
    return f"{classes.index(v['rectanglelabels'][0])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _label_lines(labels_dir: Path, image: str) -> list[str]:
    path = Path(labels_dir) / Path(image).with_suffix(".txt")
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def export_tasks(manifest: Path, labels_dir: Path, flags: Path, out_json: Path, doc_root: Path, model_version: str,
                 limit: int | None = None, with_predictions: bool = True, classes: list[str] | None = None) -> int:
    from training.autolabel import pick_spread

    manifest = Path(manifest)
    frames_dir = manifest.parent
    rows = _read_jsonl(manifest)
    if limit:
        rows = sorted(pick_spread(rows, limit), key=rows.index)
    flag_by_image = {f["image"]: f for f in _read_jsonl(flags)} if flags and Path(flags).exists() else {}
    classes = classes or load_sop(Path("sops/server_fan.yaml")).detector.classes

    def is_flagged(row):
        return not flag_by_image.get(row["image"], {"ok": True})["ok"]

    tasks = []
    for row in sorted(rows, key=lambda r: not is_flagged(r)):  # stable: flagged first, then manifest order
        rel = (frames_dir / row["image"]).resolve().relative_to(Path(doc_root).resolve()).as_posix()
        task = {"data": {"image": LOCAL_FILES_PREFIX + rel, "key": row["image"], "flagged": is_flagged(row),
                         "reasons": "; ".join(flag_by_image.get(row["image"], {}).get("reasons", []))}}
        if with_predictions:
            task["predictions"] = [{"model_version": model_version,
                                    "result": [yolo_to_ls(l, classes) for l in _label_lines(labels_dir, row["image"])]}]
        tasks.append(task)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    return len(tasks)


@dataclass(frozen=True)
class ImportStats:
    annotated: int
    prediction_only: int


def _chosen_result(task: dict) -> tuple[list[dict] | None, bool]:
    """(result, from_annotation): latest non-cancelled annotation, else the first prediction with results."""
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if anns:
        latest = max(anns, key=lambda a: (a.get("updated_at") or "", a.get("id") or 0))
        return latest.get("result", []), True
    for p in task.get("predictions", []):
        if isinstance(p, dict) and "result" in p:
            return p["result"], False
    return None, False


def import_export(export_json: Path, manifest: Path, classes: list[str], out_dir: Path = Path("data/yolo"),
                  labels_dir: Path | None = None) -> ImportStats:
    manifest = Path(manifest)
    frames_dir = manifest.parent
    row_by_image = {r["image"]: r for r in _read_jsonl(manifest)}
    out_dir = Path(out_dir)
    annotated = prediction_only = 0
    for task in json.loads(Path(export_json).read_text(encoding="utf-8")):
        key = task["data"]["key"]
        split = row_by_image[key]["split"]
        result, from_annotation = _chosen_result(task)
        if result is None:  # export without prediction bodies: fall back to the auto-label file
            lines = _label_lines(labels_dir, key) if labels_dir else []
        else:
            lines = [ls_to_yolo(r, classes) for r in result if r.get("type") == "rectanglelabels"]
        annotated += from_annotation
        prediction_only += not from_annotation
        write_item(out_dir, split, frames_dir / key, lines)
    write_dataset_yaml(out_dir, classes)
    return ImportStats(annotated, prediction_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sop", type=Path, default=Path("sops/server_fan.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config")
    exp = sub.add_parser("export")
    exp.add_argument("--manifest", type=Path, default=Path("data/frames/manifest.jsonl"))
    exp.add_argument("--labels", type=Path, default=Path("data/labels_auto"))
    exp.add_argument("--doc-root", type=Path, default=Path("data"))
    exp.add_argument("--model-version", default="auto")
    exp.add_argument("--limit", type=int)
    exp.add_argument("--no-predictions", action="store_true")
    exp.add_argument("--out", type=Path, default=Path("data/labelstudio/tasks.json"))
    imp = sub.add_parser("import")
    imp.add_argument("--export", type=Path, required=True)
    imp.add_argument("--manifest", type=Path, default=Path("data/frames/manifest.jsonl"))
    imp.add_argument("--labels", type=Path, default=Path("data/labels_auto"))
    imp.add_argument("--out", type=Path, default=Path("data/yolo"))
    args = parser.parse_args()
    classes = load_sop(args.sop).detector.classes
    if args.cmd == "config":
        print(label_config(classes), end="")
    elif args.cmd == "export":
        n = export_tasks(args.manifest, args.labels, args.labels / "flags.jsonl", args.out, args.doc_root,
                         args.model_version, args.limit, not args.no_predictions, classes)
        print(f"wrote {n} tasks to {args.out}")
    else:
        stats = import_export(args.export, args.manifest, classes, args.out, args.labels)
        print(f"imported {stats.annotated} reviewed + {stats.prediction_only} auto-labelled frames into {args.out}")


if __name__ == "__main__":
    main()
