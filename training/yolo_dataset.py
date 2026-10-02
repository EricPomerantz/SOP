"""Assemble the Ultralytics dataset folder (images/, labels/, dataset.yaml) from labelled frames.

Usage: uv run python -m training.yolo_dataset   (data/frames + data/labels_auto → data/yolo)
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from sopvision.config import load_sop


def write_item(out_dir: Path, split: str, image_src: Path, label_lines: list[str]) -> None:
    image_target = out_dir / "images" / split / image_src.name
    label_target = out_dir / "labels" / split / image_src.with_suffix(".txt").name
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_src, image_target)
    label_target.write_text("".join(l + "\n" for l in label_lines), encoding="utf-8")


def write_dataset_yaml(out_dir: Path, classes: list[str]) -> Path:
    path = out_dir / "dataset.yaml"
    data = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "names": dict(enumerate(classes))}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def build_dataset(manifest: Path, labels_dir: Path, classes: list[str], out_dir: Path) -> dict[str, int]:
    """Copy every frame that has a label file; frames without one (e.g. alignment failed) are skipped."""
    frames_dir = Path(manifest).parent
    counts = {"train": 0, "val": 0, "skipped": 0}
    if out_dir.exists():
        shutil.rmtree(out_dir)  # rebuilt from scratch so removed frames don't linger
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        label = Path(labels_dir) / Path(row["image"]).with_suffix(".txt")
        if not label.exists():
            counts["skipped"] += 1
            continue
        lines = [l for l in label.read_text(encoding="utf-8").splitlines() if l.strip()]
        write_item(out_dir, row["split"], frames_dir / row["image"], lines)
        counts[row["split"]] += 1
    write_dataset_yaml(out_dir, classes)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/frames/manifest.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels_auto"))
    parser.add_argument("--sop", type=Path, default=Path("sops/server_fan.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo"))
    args = parser.parse_args()
    counts = build_dataset(args.manifest, args.labels, load_sop(args.sop).detector.classes, args.out)
    print(f"dataset at {args.out}: {counts['train']} train, {counts['val']} val, {counts['skipped']} skipped (no labels)")


if __name__ == "__main__":
    main()
