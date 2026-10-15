"""Contact sheets of generated labels, one per recording, for a quick human check.

Usage: docker compose run --rm tools -m training.review_sheets   (writes data/labels_review/Install_<run>.jpg)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from sopvision.config import load_sop
from training.autolabel import pick_spread
from training.yolo_format import from_yolo

COLORS = {"chassis": (255, 255, 255), "fan": (0, 200, 0), "fan_bay_empty": (200, 200, 200), "psu": (0, 140, 255),
          "psu_bay_empty": (200, 120, 255), "cover": (255, 200, 0), "hand": (255, 80, 80)}
TILE = (640, 360)


def draw_labels(img: np.ndarray, lines: list[str], classes: list[str]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for line in lines:
        b = from_yolo(line, w, h, classes)
        x1, y1, x2, y2 = (int(round(v)) for v in b.xyxy)
        color = COLORS.get(b.cls, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, w // 400))
        cv2.putText(out, b.cls, (x1 + 4, y1 + max(12, h // 40)), cv2.FONT_HERSHEY_SIMPLEX, max(0.3, w / 2400), color,
                    max(1, w // 800))
    return out


def sheets(frames: Path, labels: Path, out: Path, classes: list[str], per_run: int = 12) -> list[Path]:
    rows = [json.loads(l) for l in (frames / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    flags = {f["image"]: f for f in (json.loads(l) for l in (labels / "flags.jsonl").read_text(encoding="utf-8").splitlines())}
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for run, run_rows in sorted(by_run.items()):
        tiles = []
        for r in pick_spread(run_rows, per_run):
            img = cv2.imread(str(frames / r["image"]))
            label = labels / Path(r["image"]).with_suffix(".txt")
            lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
            tile = cv2.resize(draw_labels(img, lines, classes), TILE)
            f = flags.get(r["image"], {})
            note = "NO LABELS (alignment failed)" if not label.exists() else "; ".join(f.get("reasons", [])) or "ok"
            cv2.putText(tile, f"{Path(r['image']).stem}  {note}"[:80], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        path = out / f"Install_{run}.jpg"
        cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels_auto"))
    parser.add_argument("--sop", type=Path, default=Path("sops/server_fan.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/labels_review"))
    args = parser.parse_args()
    for p in sheets(args.frames, args.labels, args.out, load_sop(args.sop).detector.classes):
        print(p)


if __name__ == "__main__":
    main()
