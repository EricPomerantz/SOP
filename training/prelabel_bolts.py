"""Pre-label lug-bolt frames for review in Label Studio: hub, empty holes and coloured bolts by colour, plus the zero-shot
hand / wrench boxes from training/zero_shot_boxes.py. Everything here is a suggestion a person corrects.

The camera, hub and wheel stay put between runs (the wheel isn't turned), so the five holes sit at fixed pixel positions
up to camera shake. Shake is measured per frame (phase correlation of the wheel against a reference frame) and the holes
moved with it. At each hole, a bolt is whichever colour fills the patch; with no colour it's an empty hole, unless a
hand or the wrench covers it (then no box).

    docker compose run --rm -v ./training:/app/training:ro -v ./data/lug_bolts:/app/data/lug_bolts \
        --entrypoint python sopvision -m training.prelabel_bolts --sop sops/lug_bolts.yaml \
        --frames data/lug_bolts/frames --zero-shot data/lug_bolts/zero_shot.jsonl --out data/lug_bolts/labels_auto
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sopvision.config import load_sop
from training.yolo_format import LabeledBox, to_yolo

# Hole centres (px, 1920x1080) in the reference frame, holes 1-5 as the SOP numbers them (top, then clockwise).
HOLES = {"hole_1": (991, 433), "hole_2": (1053, 514), "hole_3": (996, 593), "hole_4": (895, 572), "hole_5": (894, 470)}
REF = ("correct7", 61.0)  # the reference frame (run, seconds): all five bolts in, nobody in view
WHEEL = (slice(300, 750), slice(750, 1250))  # rows, cols compared for shake (frame to frame)
RING = (slice(380, 650), slice(820, 1120))  # the ring of holes: each run's offset (median over its frames)
PATCH = 18  # half-size of the patch looked at around a hole
BOLT, EMPTY, SOCKET = 50, 42, 90  # box sizes (px)
HUB_HALF = 120  # hub box: the ring of holes plus about a bolt head (the SOP's slot coordinates assume this)
MIN_FILL = 0.25  # fraction of the patch a colour must cover to call a bolt
MAX_SIDE = {"torque_wrench": 260}  # zero-shot "wrench head" boxes bigger than this are the wheel or the whole wrench


def colour_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return {
        "lug_bolt_red": (s > 100) & (v > 80) & ((h >= 160) | (h <= 6)),
        "lug_bolt_gold": (s > 100) & (v > 120) & (h >= 18) & (h <= 35),
        "lug_bolt_blue": (s > 100) & (v > 80) & (h >= 95) & (h <= 125),
        "lug_bolt_green": (s > 40) & (v > 100) & (h >= 45) & (h <= 94),
    }


def shift(ref: np.ndarray, image: np.ndarray, region) -> tuple[float, float]:
    """Translation of `region` of image against the same region of the reference frame (0, 0 when unsure)."""
    a = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)[region].astype(np.float32)
    b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[region].astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return (dx, dy) if response > 0.05 and abs(dx) < 60 and abs(dy) < 60 else (0.0, 0.0)


def hole_shifts(ref: np.ndarray, images: list[np.ndarray]) -> list[tuple[float, float]]:
    """Per frame of one run: where the holes moved. The run's offset comes from the ring of holes (median, so hands
    over it don't matter); frame-to-frame shake from the whole wheel, which hands cover less."""
    ring = np.median([shift(ref, im, RING) for im in images], axis=0)
    wheel = [shift(ref, im, WHEEL) for im in images]
    base = np.median(wheel, axis=0)
    return [(float(ring[0] + w[0] - base[0]), float(ring[1] + w[1] - base[1])) for w in wheel]


def _square(cx: float, cy: float, size: float) -> tuple[float, float, float, float]:
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def _fits(box: dict) -> bool:
    x1, y1, x2, y2 = box["xyxy"]
    return max(x2 - x1, y2 - y1) <= MAX_SIDE.get(box["cls"], 10_000)


def _covered(x: float, y: float, occluders: list[dict]) -> bool:
    return any(b["xyxy"][0] <= x <= b["xyxy"][2] and b["xyxy"][1] <= y <= b["xyxy"][3] for b in occluders)


def label_frame(image: np.ndarray, d: tuple[float, float], zero_shot: list[dict],
                had_bolt: set[str]) -> tuple[list[LabeledBox], list[str]]:
    """Boxes for one frame. had_bolt: holes that showed a bolt earlier in the run (updated here); one of those showing
    no colour now is taken to have the wrench socket over it, not to be empty (a removed bolt is the exception, so
    those frames are flagged)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = colour_masks(hsv)
    boxes, notes = [], []
    centres = {k: (x + d[0], y + d[1]) for k, (x, y) in HOLES.items()}
    cx, cy = np.mean([c[0] for c in centres.values()]), np.mean([c[1] for c in centres.values()])
    boxes.append(LabeledBox("hub", (cx - HUB_HALF, cy - HUB_HALF, cx + HUB_HALF, cy + HUB_HALF), 1.0))
    for hole, (x, y) in centres.items():
        xi, yi = int(round(x)), int(round(y))
        fills = {c: float(m[yi - PATCH:yi + PATCH, xi - PATCH:xi + PATCH].mean()) for c, m in masks.items()}
        cls, fill = max(fills.items(), key=lambda kv: kv[1])
        if fill >= MIN_FILL:
            ys, xs = np.nonzero(masks[cls][yi - PATCH:yi + PATCH, xi - PATCH:xi + PATCH])
            boxes.append(LabeledBox(cls, _square(xi - PATCH + xs.mean(), yi - PATCH + ys.mean(), BOLT), fill))
            had_bolt.add(hole)
        elif hole in had_bolt and _covered(x, y, [b for b in zero_shot if b["cls"] == "hand"]):
            notes.append(f"{hole}: under a hand (bolt or wrench?)")
        elif hole in had_bolt:
            if not _covered(x, y, [b for b in zero_shot if b["cls"] == "torque_wrench"]):
                boxes.append(LabeledBox("torque_wrench", _square(x, y, SOCKET), 0.5))
            notes.append(f"{hole}: wrench on it, or the bolt was removed?")
        elif _covered(x, y, zero_shot):
            notes.append(f"{hole} covered")
        else:
            boxes.append(LabeledBox("hole_empty", _square(x, y, EMPTY), 1.0))
    boxes += [LabeledBox(b["cls"], tuple(b["xyxy"]), b["conf"]) for b in zero_shot]
    return boxes, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sop", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--zero-shot", type=Path, help="training/zero_shot_boxes.py output (hand, torque_wrench)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", type=Path, default=Path("/datasets/lug_bolts"))
    ap.add_argument("--sheets", type=Path, help="also write overlay images here, for a quick look")
    args = ap.parse_args()
    classes = load_sop(args.sop).detector.classes
    cap = cv2.VideoCapture(str(args.datasets / "runs" / REF[0] / "video.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, REF[1] * 1000)
    ok, ref = cap.read()
    if not ok:
        raise SystemExit(f"can't read the reference frame {REF}")
    zero_shot = {}
    if args.zero_shot:
        zero_shot = {r["image"]: [b for b in r["boxes"] if _fits(b)]
                     for r in map(json.loads, args.zero_shot.read_text().splitlines())}
    rows = [json.loads(l) for l in (args.frames / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    flags = []
    shifts = {}
    for run in dict.fromkeys(r["run"] for r in rows):
        run_rows = [r for r in rows if r["run"] == run]
        shifts |= dict(zip((r["image"] for r in run_rows),
                           hole_shifts(ref, [cv2.imread(str(args.frames / r["image"])) for r in run_rows])))
    had_bolt: dict[str, set[str]] = {}
    for row in sorted(rows, key=lambda r: (r["run"], r["t"])):
        image = cv2.imread(str(args.frames / row["image"]))
        h, w = image.shape[:2]
        d = shifts[row["image"]]
        boxes, notes = label_frame(image, d, zero_shot.get(row["image"], []), had_bolt.setdefault(row["run"], set()))
        target = args.out / Path(row["image"]).with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(to_yolo(b, w, h, classes) + "\n" for b in boxes), encoding="utf-8")
        # Review first: frames where something covers a hole or the wrench is (or should be) in view.
        reasons = notes + (["torque press: check the wrench box"] if row.get("tag") == "torque" else [])
        flags.append({"image": row["image"], "ok": not reasons, "reasons": reasons, "shift": [round(d[0], 1), round(d[1], 1)]})
        if args.sheets:
            _overlay(image, boxes, f"{row['image']} shift {d[0]:.0f},{d[1]:.0f}", args.sheets / Path(row["image"]).name)
    (args.out / "flags.jsonl").write_text("".join(json.dumps(f) + "\n" for f in flags), encoding="utf-8")
    print(f"{len(rows)} frames labelled -> {args.out}; {sum(not f['ok'] for f in flags)} flagged for review first")


COLOURS = {"hub": (255, 255, 255), "hole_empty": (160, 160, 160), "lug_bolt_blue": (255, 0, 0),
           "lug_bolt_green": (0, 200, 0), "lug_bolt_red": (0, 0, 255), "lug_bolt_gold": (0, 215, 255),
           "torque_wrench": (255, 0, 255), "hand": (255, 200, 80)}


def _overlay(image: np.ndarray, boxes: list[LabeledBox], caption: str, path: Path) -> None:
    out = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOURS.get(b.cls, (0, 255, 255)), 2)
    cv2.putText(out, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out[200:900, 600:1400])


if __name__ == "__main__":
    main()
