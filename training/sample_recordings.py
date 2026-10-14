"""Sample detector-training frames from an SOP's timeline recordings (train and val runs only, never test).

Frames every --every seconds through each run, plus one just before each torque press, when the wrench is on a bolt.
Writes <out>/<split>/<run>_<ms>.jpg and <out>/manifest.jsonl ({"image", "run", "split", "t", "tag"}), the manifest
the Label Studio round trip (training/labelstudio.py) and training/yolo_dataset.py read.

    docker compose run --rm -v ./training:/app/training:ro -v ./data/lug_bolts:/app/data/lug_bolts \
        --entrypoint python sopvision -m training.sample_recordings --sop sops/lug_bolts.yaml \
        --out data/lug_bolts/frames
"""

import argparse
import json
from pathlib import Path

import cv2

from sopvision.config import load_sop
from sopvision.recordings import load_recordings
from sopvision.settings import Settings
from sopvision.sop.torque import read_torque_log

PRESS_LEAD = 0.3  # seconds before a torque press: the wrench is still on the bolt


def plan(duration: float, presses: list[float], every: float) -> list[tuple[float, str]]:
    """(t, tag) sample times: a regular grid from 0.5 s, plus one before each press; near-duplicates dropped."""
    times = [(t, "grid") for t in _grid(duration, every)]
    for p in presses:
        t = round(max(0.0, p - PRESS_LEAD), 2)
        if t < duration and all(abs(t - u) > 0.5 for u, _ in times):
            times.append((t, "torque"))
    return sorted(times)


def _grid(duration: float, every: float) -> list[float]:
    out, t = [], 0.5
    while t < duration - 0.2:
        out.append(round(t, 2))
        t += every
    return out


def extract(video: Path, run: str, split: str, times: list[tuple[float, str]], out: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    rows = []
    try:
        for t, tag in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, image = cap.read()
            if not ok:
                continue
            image_name = f"{split}/{run}_{round(t * 1000):06d}.jpg"
            (out / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"image": image_name, "run": run, "split": split, "t": round(t, 2), "tag": tag})
    finally:
        cap.release()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sop", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=float, default=4.0, help="seconds between regular samples")
    args = ap.parse_args()
    sop = load_sop(args.sop)
    if (args.out / "manifest.jsonl").exists():
        raise SystemExit(f"{args.out}/manifest.jsonl exists; refusing to overwrite (labels refer to its images)")
    rows = []
    for rec in load_recordings(sop, Settings()):
        if rec.split == "test":  # test runs are never sampled for training
            continue
        meta = json.loads((rec.video.parent / "meta.json").read_text(encoding="utf-8"))
        presses = [r.t for r in read_torque_log(rec.torque_log)] if rec.torque_log else []
        rows += extract(rec.video, rec.id, rec.split, plan(meta["seconds"], presses, args.every), args.out)
        print(f"{rec.split}:{rec.id}: {sum(r['run'] == rec.id for r in rows)} frames")
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} frames -> {args.out}/manifest.jsonl")


if __name__ == "__main__":
    main()
