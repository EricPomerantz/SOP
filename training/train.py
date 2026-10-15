"""Train the SOP detector and benchmark it (spec §6.1 step 5).

Usage:
    docker compose run --rm trainer train --data data/yolo/dataset.yaml
    docker compose run --rm trainer bench --weights models/server_fan/best.pt --video <raw video>
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2


def summarize_metrics(results_dict: dict) -> dict:
    return {"map50": round(float(results_dict["metrics/mAP50(B)"]), 4),
            "map50_95": round(float(results_dict["metrics/mAP50-95(B)"]), 4)}


def train(data: Path, model: str, imgsz: int, epochs: int, out: Path, batch: int, workers: int = 2,
          project: Path = Path("data/ultralytics")) -> dict:
    from ultralytics import YOLO

    result = YOLO(model).train(data=str(data), imgsz=imgsz, epochs=epochs, batch=batch, workers=workers, project=str(project.resolve()),
                               name=out.name, exist_ok=True, patience=30, seed=0, verbose=False)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(result.save_dir) / "weights" / "best.pt", out / "best.pt")
    metrics = summarize_metrics(result.results_dict) | {"model": model, "imgsz": imgsz, "epochs": epochs,
                                                        "run_dir": str(result.save_dir)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def bench(weights: Path, video: Path, frames: int, imgsz: int) -> float:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    images = []
    while len(images) < frames:
        ok, img = cap.read()
        if not ok:
            break
        images.append(img)
    cap.release()
    model.track(images[0], persist=True, imgsz=imgsz, verbose=False)  # warm-up
    start = time.perf_counter()
    for img in images:
        model.track(img, persist=True, imgsz=imgsz, verbose=False)
    return len(images) / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=Path("data/yolo/dataset.yaml"))
    t.add_argument("--model", default="yolo26m.pt")
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=4)  # 8 at imgsz 1280 fills a 16 GB GPU and thrashes on Windows
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--out", type=Path, default=Path("models/server_fan"))
    b = sub.add_parser("bench")
    b.add_argument("--weights", type=Path, default=Path("models/server_fan/best.pt"))
    b.add_argument("--video", type=Path, required=True)
    b.add_argument("--frames", type=int, default=300)
    b.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    if args.cmd == "train":
        print(json.dumps(train(args.data, args.model, args.imgsz, args.epochs, args.out, args.batch, args.workers), indent=2))
    else:
        print(f"{bench(args.weights, args.video, args.frames, args.imgsz):.1f} fps (detect + track, imgsz {args.imgsz})")


if __name__ == "__main__":
    main()
