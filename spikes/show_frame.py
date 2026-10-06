"""Spike: draw one frame with detections, each hand's centre and zone, and every slot centre with its reach circle.

Usage: python spikes/show_frame.py --run 12 --t 27.0 [--t 34.3 ...] -> runs/frame-Install_12-27.0.jpg
"""

import argparse
from pathlib import Path

import cv2

from sopvision.config import list_sops
from sopvision.dataset import load_index
from sopvision.perception.detector import YoloDetector
from sopvision.scene.builder import SceneBuilder
from sopvision.settings import Settings
from sopvision.sources.base import Frame


def main(run_no: int, times: list[float]) -> None:
    settings = Settings()
    sop = list_sops(settings.sops_dir)[0]
    run = {r.run: r for r in load_index(settings.sop_data_root)}[run_no]
    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    cap = cv2.VideoCapture(str(run.raw_path))
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            print(f"no frame at {t}")
            continue
        dets = detector.detect(Frame(t, 0, img))
        builder = SceneBuilder(sop)
        scene, _ = builder.update(t, dets)
        box = scene.chassis_box
        to_px = lambda u, v: (int(box[0] + u * (box[2] - box[0])), int(box[1] + v * (box[3] - box[1])))
        reach = 2 * sop.layout.match_radius
        for name, spec in sop.layout.slots.items():
            c = to_px(*spec.at)
            cv2.circle(img, c, int(reach * (box[2] - box[0])), (200, 200, 200), 1)
            cv2.circle(img, c, 6, (0, 255, 255), -1)
            cv2.putText(img, name, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for d in dets:
            x1, y1, x2, y2 = map(int, d.xyxy)
            colour = (0, 0, 255) if d.cls == "hand" else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(img, d.cls, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if d.cls == "hand":
                cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 8, (0, 0, 255), -1)
        cv2.putText(img, f"t={t}s hands near: {[h.near for h in scene.hands]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        out = Path(f"runs/frame-Install_{run_no}-{t}.jpg")
        cv2.imwrite(str(out), cv2.resize(img, None, fx=0.5, fy=0.5))
        print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=12)
    ap.add_argument("--t", type=float, action="append", required=True)
    a = ap.parse_args()
    main(a.run, a.t)
