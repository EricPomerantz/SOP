"""Throwaway: try Ultralytics YOLOE text prompts on the same 20 comparison frames."""
import json
from pathlib import Path

import cv2
from ultralytics import YOLOE

from sopvision.config import load_sop
from training.autolabel import _check_row, _overlay, pick_spread, postprocess
from training.yolo_format import LabeledBox

sop = load_sop(Path("sops/server_fan.yaml"))
prompts = sop.detector.autolabel_prompts
classes = list(prompts)
model = YOLOE("yoloe-11l-seg.pt")
model.set_classes(list(prompts.values()), model.get_text_pe(list(prompts.values())))
rows = pick_spread([json.loads(l) for l in Path("data/frames/manifest.jsonl").read_text().splitlines()], 20)
out = Path("data/autolabel_compare/yoloe"); out.mkdir(parents=True, exist_ok=True)
passed = 0
for row in rows:
    img = cv2.imread(f"data/frames/{row['image']}")
    r = model.predict(img, conf=0.15, verbose=False)[0]
    boxes = postprocess([LabeledBox(classes[int(c)], tuple(b), float(s)) for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())])
    res = _check_row(row, boxes); passed += res.ok
    cv2.imwrite(str(out / Path(row["image"]).name), _overlay(img, boxes, "OK" if res.ok else "FLAG: " + "; ".join(res.reasons)))
print(f"yoloe: {passed}/{len(rows)} pass")
