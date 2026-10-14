"""Zero-shot boxes (Grounding DINO) for the classes colour can't find: hands and the torque wrench.

Standalone (transformers + Pillow only) so it runs in the GPU image that already has them, the Decider's:

    docker compose --profile decider run --rm --no-deps -v ./training:/w/training:ro -v ./data/lug_bolts:/w/data \
        --entrypoint python decider /w/training/zero_shot_boxes.py /w/data/frames /w/data/zero_shot.jsonl \
        --prompt hand="a hand." --prompt torque_wrench="the head of a torque wrench."

Writes one line per manifest image: {"image", "boxes": [{"cls", "xyxy", "conf"}]} (pixels). Each class is prompted on
its own, so a box's class is exact. training/prelabel_bolts.py merges these with its colour boxes.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[dict], iou: float = 0.5) -> list[dict]:
    kept = []
    for b in sorted(boxes, key=lambda b: b["conf"], reverse=True):
        if all(b["cls"] != k["cls"] or _iou(b["xyxy"], k["xyxy"]) < iou for k in kept):
            kept.append(b)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", action="append", required=True, help="class=text prompt (repeatable)")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--max-area", type=float, default=0.25, help="drop boxes larger than this fraction of the image")
    args = ap.parse_args()
    prompts = dict(p.split("=", 1) for p in args.prompt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    rows = [json.loads(l) for l in (args.frames / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            image = Image.open(args.frames / row["image"]).convert("RGB")
            w, h = image.size
            boxes = []
            for cls, text in prompts.items():
                inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inputs)
                res = processor.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=args.threshold, text_threshold=0.25, target_sizes=[(h, w)])[0]
                for score, box in zip(res["scores"], res["boxes"]):
                    x1, y1, x2, y2 = (round(float(v), 1) for v in box.tolist())
                    if (x2 - x1) * (y2 - y1) <= args.max_area * w * h:
                        boxes.append({"cls": cls, "xyxy": [x1, y1, x2, y2], "conf": round(float(score), 3)})
            f.write(json.dumps({"image": row["image"], "boxes": nms(boxes)}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)}", flush=True)
    print(f"{len(rows)} frames -> {args.out}")


if __name__ == "__main__":
    main()
