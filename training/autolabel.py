"""Zero-shot auto-labelling of sampled frames (spec §6.1 steps 2–3).

Each class is prompted separately with its text from the SOP config, so a box's class is exact.

Usage:
  uv run --group training python -m training.autolabel compare --n 20 --out data/autolabel_compare
  uv run --group training python -m training.autolabel run --labeler gdino --out data/labels_auto
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from sopvision.config import load_sop
from training.count_check import check, expected_counts
from training.yolo_format import LabeledBox, to_yolo

SINGLE_CLASSES = ("chassis", "cover")
COLORS = {"chassis": (255, 255, 255), "fan": (0, 200, 0), "fan_bay_empty": (160, 160, 160), "psu": (0, 140, 255),
          "psu_bay_empty": (120, 120, 200), "cover": (255, 200, 0), "hand": (255, 80, 80)}


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def postprocess(boxes: list[LabeledBox], iou: float = 0.5, single: tuple[str, ...] = SINGLE_CLASSES) -> list[LabeledBox]:
    """Class-wise NMS; keep only the best box for classes that appear once per frame."""
    kept: list[LabeledBox] = []
    for cls in dict.fromkeys(b.cls for b in boxes):
        ranked = sorted((b for b in boxes if b.cls == cls), key=lambda b: b.conf, reverse=True)
        chosen: list[LabeledBox] = []
        for b in ranked:
            if all(_iou(b.xyxy, c.xyxy) < iou for c in chosen):
                chosen.append(b)
        kept += chosen[:1] if cls in single else chosen
    return kept


def count_classes(boxes: list[LabeledBox]) -> dict[str, int]:
    return dict(Counter(b.cls for b in boxes))


def pick_spread(rows: list[dict], n: int) -> list[dict]:
    """n rows evenly spread over SOP positions (then runs)."""
    ordered = sorted(rows, key=lambda r: (r["position"], r["run"], r["image"]))
    if n >= len(ordered):
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]


class Labeler(Protocol):
    name: str

    def label(self, image_rgb: np.ndarray) -> list[LabeledBox]: ...


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class GroundingDinoLabeler:
    name = "gdino"

    def __init__(self, prompts: dict[str, str], model_id: str = "IDEA-Research/grounding-dino-base",
                 box_threshold: float = 0.35, text_threshold: float = 0.25):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.prompts = prompts
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.device = _device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def label(self, image_rgb: np.ndarray) -> list[LabeledBox]:
        import torch
        from PIL import Image

        image = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        boxes = []
        for cls, prompt in self.prompts.items():
            inputs = self.processor(images=image, text=[[prompt]], return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[(h, w)])[0]
            boxes += [LabeledBox(cls, tuple(float(v) for v in box.tolist()), float(score))
                      for score, box in zip(res["scores"], res["boxes"])]
        return postprocess(boxes)


class Sam3Labeler:
    name = "sam3"

    def __init__(self, prompts: dict[str, str], model_id: str = "facebook/sam3", threshold: float = 0.5):
        from transformers import Sam3Model, Sam3Processor

        self.prompts = prompts
        self.threshold = threshold
        self.device = _device()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

    def label(self, image_rgb: np.ndarray) -> list[LabeledBox]:
        import torch
        from PIL import Image

        image = Image.fromarray(image_rgb)
        boxes = []
        for cls, prompt in self.prompts.items():
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                out, threshold=self.threshold, mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist())[0]
            boxes += [LabeledBox(cls, tuple(float(v) for v in box.tolist()), float(score))
                      for score, box in zip(res["scores"], res["boxes"])]
        return postprocess(boxes)


LABELERS = {"gdino": GroundingDinoLabeler, "sam3": Sam3Labeler}


def _read_manifest(frames: Path) -> list[dict]:
    return [json.loads(line) for line in (frames / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


def _check_row(row: dict, boxes: list[LabeledBox]):
    counts = count_classes(boxes)
    return check(counts, expected_counts(row["actions_by_position"], row["position"]), hands=counts.get("hand", 0))


def _overlay(image_bgr: np.ndarray, boxes: list[LabeledBox], caption: str) -> np.ndarray:
    out = image_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy)
        color = COLORS.get(b.cls, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, f"{b.cls} {b.conf:.2f}", (x1 + 4, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(out, caption, (10, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return out


def compare(frames: Path, out: Path, n: int, names: list[str], prompts: dict[str, str]) -> dict[str, int]:
    rows = pick_spread(_read_manifest(frames), n)
    summary = {}
    for name in names:
        try:
            labeler = LABELERS[name](prompts)
        except Exception as e:  # e.g. gated model without access
            print(f"{name}: unavailable ({type(e).__name__}: {str(e)[:120]})")
            continue
        passed = 0
        for row in rows:
            image = cv2.imread(str(frames / row["image"]))
            boxes = labeler.label(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result = _check_row(row, boxes)
            passed += result.ok
            caption = "OK" if result.ok else "FLAG: " + "; ".join(result.reasons)
            target = out / name / Path(row["image"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), _overlay(image, boxes, caption))
        summary[name] = passed
        print(f"{name}: {passed}/{len(rows)} pass")
    (out / "summary.json").write_text(json.dumps({"n": len(rows), "passed": summary}, indent=2), encoding="utf-8")
    return summary


def run(frames: Path, out: Path, name: str, prompts: dict[str, str], classes: list[str]) -> None:
    labeler = LABELERS[name](prompts)
    flags = []
    for i, row in enumerate(_read_manifest(frames)):
        image = cv2.imread(str(frames / row["image"]))
        h, w = image.shape[:2]
        boxes = labeler.label(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        target = out / Path(row["image"]).with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(to_yolo(b, w, h, classes) + "\n" for b in boxes), encoding="utf-8")
        result = _check_row(row, boxes)
        flags.append({"image": row["image"], "ok": result.ok, "reasons": result.reasons})
        if (i + 1) % 50 == 0:
            print(f"{i + 1} frames labelled")
    (out / "flags.jsonl").write_text("".join(json.dumps(f) + "\n" for f in flags), encoding="utf-8")
    ok = sum(f["ok"] for f in flags)
    print(f"{name}: {ok}/{len(flags)} frames pass the count check; {len(flags) - ok} flagged for review first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("compare", "run"):
        p = sub.add_parser(cmd)
        p.add_argument("--frames", type=Path, default=Path("data/frames"))
        p.add_argument("--sop", type=Path, default=Path("sops/server_fan.yaml"))
        p.add_argument("--out", type=Path, required=True)
    sub.choices["compare"].add_argument("--n", type=int, default=20)
    sub.choices["compare"].add_argument("--labelers", default="gdino,sam3")
    sub.choices["run"].add_argument("--labeler", choices=sorted(LABELERS), required=True)
    args = parser.parse_args()
    sop = load_sop(args.sop)
    prompts = sop.detector.autolabel_prompts
    if args.cmd == "compare":
        compare(args.frames, args.out, args.n, args.labelers.split(","), prompts)
    else:
        run(args.frames, args.out, args.labeler, prompts, sop.detector.classes)


if __name__ == "__main__":
    main()
