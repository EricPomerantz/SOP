"""Throwaway: which short prompts does SAM 3 respond to on these server frames?

Prints, per prompt and frame, how many instances score >= 0.3 and the top scores,
and saves overlays of every prompt's detections to data/autolabel_compare/sam3_prompts/.
"""

from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

FRAMES = [  # (frame, what's true in it)
    "train/Install_4_02_p10.jpg",   # fan 1 starting: 6 empty fan bays, 2 empty PSU bays
    "train/Install_4_06_p90.jpg",   # fan 5 ending: ~5 fans, ~1 empty bay
    "train/Install_4_09_p65.jpg",   # PSU 2: 6 fans, 1-2 PSUs
    "train/Install_4_11_p35.jpg",   # after cover: cover on
]
VARIANTS = {
    "chassis": ["server", "server chassis", "rack server"],
    "fan": ["fan", "cooling fan", "black fan"],
    "fan_bay_empty": ["fan slot", "empty slot", "empty fan slot"],
    "psu": ["power supply", "power supply unit"],
    "psu_bay_empty": ["empty bay", "empty power supply bay"],
    "cover": ["server cover", "lid", "metal cover"],
}
THRESHOLD = 0.3

device = "cuda"
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
out_dir = Path("data/autolabel_compare/sam3_prompts")
out_dir.mkdir(parents=True, exist_ok=True)

for frame in FRAMES:
    bgr = cv2.imread(f"data/frames/{frame}")
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(f"\n== {frame}")
    for cls, prompts in VARIANTS.items():
        for prompt in prompts:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            res = processor.post_process_instance_segmentation(
                out, threshold=THRESHOLD, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
            scores = sorted((float(s) for s in res["scores"]), reverse=True)
            print(f"  {cls:14s} {prompt!r:28s} n={len(scores):2d} top={[round(s, 2) for s in scores[:4]]}")
            vis = bgr.copy()
            for s, b in zip(res["scores"], res["boxes"]):
                x1, y1, x2, y2 = (int(v) for v in b.tolist())
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f"{float(s):.2f}", (x1 + 4, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{Path(frame).stem}__{prompt.replace(' ', '_')}.jpg"), cv2.resize(vis, (960, 540)))
