"""Smoke test for Mapika/decider-2b-vision: image + text context + bounded options -> probabilities.

Preparation only: nothing in sopvision uses this model yet. Run it in the Decider container, ideally with the
4B service stopped (both models on a 16 GB GPU next to YOLO is more than this test needs):

    docker compose --profile decider run --rm decider python scripts/decider_vision_smoke.py

Two decisions:
  1. a synthetic image with a known answer (a red square: "what colour is the square?") — the model must get it;
  2. a real frame from a test recording, with the SOP context and the SOP's step options — any valid distribution
     passes; the answer is printed, not judged (that is the later vision experiment).
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker"))
from decider_serve import DEFAULTS, ensure_snapshot  # noqa: E402

REPO = "Mapika/decider-2b-vision"
FRAME_VIDEO = Path(os.environ.get("SMOKE_VIDEO", "/dataset/server_fan/raw/Install_12.MP4"))
FRAME_AT_S = float(os.environ.get("SMOKE_FRAME_AT", "60"))
STEP_OPTIONS = ["fan_1: Install the first fan", "fan_2: Install the second fan", "fan_3: Install the third fan",
                "psu_1: Install the first power supply", "cover: Place the cover on top of the server",
                "idle: No one is working on the server"]


def load_model():
    import torch

    path = ensure_snapshot(REPO, os.environ.get("DECIDER_VISION_REVISION") or DEFAULTS[REPO])
    sys.path.insert(0, str(path))  # the snapshot's own decider/ package (vision variant)
    from decider.infer import Example, Q
    from decider.vision import VisionDecisionModel

    start = time.perf_counter()
    model = VisionDecisionModel(str(path), grad_ckpt=False).cuda().eval()
    print(f"[vision] loaded in {time.perf_counter() - start:.1f}s, "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated", flush=True)
    return model, Example, Q


def decide(model, Example, Q, image, context: str, question: str, options: list[str]) -> dict:
    import torch

    start = time.perf_counter()
    with torch.no_grad():
        logits = model.slot_logits(model.prepare([(image, Example(context, [Q(question, options, 0)]))]))
    probs = torch.softmax(logits, -1)[0, :len(options)].tolist()
    best = max(range(len(options)), key=probs.__getitem__)
    return {"choice": options[best], "probabilities": dict(zip(options, (round(p, 4) for p in probs))),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1)}


def red_square():
    import numpy as np

    img = np.full((224, 224, 3), 255, np.uint8)
    img[60:164, 60:164] = (220, 20, 20)
    return img


def recording_frame():
    import cv2

    cap = cv2.VideoCapture(str(FRAME_VIDEO))
    cap.set(cv2.CAP_PROP_POS_MSEC, FRAME_AT_S * 1000)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = 640 / max(h, w)  # keep the visual token count modest
    return cv2.resize(rgb, (int(w * scale), int(h * scale)))


def check(result: dict, options: list[str]) -> None:
    probs = result["probabilities"]
    assert set(probs) == set(options), probs
    assert abs(sum(probs.values()) - 1) < 1e-3, probs
    assert result["choice"] in options


def main() -> int:
    model, Example, Q = load_model()
    colours = ["red", "green", "blue", "yellow"]
    known = decide(model, Example, Q, red_square(), "A picture of a coloured square.", "What colour is the square?", colours)
    print("[vision] known decision:", json.dumps(known), flush=True)
    check(known, colours)
    assert known["choice"] == "red", f"expected red, got {known}"

    frame = recording_frame()
    if frame is None:
        print(f"[vision] no frame from {FRAME_VIDEO}; skipped the SOP frame decision", flush=True)
        return 0
    context = ("Server fan install SOP, seen from above. Steps: six fans, two power supplies, then the cover. "
               "Detected by YOLO: chassis visible; fan_bay_1 installed; fan_bay_2 installed; fan_bay_3 empty; "
               "hand near fan_bay_3; holding: fan.")
    sop = decide(model, Example, Q, frame, context, "Which step is the operator performing right now?", STEP_OPTIONS)
    print("[vision] SOP frame decision (not judged):", json.dumps(sop), flush=True)
    check(sop, STEP_OPTIONS)
    print("[vision] smoke test passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
