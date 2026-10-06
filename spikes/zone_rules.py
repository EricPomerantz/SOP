"""Spike: which rule for "the bay a hand is at" agrees best with the step really being done?

For every recorded decision time (spikes/dump_scenes.py output), detects hands on that frame and assigns each a zone
under several rules, then checks the zone against the ground-truth step's bay. Usage: python spikes/zone_rules.py
"""

import json
import math
from collections import Counter
from pathlib import Path

import cv2

from sopvision.config import list_sops
from sopvision.dataset import load_index
from sopvision.perception.detector import YoloDetector
from sopvision.scene.geometry import box_distance_norm, to_norm
from sopvision.settings import Settings
from sopvision.sources.base import Frame


def norm_box(hand_box, chassis_box):
    (x1, y1), (x2, y2) = to_norm(hand_box[:2], chassis_box), to_norm(hand_box[2:], chassis_box)
    return x1, y1, x2, y2


def tip_point(nb, depth: float):
    """A point `depth` of the way from the fingertip edge (opposite the side sticking furthest out of the chassis,
    where the arm comes in) towards the box centre; the centre when the box doesn't stick out."""
    x1, y1, x2, y2 = nb
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side, out = max({"left": -x1, "right": x2 - 1, "top": -y1, "bottom": y2 - 1}.items(), key=lambda kv: kv[1])
    if out <= 0:
        return cx, cy
    edge = {"left": (x2, cy), "right": (x1, cy), "top": (cx, y2), "bottom": (cx, y1)}[side]
    return edge[0] + depth * (cx - edge[0]), edge[1] + depth * (cy - edge[1])


def zone(sop, hand_box, chassis_box, point) -> str:
    reach = 2 * sop.layout.match_radius
    u, v = point
    cands = [(math.hypot(s.at[0] - u, s.at[1] - v), n) for n, s in sop.layout.slots.items()
             if box_distance_norm(s.at, hand_box, chassis_box) <= reach]
    return min(cands)[1] if cands else "chassis/outside"


RULES = {
    "box centre (today)": lambda nb: tip_point(nb, 1.0),
    "fingertip edge": lambda nb: tip_point(nb, 0.0),
    "quarter in from fingertips": lambda nb: tip_point(nb, 0.25),
    "halfway (fingertips-centre)": lambda nb: tip_point(nb, 0.5),
}


def main() -> None:
    settings = Settings()
    sop = list_sops(settings.sops_dir)[0]
    index = {r.run: r for r in load_index(settings.sop_data_root)}
    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    bay_of = {s.id: next(iter(s.evidence)) for s in sop.steps if s.evidence}
    tally = {r: Counter() for r in RULES}
    n = 0
    for run_no in (12, 13):
        run = index[run_no]
        cap = cv2.VideoCapture(str(run.raw_path))
        for r in json.loads(Path(f"runs/scenes-Install_{run_no}.json").read_text(encoding="utf-8")):
            truth = next((s.step_id for s in run.segments if s.start <= r["t"] < s.end), None)
            if truth not in bay_of:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, r["t"] * 1000)
            ok, img = cap.read()
            if not ok:
                continue
            dets = detector.detect(Frame(r["t"], 0, img))
            chassis = max((d for d in dets if d.cls == sop.layout.anchor), key=lambda d: d.conf, default=None)
            if chassis is None:
                continue
            hands = [d.xyxy for d in dets if d.cls == "hand"]
            n += 1
            for name, rule in RULES.items():
                zones = {zone(sop, hb, chassis.xyxy, rule(norm_box(hb, chassis.xyxy))) for hb in hands}
                bays = zones - {"chassis/outside"}
                tally[name]["at the step's own bay"] += bay_of[truth] in bays
                tally[name]["at some other bay"] += bool(bays - {bay_of[truth]})
                tally[name]["at psu_bay_2 during a fan step"] += truth.startswith("fan") and "psu_bay_2" in bays
                tally[name]["at cover before the cover step"] += truth != "cover" and "cover" in bays
    print(f"{n} decision moments during steps with a bay\n")
    keys = list(next(iter(tally.values())))
    print(f"{'rule':30}" + "".join(f"{k:>33}" for k in keys))
    for name, c in tally.items():
        print(f"{name:30}" + "".join(f"{100 * c[k] / n:>32.0f}%" for k in keys))


if __name__ == "__main__":
    main()
