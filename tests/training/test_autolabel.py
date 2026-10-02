from training.autolabel import count_classes, pick_spread, postprocess
from training.yolo_format import LabeledBox



def test_postprocess_nms_and_single_chassis():
    boxes = [LabeledBox("chassis", (0, 0, 10, 10), 0.6), LabeledBox("chassis", (1, 1, 11, 11), 0.9),
             LabeledBox("fan", (0, 0, 10, 10), 0.8), LabeledBox("fan", (1, 0, 11, 10), 0.7),
             LabeledBox("fan", (20, 0, 30, 10), 0.7)]
    out = postprocess(boxes)
    assert [b.conf for b in out if b.cls == "chassis"] == [0.9]
    assert len([b for b in out if b.cls == "fan"]) == 2



def test_count_classes():
    boxes = [LabeledBox("fan", (0, 0, 1, 1), 1), LabeledBox("fan", (2, 2, 3, 3), 1), LabeledBox("hand", (0, 0, 1, 1), 1)]
    assert count_classes(boxes) == {"fan": 2, "hand": 1}


def test_pick_spread_covers_every_position():
    rows = [{"image": f"r{r}_p{p}_{k}", "run": r, "position": p} for r in range(1, 11) for p in range(1, 12) for k in range(6)]
    picked = pick_spread(rows, 20)
    assert len(picked) == 20 and {r["position"] for r in picked} == set(range(1, 12))
