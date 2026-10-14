import importlib.util
import random
import sys
from pathlib import Path

from sopvision.dataset import Segment

sys.path.insert(0, str(Path("scripts").resolve()))
_spec = importlib.util.spec_from_file_location("build_training_set", "scripts/build_training_set.py")
bts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bts)

STEPS = ["fan_1", "fan_2", "fan_3", "fan_4", "fan_5", "fan_6", "psu_1", "psu_2", "cover"]
SEGS = [Segment("off_sop", 0.0, 5.0), Segment("fan_1", 5.0, 15.0), Segment("fan_2", 15.0, 25.0)]


def record(t, active="fan_2", gt=("fan_2",), done=("fan_1",)):
    slots = {f"fan_bay_{i}": {"state": "installed" if f"fan_{i}" in done else "empty", "conf": 0.9} for i in range(1, 7)}
    return {"id": f"train:Install_1/{t}", "source": "train:Install_1", "request_id": str(t), "t_request": t,
            "request": {"state": {"sop": "x", "steps": [{"id": s, "status": "done" if s in done else "pending"} for s in STEPS],
                                  "scene": {"t": t, "slots": slots, "hands": [{"near": "fan_bay_2"}], "held": []},
                                  "recent_events": [f"{t - 1:.1f}s hand entered fan_bay_2"]},
                        "questions": {"current_step_options": {s: s for s in STEPS}, "active_step_id": active,
                                      "active_step_question": "done?"}},
            "tracker": {"done_t": {"fan_1": 15.5}},
            "ground_truth": {"current_step": list(gt), "anomaly": False}}


def test_labels_come_from_the_timeline():
    assert bts.labels(record(20.0), SEGS) == {"current_step": "fan_2", "active_step_done": False, "anomaly": False}
    assert bts.labels(record(27.0), SEGS)["active_step_done"] is True
    assert "active_step_done" not in bts.labels(record(25.5), SEGS)  # within 1 s of the step's end
    assert bts.labels(record(2.0, gt=("idle", "off_sop")), SEGS)["current_step"] == "idle"


def test_noise_changes_the_scene_not_the_record():
    rec = record(20.0)
    before = repr(rec)
    changed = [bts.noisy(rec, random.Random(i)) for i in range(20)]
    assert repr(rec) == before
    assert any(c["request"]["state"] != rec["request"]["state"] for c in changed)


def test_deviation_labels(sop):
    rec, lab = record(20.0), {"current_step": "fan_2", "active_step_done": False, "anomaly": False}
    _, work = bts.deviation(rec, "out_of_order_work", lab, sop)
    assert work == {"current_step": "fan_4", "active_step_done": False}
    _, install = bts.deviation(rec, "out_of_order_install", lab, sop)
    assert install == {"anomaly": True, "active_step_done": False}
    _, idle = bts.deviation(rec, "idle_bench", lab, sop)
    assert idle == {"current_step": "idle", "anomaly": False, "active_step_done": False}


def test_rows_use_the_wire_questions():
    r = bts.row(record(20.0), {"current_step": "fan_2"}, "in_order", "train", 0)
    assert r["questions"]["current_step"]["type"] == "choice" and r["questions"]["anomaly"]["type"] == "noul"
    assert r["expected"] == {"current_step": "fan_2"} and r["split"] == "train"
