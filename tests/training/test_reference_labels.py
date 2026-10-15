import json
from types import SimpleNamespace

import pytest

from training.reference_labels import REFERENCE_GROUPS, export_reference_tasks, import_reference_annotations


def test_reference_groups_cover_every_part_class_once():
    groups = [(frame, cls, n) for frame, items in REFERENCE_GROUPS.items() for cls, n, _ in items]
    assert {cls for _, cls, _ in groups} == {"chassis", "fan_bay_empty", "psu_bay_empty", "fan", "psu", "cover"}
    assert dict((cls, n) for _, cls, n in groups) == pytest.approx(
        {"chassis": 1, "fan_bay_empty": 6, "psu_bay_empty": 2, "fan": 6, "psu": 2, "cover": 1})


import numpy as np

from training.reference_labels import assign_bay_identities, bay_state, transform_box


def test_transform_box_by_homography_takes_axis_aligned_bounds():
    shift = np.array([[1, 0, 10], [0, 1, -5], [0, 0, 1]], float)
    assert transform_box((0, 0, 100, 50), shift) == pytest.approx((10, -5, 110, 45))
    rot90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)  # (x, y) -> (-y, x)
    assert transform_box((0, 0, 100, 50), rot90) == pytest.approx((-50, 0, 0, 100))


def test_bay_identity_is_the_bay_that_changes_during_each_step():
    # change[step][bay]: how much each bay's crop changed across that step's clip
    change = np.array([[0.1, 0.9, 0.0],   # step fan_1 changed bay 1
                       [0.8, 0.1, 0.2],   # step fan_2 changed bay 0
                       [0.0, 0.3, 0.7]])  # step fan_3 changed bay 2
    assert assign_bay_identities(change) == [1, 0, 2]


SEQ = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_bay_state_from_timeline():
    # fan_bay for step action 3 (fan_3), frame in position 6 (fan_5 clip): fan_3 finished earlier
    assert bay_state(step_action=3, actions_by_position=SEQ, position=6) == "installed"
    assert bay_state(step_action=5, actions_by_position=SEQ, position=4) == "empty"
    assert bay_state(step_action=5, actions_by_position=SEQ, position=6) == "in_progress"


from training.reference_labels import appearance_diff, decide_bay


def test_appearance_diff_ignores_brightness_and_contrast():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (40, 60), dtype=np.uint8)
    brighter = np.clip(crop.astype(int) * 0.8 + 30, 0, 255).astype(np.uint8)
    other = rng.integers(0, 255, (40, 60), dtype=np.uint8)
    assert appearance_diff(crop, brighter) < 0.1 < appearance_diff(crop, other)


def decide(state, d_empty, d_full, occluded=False):
    return decide_bay(state, d_empty, d_full, separation=1.0, occluded=occluded, empty_cls="fan_bay_empty", full_cls="fan")


def test_in_progress_bay_goes_by_appearance():
    assert decide("in_progress", d_empty=0.2, d_full=0.9) == ("fan_bay_empty", None)
    assert decide("in_progress", d_empty=0.9, d_full=0.2) == ("fan", None)


def test_timeline_wins_but_disagreement_is_flagged():
    assert decide("installed", d_empty=0.9, d_full=0.1) == ("fan", None)
    assert decide("installed", d_empty=0.1, d_full=0.9) == ("fan", "fan: looks empty")
    assert decide("empty", d_empty=0.1, d_full=0.9) == ("fan_bay_empty", None)


def test_hidden_or_occluded_bays_are_left_unlabelled():
    assert decide("installed", d_empty=0.9, d_full=0.8) == (None, None)  # matches neither reference
    assert decide("installed", d_empty=0.9, d_full=0.1, occluded=True) == (None, None)


from training.reference_labels import replace_classes


def test_redraw_replaces_only_the_chosen_classes():
    existing = {"run": 4, "frames": {"start": {"t": 0.3, "boxes": [{"cls": "chassis", "xyxy": [0, 0, 9, 9]},
                                                                  {"cls": "psu_bay_empty", "xyxy": [1, 1, 2, 2]}]}}}
    new = {"start": [{"cls": "psu_bay_empty", "xyxy": [5, 5, 6, 6]}]}
    out = replace_classes(existing, new, {"psu_bay_empty"})
    assert out["frames"]["start"]["boxes"] == [{"cls": "chassis", "xyxy": [0, 0, 9, 9]}, {"cls": "psu_bay_empty", "xyxy": [5, 5, 6, 6]}]


from training.reference_labels import focus_region


def test_focus_region_finds_where_empty_and_installed_differ():
    empty = np.full((100, 200, 3), 120, np.uint8)
    full = empty.copy()
    full[30:70, 10:50] = 20  # only the left end changes (like a PSU handle)
    fx1, fy1, fx2, fy2 = focus_region(empty, full)
    assert fx1 <= 0.05 and fx2 <= 0.35 and fy1 <= 0.3 and fy2 >= 0.7


def test_focus_region_is_whole_box_when_everything_changes():
    rng = np.random.default_rng(1)
    empty = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    assert focus_region(empty, 255 - empty) == pytest.approx((0, 0, 1, 1), abs=0.1)


def test_low_separation_bays_still_label_when_visible():
    # PSU-like bay: looks differ only a little, and other recordings add noise
    cls, reason = decide_bay("empty", d_empty=0.35, d_full=0.7, separation=0.45, occluded=False,
                             empty_cls="psu_bay_empty", full_cls="psu")
    assert (cls, reason) == ("psu_bay_empty", None)


@pytest.fixture
def reference_tasks(tmp_path, monkeypatch):
    from training import reference_labels

    reference = tmp_path / "Install_4.json"
    original = {"run": 4, "frames": {
        "start": {"t": 0.3, "boxes": [{"cls": "chassis", "xyxy": [20.1, 10.4, 180.3, 90.6]},
                                     {"cls": "psu_bay_empty", "xyxy": [30, 20, 50, 40]}]},
        "filled": {"t": 74.8, "boxes": [{"cls": "fan", "xyxy": [20, 20, 60, 60]}]},
        "covered": {"t": 86.0, "boxes": [{"cls": "cover", "xyxy": [20, 10, 100, 90]}]},
    }}
    reference.write_text(json.dumps(original))
    monkeypatch.setattr(reference_labels, "load_index", lambda root: [SimpleNamespace(run=4, raw_path="video.mp4")])
    monkeypatch.setattr(reference_labels, "read_frame", lambda video, timestamp: np.zeros((100, 200, 3), np.uint8))
    tasks = tmp_path / "tasks.json"
    assert export_reference_tasks(4, tasks, tmp_path, reference) == 3
    return reference, tasks, original


def test_reference_labelstudio_round_trip_preserves_schema_and_pixels(reference_tasks):
    reference, tasks_path, original = reference_tasks
    tasks = json.loads(tasks_path.read_text())
    assert tasks[0]["data"]["image"] == "/data/local-files/?d=Install_4_start.png"
    assert (tasks[0]["data"]["width"], tasks[0]["data"]["height"]) == (200, 100)
    assert (tasks_path.parent / "Install_4_start.png").is_file()
    for task in tasks:
        task["annotations"] = [{"id": 1, "result": task["predictions"][0]["result"]}]
    tasks_path.write_text(json.dumps(tasks))
    assert import_reference_annotations(tasks_path, reference.parent) == [reference]
    assert json.loads(reference.read_text()) == original


def test_reference_partial_review_keeps_other_classes_and_frames(reference_tasks):
    reference, tasks_path, original = reference_tasks
    tasks = json.loads(tasks_path.read_text())
    start = tasks[0]
    start["data"]["classes"] = ["psu_bay_empty"]
    result = start["predictions"][0]["result"][1]
    result["value"]["x"] = 25
    start["annotations"] = [{"result": [result]}]
    tasks_path.write_text(json.dumps(tasks))
    import_reference_annotations(tasks_path, reference.parent)
    saved = json.loads(reference.read_text())
    assert saved["frames"]["start"]["boxes"] == [original["frames"]["start"]["boxes"][0],
                                                {"cls": "psu_bay_empty", "xyxy": [50, 20, 70, 40]}]
    assert saved["frames"]["filled"] == original["frames"]["filled"]


def test_first_reference_import_creates_readable_pixel_artifact(reference_tasks, tmp_path):
    _, tasks_path, original = reference_tasks
    tasks = json.loads(tasks_path.read_text())
    for task in tasks:
        task["annotations"] = [{"result": task["predictions"][0]["result"]}]
    tasks_path.write_text(json.dumps(tasks))
    output = tmp_path / "new-reference"
    paths = import_reference_annotations(tasks_path, output)
    assert paths == [output / "Install_4.json"]
    assert json.loads(paths[0].read_text()) == original
    assert paths[0].stat().st_mode & 0o777 == 0o644


def test_empty_review_clears_only_selected_classes(reference_tasks):
    reference, tasks_path, original = reference_tasks
    tasks = json.loads(tasks_path.read_text())
    tasks[0]["data"]["classes"] = ["psu_bay_empty"]
    tasks[0]["annotations"] = [{"result": []}]
    tasks_path.write_text(json.dumps(tasks))
    import_reference_annotations(tasks_path, reference.parent)
    assert json.loads(reference.read_text())["frames"]["start"]["boxes"] == [original["frames"]["start"]["boxes"][0]]


def test_unreviewed_and_cancelled_tasks_leave_reference_untouched(reference_tasks):
    reference, tasks_path, _ = reference_tasks
    before = reference.read_bytes()
    tasks = json.loads(tasks_path.read_text())
    tasks[0]["annotations"] = [{"was_cancelled": True, "result": []}]
    tasks_path.write_text(json.dumps(tasks))
    assert import_reference_annotations(tasks_path, reference.parent) == []
    assert reference.read_bytes() == before


@pytest.mark.parametrize("invalid", ["class", "dimensions", "bounds", "rotation", "time", "duplicate"])
def test_invalid_annotations_do_not_overwrite_reference(reference_tasks, invalid):
    reference, tasks_path, _ = reference_tasks
    before = reference.read_bytes()
    tasks = json.loads(tasks_path.read_text())
    for task in tasks:
        task["annotations"] = [{"result": task["predictions"][0]["result"]}]
    result = tasks[-1]["annotations"][0]["result"][0]
    if invalid == "class":
        result["value"]["rectanglelabels"] = ["not-a-reference-class"]
    elif invalid == "dimensions":
        result["original_width"] = 999
    elif invalid == "bounds":
        result["value"]["x"] = 101
    elif invalid == "rotation":
        result["value"]["rotation"] = 90
    elif invalid == "time":
        tasks[-1]["data"]["t"] = 999
    else:
        tasks.append(tasks[0])
    tasks_path.write_text(json.dumps(tasks))
    with pytest.raises(ValueError):
        import_reference_annotations(tasks_path, reference.parent)
    assert reference.read_bytes() == before


def test_selective_export_contains_only_requested_classes(reference_tasks):
    reference, tasks_path, _ = reference_tasks
    assert export_reference_tasks(4, tasks_path, reference.parent, reference, {"psu_bay_empty"}) == 1
    task = json.loads(tasks_path.read_text())[0]
    assert task["data"]["classes"] == ["psu_bay_empty"]
    assert len(task["predictions"][0]["result"]) == 1
