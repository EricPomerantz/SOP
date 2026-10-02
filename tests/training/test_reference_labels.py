import pytest

from training.reference_labels import REFERENCE_GROUPS, display_to_image


def test_display_boxes_scale_back_to_image_pixels():
    # selectROIs returns (x, y, w, h) in the downscaled window; we store xyxy in full-resolution pixels
    assert display_to_image([(100, 50, 200, 100)], scale=0.5) == [(200.0, 100.0, 600.0, 300.0)]


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
