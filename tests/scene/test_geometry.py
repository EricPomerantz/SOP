import pytest

from sopvision.scene.geometry import ChassisAnchor, box_distance_norm, match_slots, to_norm
from tests.helpers import CHASSIS, det_at, rotate


def test_first_box_initialises_anchor():
    s = ChassisAnchor().update(0.0, CHASSIS)
    assert s.box == pytest.approx(CHASSIS) and s.visible and not s.lost_event and not s.found_event


def test_single_outlier_moves_anchor_little():
    a = ChassisAnchor()
    a.update(0.0, CHASSIS)
    s = a.update(0.033, (400.0, 100.0, 1400.0, 600.0))  # one-frame 30 % jump
    assert abs(s.box[0] - 100.0) <= 0.031 * 1000


def test_lost_fires_once_after_three_seconds():
    a = ChassisAnchor()
    a.update(0.0, CHASSIS)
    assert not a.update(2.9, None).lost_event
    assert a.update(3.0, None).lost_event and not a.update(3.5, None).lost_event
    assert not a.update(3.6, None).visible
    assert a.update(4.0, CHASSIS).found_event


def test_to_norm_corners():
    assert to_norm((100, 100), CHASSIS) == (0.0, 0.0) and to_norm((1100, 600), CHASSIS) == (1.0, 1.0)


def test_box_distance_norm_is_zero_inside_and_scaled_by_chassis_size():
    box = (500.0, 300.0, 600.0, 400.0)  # normalised x 0.4..0.5, y 0.4..0.6
    assert box_distance_norm((0.45, 0.5), box, CHASSIS) == 0.0
    assert box_distance_norm((0.6, 0.5), box, CHASSIS) == pytest.approx(0.1)
    assert box_distance_norm((0.45, 0.8), box, CHASSIS) == pytest.approx(0.2)


def test_matching_survives_three_degree_rotation(sop):
    dets = [det_at("fan", *rotate(sop.layout.slots[f"fan_bay_{i}"].at, deg=3)) for i in range(1, 7)]
    assert match_slots(dets, sop, CHASSIS) == {f"fan_bay_{i}": d for i, d in zip(range(1, 7), dets)}


def test_out_of_radius_and_competition(sop):
    u, v = sop.layout.slots["fan_bay_1"].at
    far = det_at("fan", u + 0.2, v)
    assert match_slots([far], sop, CHASSIS) == {}
    near, nearer = det_at("fan", u + 0.03, v), det_at("fan_bay_empty", u + 0.01, v)
    assert match_slots([near, nearer], sop, CHASSIS) == {"fan_bay_1": nearer}


def test_class_must_be_compatible(sop):
    assert match_slots([det_at("psu", *sop.layout.slots["fan_bay_1"].at)], sop, CHASSIS) == {}
