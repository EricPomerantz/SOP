import cv2
import numpy as np

from training.prelabel_bolts import HOLES, label_frame


def _frame(bolts: dict[str, tuple[int, int, int]]) -> np.ndarray:
    image = np.full((1080, 1920, 3), 30, np.uint8)
    for hole, bgr in bolts.items():
        cv2.circle(image, HOLES[hole], 22, bgr, -1)
    return image


def test_colours_become_bolts_and_bare_holes_empty():
    had = set()
    boxes, _ = label_frame(_frame({"hole_1": (255, 60, 0), "hole_2": (0, 0, 230)}), (0.0, 0.0), [], had)
    classes = sorted(b.cls for b in boxes)
    assert classes == ["hole_empty", "hole_empty", "hole_empty", "hub", "lug_bolt_blue", "lug_bolt_red"]
    assert had == {"hole_1", "hole_2"}


def test_a_hole_that_had_a_bolt_and_lost_its_colour_is_the_wrench():
    had = {"hole_1"}
    boxes, notes = label_frame(_frame({}), (0.0, 0.0), [], had)
    assert [b.cls for b in boxes].count("torque_wrench") == 1
    assert notes and "hole_1" in notes[0]


def test_holes_under_a_hand_get_no_box():
    x, y = HOLES["hole_3"]
    hand = {"cls": "hand", "xyxy": [x - 50, y - 50, x + 50, y + 50], "conf": 0.5}
    boxes, notes = label_frame(_frame({}), (0.0, 0.0), [hand], set())
    assert [b.cls for b in boxes].count("hole_empty") == 4
    assert "hole_3 covered" in notes
