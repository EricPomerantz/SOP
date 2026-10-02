import numpy as np

from training.review_sheets import draw_labels

CLASSES = ["chassis", "fan", "fan_bay_empty", "psu", "psu_bay_empty", "cover", "hand"]


def test_draw_labels_outlines_each_box_in_its_class_colour():
    img = np.zeros((100, 200, 3), np.uint8)
    out = draw_labels(img, ["1 0.5 0.5 0.2 0.4"], CLASSES)  # fan box x 80..120, y 30..70
    assert tuple(out[30, 100]) != (0, 0, 0) and tuple(out[50, 100]) == (0, 0, 0)  # edge drawn, inside untouched
    assert img.sum() == 0  # input not modified
