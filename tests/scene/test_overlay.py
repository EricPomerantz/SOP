import numpy as np

from sopvision.scene.overlay import OverlayStats, draw_overlay
from sopvision.scene.types import HandView, SceneState, SlotView

CHASSIS_PX = (200.0, 100.0, 1700.0, 1000.0)


def scene_with(states: dict[str, str]):
    slots = {k: SlotView(v, 0.9, v) if v in ("installed", "empty") else SlotView(v, last="empty") for k, v in states.items()}
    return SceneState(3, 1.0, True, 0.95, slots, [HandView("fan_bay_2")], [], CHASSIS_PX)


def slot_pixel(u, v):
    x1, y1, x2, y2 = CHASSIS_PX
    return int(x1 + u * (x2 - x1)), int(y1 + v * (y2 - y1))


def test_slots_are_outlined_in_their_state_colour_and_the_part_stays_visible(sop):
    slots = {"fan_bay_1": SlotView("installed", 0.9, "installed"), "fan_bay_2": SlotView("empty", 0.9, "empty"),
             "fan_bay_3": SlotView("occluded")}
    s = SceneState(3, 1.0, True, 0.95, slots, [], [], CHASSIS_PX)
    img = draw_overlay(np.zeros((1080, 1920, 3), np.uint8), s, [], sop, OverlayStats(30, 6, 140))
    ring = 24  # 12 px per 540 rows of frame, with no detection box to size it
    for slot, colour in (("fan_bay_1", (0, 200, 0)), ("fan_bay_2", (160, 160, 160)), ("fan_bay_3", (0, 191, 255))):
        x, y = slot_pixel(*sop.layout.slots[slot].at)
        assert not img[y, x].any()  # not filled: the part under it is still visible
        assert tuple(int(c) for c in img[y, x + ring]) == colour


def test_a_part_named_for_a_colour_is_ringed_in_that_colour(sop):
    from sopvision.scene.overlay import PART_COLOURS, part_colour

    assert part_colour("lug_bolt_gold") == ("gold", PART_COLOURS["gold"])
    assert part_colour("fan") == (None, (0, 200, 0))
    box = (900.0, 500.0, 940.0, 540.0)  # a 40 px bolt: ring radius 20 + 6
    s = SceneState(3, 1.0, True, 0.95, {"fan_bay_1": SlotView("installed", 0.9, "installed", part="lug_bolt_red",
                                                                box=box)}, [], [], CHASSIS_PX)
    img = draw_overlay(np.zeros((1080, 1920, 3), np.uint8), s, [], sop, OverlayStats(30, 6, 140))
    assert tuple(int(c) for c in img[520, 920 + 26]) == PART_COLOURS["red"]


def test_overlay_leaves_input_untouched_and_handles_small_frames(sop):
    base = np.zeros((480, 640, 3), np.uint8)
    out = draw_overlay(base, scene_with({"fan_bay_1": "installed"}), [], sop, OverlayStats(30, 6, None))
    assert out.shape == base.shape and not base.any() and out.any()


def test_no_chassis_draws_banner_only(sop):
    s = SceneState(1, 0.0, False, None, {}, [], [], None)
    out = draw_overlay(np.zeros((360, 640, 3), np.uint8), s, [], sop, OverlayStats(25, 5, None))
    assert out.any()
