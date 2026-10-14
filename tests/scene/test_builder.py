from sopvision.perception.detector import Detection
from sopvision.scene.builder import SceneBuilder
from sopvision.scene.types import HandView
from tests.helpers import CHASSIS, det_at


def chassis():
    return Detection("chassis", CHASSIS, 0.95, None)


def at(sop, slot, cls=None, **kw):
    spec = sop.layout.slots[slot]
    return det_at(cls or spec.empty_as, *spec.at, **kw)


def settle_empty(b, sop, slot="fan_bay_1"):
    b.update(0.0, [chassis(), at(sop, slot)])
    return b.update(0.6, [chassis(), at(sop, slot)])


def test_reading_must_settle_half_a_second(sop):
    b = SceneBuilder(sop)
    settle_empty(b, sop)
    fan1 = at(sop, "fan_bay_1", "fan")
    s, ev = b.update(1.0, [chassis(), fan1])
    assert s.slots["fan_bay_1"].state == "empty" and [e for e in ev if e.kind == "slot_changed"] == []
    s, ev = b.update(1.5, [chassis(), fan1])
    assert s.slots["fan_bay_1"].state == "installed"
    assert [e.describe() for e in ev if e.kind == "slot_changed"] == ["1.5s fan_bay_1 empty→installed"]


def test_hand_occludes_and_keeps_last(sop):
    b = SceneBuilder(sop)
    settle_empty(b, sop)
    s, _ = b.update(1.0, [chassis(), at(sop, "fan_bay_1", "hand", size=80)])
    view = s.slots["fan_bay_1"]
    assert (view.state, view.last, view.effective()) == ("occluded", "empty", "empty")


def test_hidden_without_hand_is_unknown(sop):
    b = SceneBuilder(sop)
    fan1 = at(sop, "fan_bay_1", "fan")
    b.update(0.0, [chassis(), fan1])
    b.update(0.6, [chassis(), fan1])
    s, _ = b.update(1.0, [chassis(), at(sop, "cover", "cover", size=300)])
    view = s.slots["fan_bay_1"]
    assert (view.state, view.last, view.effective()) == ("unknown", "installed", "installed")


def test_slot_without_empty_as_reads_empty(sop):
    b = SceneBuilder(sop)
    s, _ = b.update(0.0, [chassis()])
    assert s.slots["cover"].state == "unknown"
    s, _ = b.update(0.6, [chassis()])
    assert s.slots["cover"].state == "empty"


def test_hand_zone_events(sop):
    b = SceneBuilder(sop)
    u, v = sop.layout.slots["psu_bay_1"].at
    s, ev = b.update(0.0, [chassis(), det_at("hand", u + 0.02, v, size=60)])
    assert s.hands == [HandView("psu_bay_1")]
    assert [e.describe() for e in ev if e.kind == "hand_entered"] == ["0.0s hand entered psu_bay_1"]
    s, ev = b.update(0.1, [chassis()])
    assert s.hands == [] and [e.describe() for e in ev if e.kind == "hand_left"] == ["0.1s hand left psu_bay_1"]


def test_hand_away_from_slots_is_on_chassis_or_outside(sop):
    b = SceneBuilder(sop)
    s, _ = b.update(0.0, [chassis(), det_at("hand", 0.85, 0.5, size=40), det_at("hand", 1.6, 0.5, size=40)])
    assert s.hands == [HandView("chassis"), HandView("outside")]


def test_held_part(sop):
    b = SceneBuilder(sop)
    fan = det_at("fan", 0.8, 0.9, size=50)
    hand = det_at("hand", 0.84, 0.9, size=50)
    s, ev = b.update(0.0, [chassis(), fan, hand])
    assert s.held == ["fan"]
    assert [e.describe() for e in ev if e.kind == "held_changed"] == ["0.0s held: fan"]
    s, ev = b.update(0.1, [chassis(), hand])
    assert s.held == [] and [e.describe() for e in ev if e.kind == "held_changed"] == ["0.1s held: nothing"]


def test_chassis_lost_makes_slots_unknown(sop):
    b = SceneBuilder(sop)
    settle_empty(b, sop)
    events = []
    for t in (1.0, 2.0, 3.0, 3.6, 4.0):
        s, ev = b.update(t, [])
        events += [e.kind for e in ev]
    assert events.count("chassis_lost") == 1 and not s.chassis_visible
    assert (s.slots["fan_bay_1"].state, s.slots["fan_bay_1"].last) == ("unknown", "empty")
    _, ev = b.update(4.5, [chassis()])
    assert [e.kind for e in ev if e.kind.startswith("chassis")] == ["chassis_found"]


def test_version_increments_and_jev_dict_shape(sop):
    b = SceneBuilder(sop)
    versions = [b.update(t, [chassis(), at(sop, "fan_bay_1")])[0].version for t in (0.0, 0.6, 1.0)]
    assert versions == [1, 2, 3]
    s, _ = b.update(1.2, [chassis(), at(sop, "fan_bay_1"), at(sop, "fan_bay_2", "hand", size=80)])
    d = s.to_jev_dict()
    assert set(d) == {"version", "t", "chassis", "slots", "hands", "held"}
    assert d["chassis"] == {"visible": True, "conf": 0.95}
    assert d["slots"]["fan_bay_1"] == {"state": "empty", "conf": 0.9}
    assert d["hands"] == [{"near": "fan_bay_2"}]
    assert "chassis_box" not in d


def test_touched_when_a_hand_overlaps_the_part(sop):
    b = SceneBuilder(sop)
    fan1 = at(sop, "fan_bay_1", "fan")
    s, _ = b.update(0.0, [chassis(), fan1, det_at("hand", 0.9, 0.1, size=40)])
    assert s.slots["fan_bay_1"].touched is False
    u, v = sop.layout.slots["fan_bay_1"].at
    s, _ = b.update(0.1, [chassis(), fan1, det_at("hand", u + 0.03, v, size=40)])
    assert s.slots["fan_bay_1"].touched is True


def test_hand_on_a_corner_of_the_cover_counts_as_touching(sop):
    b = SceneBuilder(sop)
    cover = at(sop, "cover", "cover", w=600, h=400)  # big part: 0.6 x 0.8 of the chassis
    u, v = sop.layout.slots["cover"].at
    s, _ = b.update(0.0, [chassis(), cover, det_at("hand", u + 0.28, v + 0.35, size=40)])
    assert s.slots["cover"].touched is True


def test_occluded_slot_counts_as_touched(sop):
    b = SceneBuilder(sop)
    settle_empty(b, sop)
    s, _ = b.update(1.0, [chassis(), at(sop, "fan_bay_1", "hand", size=80)])
    assert s.slots["fan_bay_1"].state == "occluded" and s.slots["fan_bay_1"].touched is True


def renamed_anchor(sop, name="frame"):
    classes = [name if c == "chassis" else c for c in sop.detector.classes]
    return sop.model_copy(update={"detector": sop.detector.model_copy(update={"classes": classes}),
                                  "layout": sop.layout.model_copy(update={"anchor": name})})


def test_anchor_class_comes_from_the_layout(sop):
    b = SceneBuilder(renamed_anchor(sop))
    s, _ = b.update(0.0, [Detection("frame", CHASSIS, 0.9, None)])
    assert s.chassis_visible
    s, _ = SceneBuilder(renamed_anchor(sop)).update(0.0, [Detection("chassis", CHASSIS, 0.9, None)])
    assert not s.chassis_visible


def test_held_classes_come_from_the_slots(sop):
    slots = dict(sop.layout.slots)
    slots["fan_bay_1"] = slots["fan_bay_1"].model_copy(update={"filled_by": "battery"})
    other = sop.model_copy(update={
        "detector": sop.detector.model_copy(update={"classes": sop.detector.classes + ["battery"]}),
        "layout": sop.layout.model_copy(update={"slots": slots})})
    s, _ = SceneBuilder(other).update(0.0, [chassis(), det_at("battery", 0.8, 0.9, size=50), det_at("hand", 0.84, 0.9, size=50)])
    assert s.held == ["battery"]


# -- tools and torque (lug bolts SOP) -------------------------------------------------------------------------------
def hub():
    return Detection("hub", CHASSIS, 0.95, None)


def bolt(sop, slot):
    return at(sop, slot, "lug_bolt_gold")


def test_server_fan_state_has_no_tool_or_torque_keys(sop):
    b = SceneBuilder(sop)
    s, _ = settle_empty(b, sop)
    d = s.to_jev_dict()
    assert "tools" not in d and all("torqued" not in v for v in d["slots"].values())


def test_tool_is_located_at_a_slot_and_never_matched_as_a_part(lug_sop):
    b = SceneBuilder(lug_sop)
    b.update(0.0, [hub(), bolt(lug_sop, "hole_3")])
    s, ev = b.update(0.6, [hub(), bolt(lug_sop, "hole_3"), at(lug_sop, "hole_3", "torque_wrench", size=30)])
    assert [(t.tool, t.near) for t in s.tools] == [("torque_wrench", "hole_3")]
    assert s.slots["hole_3"].state == "installed" and s.held == []
    assert [e.describe() for e in ev if e.kind == "tool_entered"] == ["0.6s torque_wrench at hole_3"]
    assert s.to_jev_dict()["tools"] == [{"tool": "torque_wrench", "near": "hole_3"}]
    s, ev = b.update(1.0, [hub(), bolt(lug_sop, "hole_3")])
    assert s.tools == [] and [e.describe() for e in ev if e.kind == "tool_left"] == ["1.0s torque_wrench left hole_3"]


def test_torqued_flag_shows_until_the_part_leaves(lug_sop):
    b = SceneBuilder(lug_sop)
    b.update(0.0, [hub(), bolt(lug_sop, "hole_1")])
    b.update(0.6, [hub(), bolt(lug_sop, "hole_1")])
    b.mark_torqued("hole_1")
    s, _ = b.update(1.0, [hub(), bolt(lug_sop, "hole_1")])
    assert s.slots["hole_1"].torqued and s.to_jev_dict()["slots"]["hole_1"]["torqued"] is True
    b.update(1.2, [hub(), at(lug_sop, "hole_1")])  # bolt gone: hole seen empty
    s, _ = b.update(1.8, [hub(), at(lug_sop, "hole_1")])
    assert s.slots["hole_1"].state == "installed" and s.slots["hole_1"].torqued  # not yet: layout.removal_seconds
    s, _ = b.update(1.3 + lug_sop.layout.removal_seconds, [hub(), at(lug_sop, "hole_1")])
    assert s.slots["hole_1"].state == "empty" and not s.slots["hole_1"].torqued


def test_a_misread_shorter_than_the_removal_wait_keeps_the_bolt_and_its_torque(lug_sop):
    b = SceneBuilder(lug_sop)
    b.update(0.0, [hub(), bolt(lug_sop, "hole_2")])
    b.update(0.6, [hub(), bolt(lug_sop, "hole_2")])
    b.mark_torqued("hole_2")
    for t in (1.0, 3.0, 6.0):  # the detector calls the gold bolt an empty hole for 6 s
        s, ev = b.update(t, [hub(), at(lug_sop, "hole_2")])
        assert s.slots["hole_2"].state == "installed" and not [e for e in ev if e.kind == "slot_changed"]
    s, _ = b.update(6.5, [hub(), bolt(lug_sop, "hole_2")])
    s, _ = b.update(7.5 + lug_sop.layout.removal_seconds, [hub(), bolt(lug_sop, "hole_2")])
    assert s.slots["hole_2"].state == "installed" and s.slots["hole_2"].torqued


def test_without_removal_seconds_a_part_leaves_after_the_usual_settle(sop):
    assert sop.layout.removal_seconds is None
    b = SceneBuilder(sop)
    fan = at(sop, "fan_bay_1", "fan")
    b.update(0.0, [chassis(), fan])
    b.update(0.6, [chassis(), fan])
    b.update(1.0, [chassis(), at(sop, "fan_bay_1")])
    s, _ = b.update(1.5, [chassis(), at(sop, "fan_bay_1")])
    assert s.slots["fan_bay_1"].state == "empty"


def test_any_listed_bolt_colour_fills_a_hole(lug_sop):
    b = SceneBuilder(lug_sop)
    dets = [hub(), at(lug_sop, "hole_1", "lug_bolt_red"), at(lug_sop, "hole_2", "lug_bolt_gold"), at(lug_sop, "hole_3")]
    b.update(0.0, dets)
    s, _ = b.update(0.6, dets)
    assert (s.slots["hole_1"].state, s.slots["hole_2"].state, s.slots["hole_3"].state) == ("installed", "installed", "empty")


def test_fast_anchor_follow_keeps_holes_on_a_shaking_camera(lug_sop):
    from tests.helpers import det_at

    def shaken(shift):  # the whole image moves right by `shift` hub-box widths
        box = (CHASSIS[0] + shift * 1000, CHASSIS[1], CHASSIS[2] + shift * 1000, CHASSIS[3])
        spec = lug_sop.layout.slots["hole_1"]
        return [Detection("hub", box, 0.95, None), det_at("lug_bolt_gold", *spec.at, chassis=box)]

    def first_shaken_frame(sop):
        b = SceneBuilder(sop)
        for t in (0.0, 0.3, 0.6):
            b.update(t, shaken(0.0))
        s, _ = b.update(0.64, shaken(0.25))
        return s.slots["hole_1"].state

    slow = lug_sop.model_copy(update={"layout": lug_sop.layout.model_copy(update={"anchor_follow": 0.1})})
    assert lug_sop.layout.anchor_follow == 0.7
    assert first_shaken_frame(lug_sop) == "installed"  # the hole grid moved with the image
    assert first_shaken_frame(slow) != "installed"  # the old smoothing lagged: the bolt fell outside its hole


def test_server_fan_keeps_its_smoothing(sop):
    assert sop.layout.anchor_follow == 0.1


def test_a_slot_remembers_the_part_it_holds_until_it_empties(lug_sop):
    b, box = SceneBuilder(lug_sop), (0.0, 0.0, 100.0, 100.0)
    hub = Detection("hub", box, 0.95, None)
    at = lug_sop.layout.slots["hole_1"].at
    bolt = lambda cls: Detection(cls, (at[0] * 100 - 3, at[1] * 100 - 3, at[0] * 100 + 3, at[1] * 100 + 3), 0.9, None)
    for i in range(10):
        s, _ = b.update(i * 0.1, [hub, bolt("lug_bolt_gold" if i != 4 else "lug_bolt_red")])  # one misread frame
    assert (s.slots["hole_1"].part, s.slots["hole_1"].box is not None) == ("lug_bolt_gold", True)
    s, _ = b.update(1.0, [hub])  # nothing seen there and no hand: unknown, still holding the gold bolt
    assert s.slots["hole_1"].part == "lug_bolt_gold"
