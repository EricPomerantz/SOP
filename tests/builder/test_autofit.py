import numpy as np

from sopvision.builder.autofit import FrameObs, fit

HOLES = {"hole_1": (0.6, 0.155), "hole_2": (0.86, 0.5), "hole_3": (0.62, 0.823), "hole_4": (0.205, 0.731),
         "hole_5": (0.197, 0.307)}


def frames(n=200, move=True, rotate=False, seed=1):
    """Five holes on an anchor that moves (and optionally turns 90° half the time), with detection jitter."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x0, y0 = (rng.uniform(0, 800), rng.uniform(0, 400)) if move else (100, 100)
        w, h = (240, 240) if not (rotate and i % 2) else (300, 180)
        pts = []
        for j, (u, v) in enumerate(HOLES.values()):
            if rotate and i % 2:
                u, v = v, 1 - u
            cls = "bolt" if (i + j) % 3 else "hole_empty"
            pts.append((cls, u + rng.normal(0, 0.01), v + rng.normal(0, 0.01)))
        pts.append(("wrench", 0.5, 0.5))  # a tool: not a slot
        out.append(FrameObs("r1", i, (x0, y0, x0 + w, y0 + h), pts))
    out.append(FrameObs("r1", n, (0, 0, 120, 120), [("bolt", 0.9, 0.9)]))  # a half-hidden anchor: dropped
    return out


def test_slots_are_found_where_the_parts_sit_even_on_a_moving_anchor():
    result = fit(frames(), {"bolt"}, {"hole_empty"})
    assert len(result.slots) == 5 and result.frames_dropped == 1 and result.warnings == []
    for slot in result.slots:
        nearest = min(HOLES.values(), key=lambda h: np.hypot(h[0] - slot["at"][0], h[1] - slot["at"][1]))
        assert np.hypot(nearest[0] - slot["at"][0], nearest[1] - slot["at"][1]) < 0.01
        assert slot["filled_by"] == ["bolt"] and slot["empty_as"] == "hole_empty"
        assert 0.05 <= slot["radius"] <= 0.4 * slot["nearest"]
    assert [s["name"] for s in result.slots] == ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5"]
    assert result.slots[0]["at"][1] < 0.2  # reading order: the top hole first


def test_a_rotating_anchor_is_warned_about():
    result = fit(frames(rotate=True), {"bolt"}, {"hole_empty"})
    assert any("rotates" in w for w in result.warnings)


def test_no_anchor_or_no_parts_says_what_to_check():
    assert "anchor" in fit([], {"bolt"}, set()).warnings[0]
    assert "roles" in fit(frames(5), {"nut"}, set()).warnings[0]
