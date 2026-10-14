from training.sample_recordings import plan


def test_plan_grid_plus_press_frames():
    times = plan(10.0, presses=[3.0, 6.6], every=4.0)
    assert times == [(0.5, "grid"), (2.7, "torque"), (4.5, "grid"), (6.3, "torque"), (8.5, "grid")]


def test_plan_skips_press_frames_next_to_grid_frames():
    assert plan(10.0, presses=[1.0], every=4.0) == [(0.5, "grid"), (4.5, "grid"), (8.5, "grid")]
