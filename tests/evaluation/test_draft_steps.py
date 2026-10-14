import importlib.util

import pytest

_spec = importlib.util.spec_from_file_location("draft_steps", "scripts/draft_steps.py")
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def present(*spans, end=40.0, step=0.5):
    """Someone at the wheel during the given (start, end) spans, sampled every `step` seconds."""
    times = [i * step for i in range(int(end / step) + 1)]
    return [(t, any(a <= t < b for a, b in spans)) for t in times]


def test_steps_run_from_one_completion_to_the_next_then_idle_to_the_end():
    rows = ds.cut([(4.0, "place_1"), (9.0, "place_2")], present((0, 10)), 12.0)
    assert rows == [("place_1", 0.0, 4.0), ("place_2", 4.0, 9.0), ("idle", 9.0, 12.0)]


def test_a_long_absence_inside_a_step_is_idle():
    rows = ds.cut([(5.0, "place_1"), (30.0, "place_2")], present((0, 6), (20, 31)), 32.0)
    assert rows == [("place_1", 0.0, 5.0), ("place_2", 5.0, 6.0), ("idle", 6.0, 20.0), ("place_2", 20.0, 30.0),
                    ("idle", 30.0, 32.0)]


def test_a_short_absence_is_not_idle():
    rows = ds.cut([(10.0, "place_1")], present((0, 4), (6, 11)), 10.0)
    assert rows == [("place_1", 0.0, 10.0)]


def test_install_rejects_gaps_overlaps_and_backwards_rows():
    ds.check_contiguous([("place_1", 0.0, 2.0), ("place_2", 2.0, 4.0)])
    for rows in ([("place_1", 0.0, 2.0), ("place_2", 2.5, 4.0)],   # gap
                 [("place_1", 0.0, 2.0), ("place_2", 1.5, 4.0)],   # overlap
                 [("place_1", 2.0, 1.0)]):                         # ends before it starts
        with pytest.raises(ValueError):
            ds.check_contiguous(rows)
