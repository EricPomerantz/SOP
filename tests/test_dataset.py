from pathlib import Path

import pytest

from sopvision.dataset import load_index, parse_clip_name
from sopvision.settings import Settings


def make_tree(root: Path, runs: dict[tuple[str, int], list[tuple[int, int, float]]], raw: dict[int, float]):
    """runs: (split_dir, run) -> [(action, position, duration)]. Returns a probe for the fake files."""
    durations: dict[str, float] = {}
    for (split_dir, run), clips in runs.items():
        d = root / split_dir / f"Install_{run}"
        d.mkdir(parents=True)
        for action, position, seconds in clips:
            name = f"{action:02d}_Install_{run}_1_{position}.mp4"
            (d / name).touch()
            durations[name] = seconds
    (root / "raw").mkdir()
    for run, seconds in raw.items():
        (root / "raw" / f"Install_{run}.MP4").touch()
        durations[f"Install_{run}.MP4"] = seconds
    return lambda p: durations[Path(p).name]


def test_parse_clip_name():
    assert parse_clip_name("10_Install_1_2_11.mp4") == (10, 1, 2, 11)
    with pytest.raises(ValueError):
        parse_clip_name("Install_1.MP4")


def test_segments_are_cumulative_and_timing_flag(tmp_path):
    probe = make_tree(tmp_path, {("train", 3): [(10, 1, 5.0), (1, 2, 10.0), (2, 3, 4.0)]}, {3: 21.2})
    run = load_index(tmp_path, probe=probe)[0]
    assert [(s.step_id, s.start, s.end) for s in run.segments] == [("off_sop", 0, 5), ("fan_1", 5, 15), ("fan_2", 15, 19)]
    assert run.timing_valid is False
    assert run.actions_by_position == [10, 1, 2]


def test_timing_valid_within_a_tenth_of_a_second(tmp_path):
    probe = make_tree(tmp_path, {("train", 4): [(1, 1, 5.0), (2, 2, 5.0)]}, {4: 10.05})
    assert load_index(tmp_path, probe=probe)[0].timing_valid is True


def test_split_assignment(tmp_path):
    clips = [(1, 1, 1.0)]
    probe = make_tree(tmp_path, {("train", 4): clips, ("train", 11): clips, ("test", 12): clips}, {4: 1.0, 11: 1.0, 12: 1.0})
    assert [(r.run, r.split) for r in load_index(tmp_path, probe=probe)] == [(4, "train"), (11, "val"), (12, "test")]


@pytest.mark.dataset
def test_real_dataset_index():
    root = Settings().sop_data_root
    if not root.is_dir():
        pytest.skip("server_fan dataset not present")
    idx = {r.run: r for r in load_index(root)}
    assert sorted(idx) == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert [r for r in idx if not idx[r].timing_valid] == [3, 5]
    assert idx[12].actions_by_position == [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
