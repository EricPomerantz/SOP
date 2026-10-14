import pytest

from sopvision.config import ConfigError
from sopvision.recordings import follows_sop, load_recordings
from sopvision.settings import Settings
from tests.test_dataset import make_tree


def test_clips_use_the_sop_files_action_mapping(sop, tmp_path):
    probe = make_tree(tmp_path, {("train", 4): [(10, 1, 5.0), (1, 2, 10.0), (2, 3, 4.0)],
                                 ("test", 12): [(1, 1, 6.0)]}, {4: 19.0, 12: 6.0})
    recs = load_recordings(sop, Settings(sop_data_root=tmp_path), probe=probe)
    r4, r12 = recs
    assert (r4.id, r4.split, r4.source_id, r4.valid, r4.torque_log) == ("Install_4", "train", "train:Install_4", True, None)
    assert [(s.step_id, s.start, s.end) for s in r4.segments] == [("off_sop", 0, 5), ("fan_1", 5, 15), ("fan_2", 15, 19)]
    assert r12.source_id == "test:Install_12"


def lug_dataset(root, runs):
    (root / "lug_bolts" / "runs").mkdir(parents=True)
    (root / "lug_bolts" / "splits.yaml").write_text("train: [r01, r02]\nval: [r03]\ntest: [r04]\n", encoding="utf-8")
    for run_id, (steps, torque) in runs.items():
        d = root / "lug_bolts" / "runs" / run_id
        d.mkdir()
        (d / "video.mp4").touch()
        if steps is not None:
            (d / "steps.csv").write_text("step_id,start_s,end_s\n" + steps, encoding="utf-8")
        if torque is not None:
            (d / "torque.csv").write_text("time_s,ok,torque_nm\n" + torque, encoding="utf-8")


def test_timeline_dataset(lug_sop, tmp_path):
    lug_dataset(tmp_path, {"r01": ("idle,0,3\nplace_2,3,6\nplace_1,6,9\n", "12.5,1,110\n"),
                           "r02": (None, None), "r03": ("place_1,0,2\n", None), "r04": ("place_1,0,2\n", None)})
    recs = {r.id: r for r in load_recordings(lug_sop, Settings(datasets_dir=tmp_path))}
    assert [(r.split, r.valid) for r in recs.values()] == [("train", True), ("train", False), ("val", True), ("test", True)]
    r01 = recs["r01"]
    assert [(s.step_id, s.start, s.end) for s in r01.segments] == [("off_sop", 0, 3), ("place_2", 3, 6), ("place_1", 6, 9)]
    assert r01.torque_log.name == "torque.csv" and r01.video.name == "video.mp4"


def test_timeline_rejects_unknown_steps_and_missing_splits(lug_sop, tmp_path):
    with pytest.raises(ConfigError, match="splits.yaml"):
        load_recordings(lug_sop, Settings(datasets_dir=tmp_path))
    lug_dataset(tmp_path, {"r01": ("fan_1,0,2\n", None), "r02": (None, None), "r03": (None, None), "r04": (None, None)})
    with pytest.raises(ConfigError, match="unknown step 'fan_1'"):
        load_recordings(lug_sop, Settings(datasets_dir=tmp_path))


def test_a_clean_run_does_every_step_once_in_order(lug_sop):
    places = ["place_2", "place_1", "place_3", "place_5", "place_4"]
    torques = ["torque_1", "torque_3", "torque_5", "torque_2", "torque_4"]
    assert follows_sop(["idle", *places, "idle", *torques, "idle"], lug_sop)  # places in any order, idle allowed
    assert not follows_sop([*places, "torque_3", "torque_1", *torques[2:]], lug_sop)  # star order broken
    assert not follows_sop([*places, *torques[:2], "off_sop", *torques[2:]], lug_sop)  # a retorque, a bolt out
    assert not follows_sop([*places, *torques[:2], "torque_1", *torques[2:]], lug_sop)  # a step done twice
    assert not follows_sop([*places, *torques[:4]], lug_sop)  # a step missing
    assert not follows_sop([*places[:2], "torque_1", *places[2:], *torques[1:]], lug_sop)  # torque before all placed
