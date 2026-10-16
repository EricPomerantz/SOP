import pytest

from sopvision.builder.timelines import check_rows, cut

STEPS = {"place_1", "torque_1"}


def test_a_saved_timeline_must_be_contiguous_known_and_cover_the_video():
    good = [("place_1", 0.0, 4.0), ("idle", 4.0, 9.0), ("torque_1", 9.0, 12.0)]
    check_rows(good, STEPS, 12.2)
    for rows, message in (([("place_1", 0.0, 4.0), ("torque_1", 4.5, 12.0)], "row before ends"),
                          ([("place_1", 0.0, 4.0), ("torque_1", 3.5, 12.0)], "row before ends"),
                          ([("place_9", 0.0, 12.0)], "Unknown step"),
                          ([("place_1", 1.0, 12.0)], "not 0"),
                          ([("place_1", 0.0, 6.0)], "video is 12.00 s"),
                          ([("place_1", 0.0, 0.0)], "ends before"),
                          ([], "empty")):
        with pytest.raises(ValueError, match=message):
            check_rows(rows, STEPS, 12.0)


def test_drafts_end_steps_at_completions_and_split_out_idle():
    present = [(t / 2, not 10 <= t / 2 < 18) for t in range(60)]
    rows = cut([(4.0, "place_1"), (25.0, "torque_1")], present, 30.0)
    assert rows == [("place_1", 0.0, 4.0), ("torque_1", 4.0, 10.0), ("idle", 10.0, 18.0), ("torque_1", 18.0, 25.0),
                    ("idle", 25.0, 30.0)]


def test_previews_are_written_for_runs_without_one(tmp_path):
    import cv2
    import numpy as np

    from sopvision.builder.timelines import write_previews
    from sopvision.settings import Settings

    sop = tmp_path / "wheel.yaml"
    sop.write_text("sop: wheel\ndataset: {dir: wheel}\n")
    for run in ("r1", "r2"):
        folder = tmp_path / "datasets/wheel/runs" / run
        folder.mkdir(parents=True)
        w = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(10):
            w.write(np.zeros((48, 64, 3), np.uint8))
        w.release()
    assert write_previews(sop, Settings(datasets_dir=tmp_path / "datasets"), ["r2"]) == ["r2"]
    assert (tmp_path / "datasets/wheel/runs/r2/preview.webm").stat().st_size > 0
    assert not (tmp_path / "datasets/wheel/runs/r1/preview.webm").exists()
