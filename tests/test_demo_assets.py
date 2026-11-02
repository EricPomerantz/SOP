import importlib.util
from fnmatch import fnmatch

_spec = importlib.util.spec_from_file_location("demo_assets", "scripts/demo_assets.py")
da = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(da)

SPLITS = """\
train:
- correct2
val:
- correctval1
test:
- testcorrect1
- dev4-idle-offsop
"""


def selected(path, patterns):
    return any(fnmatch(path, p) for p in patterns)


def test_test_runs_come_from_the_splits_file():
    assert da.test_runs(SPLITS) == ["testcorrect1", "dev4-idle-offsop"]


def test_core_fetches_what_the_app_needs_to_start_and_the_lug_bolt_test_runs_only():
    patterns = da.fetch_patterns("core", ["testcorrect1"])
    for path in ["models/server_fan/best.pt", "models/lug_bolts/best.pt",
                 "sop-sample-training-data/server_fan/raw/Install_12.MP4",
                 "sop-sample-training-data/server_fan/test/Install_12/x.mp4",
                 "data/scenarios/fan_removed.mp4", "data/training/models/registry.json",
                 "data/training/models/laya-sop-v2-e12/model.safetensors",
                 "data/training/models/laya-sop-v2-e12/tokenizer/tokenizer.json",
                 "datasets/lug_bolts/splits.yaml", "datasets/lug_bolts/runs/testcorrect1/video.mp4"]:
        assert selected(path, patterns), path
    for path in ["datasets/lug_bolts/runs/correct2/video.mp4",
                 "sop-sample-training-data/server_fan/augmented/a.mp4"]:
        assert not selected(path, patterns), path


def test_all_adds_every_lug_bolt_run():
    assert selected("datasets/lug_bolts/runs/correct2/video.mp4", da.fetch_patterns("all", []))


def test_the_published_set_leaves_out_the_training_resume_state():
    assert selected("datasets/lug_bolts/runs/correct2/video.mp4", da.PUBLISH)
    assert not selected("data/training/models/laya-sop-v2-e12/checkpoint_latest/model.safetensors", da.PUBLISH)
    assert selected("data/training/models/laya-sop-v2-e12/checkpoint_latest/model.safetensors", da.IGNORE)
