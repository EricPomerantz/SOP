"""Server-side stage rules: what each builder stage still needs, in the user's words."""

import pytest

from sopvision.builder import drafts
from sopvision.builder.status import checklist
from sopvision.settings import Settings

CLASSES = ["hub", "hole_empty", "bolt", "wrench", "hand"]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sops").mkdir()
    return Settings(sops_dir=tmp_path / "sops", datasets_dir=tmp_path / "datasets",
                    laya_registry=tmp_path / "registry.json")


def run(settings, sop_dir, run_id, split, steps=True):
    folder = settings.datasets_dir / sop_dir / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "video.mp4").write_bytes(b"")
    if steps:
        (folder / "steps.csv").write_text("step_id,start_s,end_s\n")
    splits = settings.datasets_dir / sop_dir / "splits.yaml"
    old = splits.read_text() if splits.exists() else ""
    splits.write_text(old + f"{split}:\n- {run_id}\n" if f"{split}:" not in old else old.replace(f"{split}:\n", f"{split}:\n- {run_id}\n"))


def stages_of(settings, sf):
    return {s["id"]: s for s in checklist(settings, sf, None, False)["stages"]}


def test_an_empty_draft_lists_every_stages_first_blocker(settings):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "")
    st = stages_of(settings, sf)
    assert [s for s in st] == ["detector", "recordings", "layout", "steps", "timelines", "train"]
    assert not st["detector"]["done"] and st["detector"]["blockers"][0] == "Import a detector model"
    assert st["recordings"]["blockers"] == ["Record at least one train run", "Record at least one val run",
                                            "Record at least one test run"]
    assert st["layout"]["blockers"] == ["Add at least one slot"]
    assert st["steps"]["blockers"] == ["Add at least one step"]
    assert st["timelines"]["blockers"] == ["Record some runs first"]
    assert st["train"]["blockers"] == ["Train a decision model"]
    assert checklist(settings, sf, None, False)["publish"]["ready"] is False


def test_detector_stage_needs_weights_an_anchor_and_a_part(settings, tmp_path):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "")
    (tmp_path / "best.pt").write_bytes(b"x")
    data = dict(sf.data, detector={"weights": str(tmp_path / "best.pt"), "classes": CLASSES})
    sf = drafts.write(sf, data)
    assert stages_of(settings, sf)["detector"]["blockers"] == ["Pick the anchor class", "Mark at least one class as a part"]
    data["layout"] = dict(data["layout"], anchor="hub")
    data["detector"] = dict(data["detector"], roles={"bolt": "part"})
    sf = drafts.write(sf, data)
    d = stages_of(settings, sf)["detector"]
    assert d["done"] and d["blockers"] == []


def test_recordings_and_timelines_count_runs_per_split(settings):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "")
    run(settings, "wheel", "r1", "train")
    run(settings, "wheel", "r2", "val", steps=False)
    st = stages_of(settings, sf)
    assert st["recordings"]["blockers"] == ["Record at least one test run"]
    assert st["timelines"]["blockers"] == ["1 run still needs a timeline"]
    run(settings, "wheel", "r3", "test")
    (settings.datasets_dir / "wheel" / "runs" / "r2" / "steps.csv").write_text("step_id,start_s,end_s\n")
    st = stages_of(settings, sf)
    assert st["recordings"]["done"] and st["timelines"]["done"]


def test_publish_blockers_mirror_the_publish_checks(settings, tmp_path):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "")
    pub = checklist(settings, sf, None, False)["publish"]
    assert pub["ready"] is False and any("step" in b for b in pub["blockers"])
    assert any("weights" in b.lower() or "detector" in b.lower() for b in pub["blockers"])
