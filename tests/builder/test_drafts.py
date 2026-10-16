from pathlib import Path

import pytest
import yaml

from sopvision.builder import detector as det
from sopvision.builder import drafts
from sopvision.builder.drafts import DraftError
from sopvision.settings import Settings

CLASSES = ["hub", "hole_empty", "bolt_red", "bolt_blue", "wrench", "hand"]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sops").mkdir()
    return Settings(sops_dir=tmp_path / "sops", datasets_dir=tmp_path / "datasets",
                    laya_registry=tmp_path / "registry.json")


def fake_loader(task="detect", names=CLASSES):
    return lambda path: (task, list(names))


def filled(sf):
    """The draft as the builder pages would fill it in."""
    data = dict(sf.data)
    data["layout"] = {"anchor": "hub", "match_radius": 0.15, "tools": ["wrench"],
                      "slots": {"hole_1": {"at": [0.6, 0.15], "filled_by": ["bolt_red", "bolt_blue"],
                                           "empty_as": "hole_empty"},
                                "hole_2": {"at": [0.86, 0.5], "filled_by": ["bolt_red", "bolt_blue"],
                                           "empty_as": "hole_empty"}}}
    data.update(drafts.generate_steps(data))
    return drafts.write(sf, data)


def test_a_new_draft_is_saved_incomplete_and_reports_what_is_missing(settings):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "Bolts on a wheel")
    assert sf.path == settings.sops_dir / "drafts" / "wheel.yaml" and sf.draft
    assert any("anchor" in e for e in drafts.validate(sf.data))
    with pytest.raises(DraftError, match="already exists"):
        drafts.new_draft(settings, "wheel", "", "")
    with pytest.raises(DraftError, match="lower-case"):
        drafts.new_draft(settings, "../x", "", "")


def test_importing_a_detector_sets_the_classes_and_keeps_roles_that_still_exist(settings, tmp_path):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "")
    upload = tmp_path / "up.pt"
    upload.write_bytes(b"weights")
    sf, warnings = det.import_upload(sf, upload, loader=fake_loader())
    assert sf.data["detector"]["classes"] == CLASSES and Path(sf.data["detector"]["weights"]).exists()
    assert str(sf.data["detector"]["weights"]).startswith("models/wheel/detector-") and warnings == []
    data = dict(sf.data, layout=dict(sf.data["layout"], anchor="hub", tools=["wrench"]))
    data["detector"] = dict(data["detector"], roles={"bolt_red": "part", "hole_empty": "empty"})
    sf = drafts.write(sf, data)
    sf, warnings = det.apply_detector(sf, Path("models/x.pt"), ["hub", "bolt_red"])
    assert sf.data["detector"]["roles"] == {"bolt_red": "part"} and sf.data["layout"]["tools"] == []
    assert sf.data["layout"]["anchor"] == "hub" and "hand" in warnings[0]


def test_only_detection_models_are_accepted(settings, tmp_path):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "")
    (tmp_path / "seg.pt").write_bytes(b"x")
    with pytest.raises(DraftError, match="segment model"):
        det.import_upload(sf, tmp_path / "seg.pt", loader=fake_loader("segment"))
    with pytest.raises(DraftError, match="pick a model"):
        det.import_existing(sf, "/etc/passwd", loader=fake_loader())


def test_generated_steps_place_in_any_order_then_torque_in_sequence(settings, tmp_path):
    sf = filled(drafts.new_draft(settings, "wheel", "Wheel", ""))
    assert [s["id"] for s in sf.data["steps"]] == ["place_hole_1", "place_hole_2", "torque_hole_1", "torque_hole_2"]
    assert sf.data["groups"] == [["place_hole_1", "place_hole_2"], {"sequence": ["torque_hole_1", "torque_hole_2"]}]
    no_tool = dict(sf.data, layout=dict(sf.data["layout"], tools=[]))
    assert drafts.generate_steps(no_tool)["order"] == "strict"


def test_publish_moves_a_valid_draft_and_editing_bumps_the_version(settings, tmp_path):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "")
    with pytest.raises(DraftError, match="Fix these first"):
        drafts.publish(settings, "wheel")
    sf = filled(sf)
    weights = Path("models/wheel/best.pt")
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    data = dict(sf.data, detector={"weights": str(weights), "classes": CLASSES})
    drafts.write(sf, data)
    path = drafts.publish(settings, "wheel")
    assert path == settings.sops_dir / "wheel.yaml" and not (settings.sops_dir / "drafts" / "wheel.yaml").exists()
    assert drafts.validate(yaml.safe_load(path.read_text())) == []

    draft = drafts.start_edit(settings, "wheel")
    assert draft.path == settings.sops_dir / "drafts" / "wheel.yaml"
    drafts.write(draft, dict(draft.data, name="Wheel v2"))
    drafts.publish(settings, "wheel")
    published = yaml.safe_load(path.read_text())
    assert (published["name"], published["version"]) == ("Wheel v2", 2)
    assert (settings.sops_dir / ".history" / "wheel.v1.yaml").exists()


def test_a_draft_can_not_change_its_id_and_published_files_are_not_written(settings):
    sf = drafts.new_draft(settings, "wheel", "Wheel", "")
    with pytest.raises(DraftError, match="can't change"):
        drafts.write(sf, dict(sf.data, sop="other"))
    with pytest.raises(DraftError, match="published"):
        drafts.write(drafts.SopFile("wheel", sf.path, False, sf.data), sf.data)
