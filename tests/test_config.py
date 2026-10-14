import re
from pathlib import Path

import pytest
import yaml

from sopvision.config import ConfigError, load_sop

SERVER_FAN = Path("sops/server_fan.yaml")

FANS = {f"fan_{i}" for i in range(1, 7)}


def write_mutated(tmp_path, mutate):
    data = yaml.safe_load(SERVER_FAN.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "sop.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_server_fan_loads(sop):
    assert [s.id for s in sop.steps][:2] == ["fan_1", "fan_2"] and len(sop.steps) == 9
    assert len(sop.layout.slots) == 9 and sop.slot_radius("cover") == 0.25
    assert sop.slot_radius("fan_bay_1") == 0.08
    assert sop.step_label("psu_1") == "PSU 1" and sop.step("cover").evidence == {"cover": "installed"}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["layout"]["slots"]["fan_bay_1"].update(filled_by="fann"), "slot fan_bay_1: filled_by 'fann' is not a detector class"),
        (lambda d: d["steps"][0].update(evidence={"fan_bay_9": "installed"}), "step fan_1: evidence slot 'fan_bay_9' is not declared"),
        (lambda d: d["steps"][1].update(id="fan_1"), "duplicate step id 'fan_1'"),
        (lambda d: d["layout"]["slots"]["fan_bay_1"].update(at=[1.2, 0.5]), "slot fan_bay_1: coordinates must be within [0, 1]"),
        (lambda d: d.update(order="groups", groups=[["fan_1"]]), "groups must contain every step exactly once"),
    ],
)
def test_invalid_config_names_field_and_problem(tmp_path, mutate, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_sop(write_mutated(tmp_path, mutate))


def test_error_message_starts_with_file_name(tmp_path):
    with pytest.raises(ConfigError, match=r"^sop\.yaml: "):
        load_sop(write_mutated(tmp_path, lambda d: d["steps"][1].update(id="fan_1")))


def test_evidence_value_must_be_installed_or_empty(tmp_path):
    with pytest.raises(ConfigError, match="fan_bay_1"):
        load_sop(write_mutated(tmp_path, lambda d: d["steps"][0].update(evidence={"fan_bay_1": "present"})))


def test_must_precede_by_order_mode(sop):
    assert sop.must_precede("fan_3") == {"fan_1", "fan_2"}
    assert sop.model_copy(update={"order": "any"}).must_precede("cover") == set()
    grouped = sop.model_copy(update={"order": "groups", "groups": [sorted(FANS), ["psu_1", "psu_2"], ["cover"]]})
    assert grouped.must_precede("psu_1") == FANS
    assert grouped.must_precede("fan_3") == set()
    assert grouped.must_precede("cover") == FANS | {"psu_1", "psu_2"}


def test_hands_off_steps_and_timing(sop):
    assert sop.step("cover").hands_off is True and sop.step("fan_1").hands_off is False
    assert sop.decisions.evidence_only_after == 1.0 and sop.decisions.hands_off_seconds == 0.5


def test_anchor_must_be_a_detector_class(tmp_path):
    with pytest.raises(ConfigError, match=re.escape("layout: anchor 'frame' is not a detector class")):
        load_sop(write_mutated(tmp_path, lambda d: d["layout"].update(anchor="frame")))


def test_display_name(sop):
    assert sop.name == "Server fan install"


def test_sequence_groups_order_steps_within_the_group(sop):
    from sopvision.config import SequenceGroup

    psus_then_fans_in_sequence = sop.model_copy(update={"order": "groups", "groups": [
        ["psu_1", "psu_2"], SequenceGroup(sequence=["fan_1", "fan_3", "fan_5", "fan_2", "fan_4", "fan_6"]), ["cover"]]})
    assert psus_then_fans_in_sequence.must_precede("psu_2") == set()
    assert psus_then_fans_in_sequence.must_precede("fan_1") == {"psu_1", "psu_2"}
    assert psus_then_fans_in_sequence.must_precede("fan_2") == {"psu_1", "psu_2", "fan_1", "fan_3", "fan_5"}
    assert psus_then_fans_in_sequence.must_precede("cover") == {"psu_1", "psu_2"} | FANS


def test_sequence_groups_load_from_yaml(tmp_path):
    def mutate(d):
        d.update(order="groups", groups=[["psu_1", "psu_2", "cover"],
                                         {"sequence": ["fan_6", "fan_5", "fan_4", "fan_3", "fan_2", "fan_1"]}])
    sop = load_sop(write_mutated(tmp_path, mutate))
    assert sop.must_precede("fan_1") == {"psu_1", "psu_2", "cover", "fan_6", "fan_5", "fan_4", "fan_3", "fan_2"}


def test_tools_must_be_detector_classes_and_not_parts(tmp_path):
    with pytest.raises(ConfigError, match="tool 'wrench' is not a detector class"):
        load_sop(write_mutated(tmp_path, lambda d: d["layout"].update(tools=["wrench"])))
    with pytest.raises(ConfigError, match="tool 'fan' is also a part"):
        load_sop(write_mutated(tmp_path, lambda d: d["layout"].update(tools=["fan"])))


def test_torqued_is_a_valid_evidence_value(tmp_path):
    sop = load_sop(write_mutated(tmp_path, lambda d: d["steps"][0].update(evidence={"fan_bay_1": "torqued"})))
    assert sop.steps[0].evidence == {"fan_bay_1": "torqued"}


def test_filled_by_may_list_several_classes(lug_sop, tmp_path):
    assert lug_sop.layout.slots["hole_1"].fills == ["lug_bolt_blue", "lug_bolt_green", "lug_bolt_red", "lug_bolt_gold"]
    with pytest.raises(ConfigError, match="filled_by 'fan_pink' is not a detector class"):
        load_sop(write_mutated(tmp_path, lambda d: d["layout"]["slots"]["fan_bay_1"].update(filled_by=["fan", "fan_pink"])))


def test_anchor_follow_must_be_in_0_1(tmp_path):
    for bad in (0, 1.5):
        with pytest.raises(ConfigError, match="anchor_follow"):
            load_sop(write_mutated(tmp_path, lambda d: d["layout"].update(anchor_follow=bad)))
