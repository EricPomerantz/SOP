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
