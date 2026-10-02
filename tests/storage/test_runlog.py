import pytest

from sopvision.storage.runlog import RunEvent, RunLog, RunLogClosedError, read_events


def test_round_trip(tmp_path):
    log = RunLog(tmp_path / "r.jsonl")
    events = [RunEvent("r1", 0.0, "run_started", {"sop": "server_fan_install"}),
              RunEvent("r1", 1.5, "alert", {"level": "warning", "kind": "off_sop", "message": "Off-SOP activity"})]
    for e in events:
        log.write(e)
    log.close()
    assert read_events(tmp_path / "r.jsonl") == events


def test_each_write_is_flushed_immediately(tmp_path):
    log = RunLog(tmp_path / "r.jsonl")
    log.write(RunEvent("r1", 0.0, "run_started", {}))
    assert len(read_events(tmp_path / "r.jsonl")) == 1  # readable before close
    log.close()


def test_write_after_close_raises(tmp_path):
    log = RunLog(tmp_path / "r.jsonl")
    log.close()
    with pytest.raises(RunLogClosedError):
        log.write(RunEvent("r1", 0.0, "run_started", {}))
