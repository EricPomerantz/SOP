from sopvision.storage.report import build_report, report_csv
from sopvision.storage.runlog import RunEvent

HEADER = "row_type,id,t,status,duration_s,message"


def events():
    return [
        RunEvent("r1", 0.0, "run_started", {"sop": "server_fan_install", "source": "test:Install_12",
                                            "steps": ["fan_1", "fan_2"]}),
        RunEvent("r1", 2.7, "step_update", {"step_id": "fan_1", "status": "active", "source": None, "duration": None}),
        RunEvent("r1", 9.1, "step_update", {"step_id": "fan_1", "status": "done", "source": "jev", "duration": 6.4}),
        RunEvent("r1", 4.0, "alert", {"level": "info", "kind": "chassis_lost", "message": "x", "step_id": None}),
        RunEvent("r1", 12.0, "alert", {"level": "violation", "kind": "sop_incomplete",
                                       "message": "SOP incomplete: missing fan_2", "step_id": None}),
        RunEvent("r1", 12.0, "run_summary", {}),
    ]


def test_report_steps_and_deviations():
    r = build_report(events())
    assert (r.run_id, r.sop, r.source, r.total_seconds) == ("r1", "server_fan_install", "test:Install_12", 12.0)
    assert r.steps[0] == {"id": "fan_1", "status": "done", "duration": 6.4, "completed_by": "jev", "done_t": 9.1}
    assert r.steps[1] == {"id": "fan_2", "status": "pending", "duration": None, "completed_by": None, "done_t": None}
    assert r.deviations == [{"t": 12.0, "level": "violation", "kind": "sop_incomplete",
                             "message": "SOP incomplete: missing fan_2", "step_id": None}]


def test_csv_layout():
    lines = report_csv(build_report(events())).splitlines()
    assert lines[0] == HEADER
    assert lines[1] == "step,fan_1,9.1,done,6.4,completed by jev"
    assert lines[2] == "step,fan_2,,pending,,"
    assert lines[3].startswith("alert,sop_incomplete,12.0,violation,,")
    assert lines[-1].startswith("total,")
