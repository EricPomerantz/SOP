"""Run report: per-step outcome and duration, deviations, total time — as JSON-able data or CSV."""

import csv
import io
from dataclasses import dataclass

from sopvision.storage.runlog import RunEvent

CSV_HEADER = ["row_type", "id", "t", "status", "duration_s", "message"]


@dataclass(frozen=True)
class RunReport:
    run_id: str
    sop: str
    source: str
    total_seconds: float
    steps: list[dict]
    deviations: list[dict]


def build_report(events: list[RunEvent]) -> RunReport:
    started = next(e for e in events if e.type == "run_started")
    order = list(started.payload.get("steps", []))
    latest: dict[str, dict] = {}
    for e in events:
        if e.type == "step_update":
            latest[e.payload["step_id"]] = e.payload | {"t": e.t}
            if e.payload["step_id"] not in order:
                order.append(e.payload["step_id"])
    steps = []
    for sid in order:
        u = latest.get(sid, {})
        done = u.get("status") == "done"
        steps.append({"id": sid, "status": u.get("status", "pending"), "duration": u.get("duration") if done else None,
                      "completed_by": u.get("source") if done else None, "done_t": u.get("t") if done else None})
    deviations = [{"t": e.t, "level": e.payload["level"], "kind": e.payload["kind"], "message": e.payload["message"],
                   "step_id": e.payload.get("step_id")}
                  for e in events if e.type == "alert" and e.payload["level"] in ("warning", "violation")]
    total = round(max(e.t for e in events) - started.t, 2)
    return RunReport(started.run_id, started.payload.get("sop", ""), started.payload.get("source", ""), total, steps,
                     deviations)


def report_csv(report: RunReport) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_HEADER)
    for s in report.steps:
        w.writerow(["step", s["id"], "" if s["done_t"] is None else s["done_t"], s["status"],
                    "" if s["duration"] is None else s["duration"],
                    f"completed by {s['completed_by']}" if s["completed_by"] else ""])
    for d in report.deviations:
        w.writerow(["alert", d["kind"], d["t"], d["level"], "", d["message"]])
    w.writerow(["total", "", report.total_seconds, "", "", ""])
    return buf.getvalue()
