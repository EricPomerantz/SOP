import sys
import threading
import time

from sopvision.jobs.store import JobStore, Step
from sopvision.jobs.worker import recover, run_job

PY = sys.executable


def test_a_job_runs_its_steps_in_order_and_logs_them(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("demo", "Demo", [Step("one", [PY, "-c", "print('first')"]),
                                        Step("two", [PY, "-c", "print('second')"])])
    done = run_job(store, store.next_queued(), poll=0.05)
    assert done.state == "done" and [s.state for s in done.steps] == ["done", "done"]
    assert store.get(job.id).state == "done" and store.next_queued() is None
    log = store.log_tail(job.id)
    assert log.index("first") < log.index("second")


def test_a_failing_step_fails_the_job_and_skips_the_rest(tmp_path):
    store = JobStore(tmp_path)
    store.create("demo", "Demo", [Step("bad", [PY, "-c", "raise SystemExit(3)"]), Step("never", [PY, "-c", "0"])])
    job = run_job(store, store.next_queued(), poll=0.05)
    assert job.state == "failed" and "exited with 3" in job.error
    assert [s.state for s in job.steps] == ["failed", "queued"]


def test_cancel_stops_the_running_step(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("demo", "Demo", [Step("slow", [PY, "-c", "import time; time.sleep(60)"])])
    threading.Timer(0.5, store.cancel, [job.id]).start()
    t = time.monotonic()
    out = run_job(store, store.next_queued(), poll=0.05)
    assert out.state == "cancelled" and out.steps[0].state == "cancelled" and time.monotonic() - t < 10


def test_a_cancelled_queued_job_never_starts(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("demo", "Demo", [Step("x", [PY, "-c", "open('ran', 'w')"])])
    store.cancel(job.id)
    assert run_job(store, store.get(job.id), cwd=tmp_path).state == "cancelled"
    assert not (tmp_path / "ran").exists()


def test_jobs_left_running_by_a_stopped_worker_are_marked_failed(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("demo", "Demo", [Step("x", ["true"])])
    job.state, job.steps[0].state = "running", "running"
    store.save(job)
    recover(store)
    again = store.get(job.id)
    assert again.state == "failed" and again.steps[0].state == "failed" and "worker stopped" in again.error


def test_jobs_list_newest_first_and_queue_oldest_first(tmp_path):
    store = JobStore(tmp_path)
    a = store.create("demo", "A", [], sop="a.yaml")
    b = store.create("demo", "B", [], sop="b.yaml")
    assert [j.id for j in store.list()] == [b.id, a.id]
    assert [j.id for j in store.list(sop="a.yaml")] == [a.id]
    assert store.next_queued().id == a.id
