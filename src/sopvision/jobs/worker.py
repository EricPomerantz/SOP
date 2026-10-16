"""The worker service: runs queued jobs (sopvision.jobs.store) one at a time, step by step, as subprocesses.

    python -m sopvision.jobs.worker [--jobs-dir data/jobs] [--poll 2]

On start, jobs left `running` by a previous worker (the container was stopped) are marked failed: their steps' outputs
may be half written. A cancel marker stops the running step (SIGTERM to its process group, SIGKILL after a grace
period) and the job ends `cancelled`; a cancelled queued job never starts.
"""

import argparse
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

from sopvision.jobs.store import Job, JobStore

logger = logging.getLogger("sopvision.worker")
GRACE = 10.0  # seconds a cancelled step gets to exit after SIGTERM


def recover(store: JobStore) -> None:
    for job in store.list():
        if job.state == "running":
            job.state, job.error = "failed", "the worker stopped while this job was running"
            for step in job.steps:
                if step.state == "running":
                    step.state, step.ended = "failed", time.time()
            store.save(job)


def run_job(store: JobStore, job: Job, poll: float = 0.5, cwd: Path | None = None) -> Job:
    if store.cancel_requested(job.id):
        job.state = "cancelled"
        store.save(job)
        return job
    job.state = "running"
    store.save(job)
    with store.log_path(job.id).open("ab") as log:
        for step in job.steps:
            if step.state == "done":
                continue
            step.state, step.started = "running", time.time()
            store.save(job)
            log.write(f"\n=== {step.name}: {' '.join(step.argv)}\n".encode())
            log.flush()
            try:
                proc = subprocess.Popen(step.argv, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, start_new_session=True,
                                        env=os.environ | {"PYTHONUNBUFFERED": "1"})
            except OSError as e:
                step.state, step.ended = "failed", time.time()
                job.state, job.error = "failed", f"{step.name}: {e}"
                store.save(job)
                return job
            cancelled = _wait(store, job, proc, poll)
            step.returncode, step.ended = proc.returncode, time.time()
            if cancelled:
                step.state, job.state = "cancelled", "cancelled"
                store.save(job)
                return job
            if proc.returncode != 0:
                step.state = "failed"
                job.state, job.error = "failed", f"{step.name} exited with {proc.returncode} (see the log)"
                store.save(job)
                return job
            step.state = "done"
            store.save(job)
    job.state = "done"
    store.save(job)
    return job


def _wait(store: JobStore, job: Job, proc: subprocess.Popen, poll: float) -> bool:
    """Wait for the step; True if it was cancelled."""
    while proc.poll() is None:
        if store.cancel_requested(job.id):
            _stop(proc)
            return True
        time.sleep(poll)
    return False


def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(GRACE)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", type=Path, default=None, help="default: the JOBS_DIR setting")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between looks for a queued job")
    args = ap.parse_args()
    if args.jobs_dir is None:
        from sopvision.settings import Settings

        args.jobs_dir = Settings().jobs_dir
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    store = JobStore(args.jobs_dir)
    store.root.mkdir(parents=True, exist_ok=True)
    recover(store)
    logger.info(f"worker: watching {store.root}")
    while True:
        job = store.next_queued()
        if job is None:
            time.sleep(args.poll)
            continue
        logger.info(f"worker: {job.id} ({job.title})")
        job = run_job(store, job)
        logger.info(f"worker: {job.id} {job.state}" + (f": {job.error}" if job.error else ""))


if __name__ == "__main__":
    raise SystemExit(main())
