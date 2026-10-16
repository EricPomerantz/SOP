import { useEffect, useState } from "react";
import type { BuilderApi, Job } from "../../builderApi";

type Props = { api: BuilderApi; jobId: string | null; poll?: number; onFinished?: (job: Job) => void };

const FINISHED = ["done", "failed", "cancelled"];

/** A background job's progress: steps, the end of its log, and a cancel button. Polls until it finishes. */
export function JobPanel({ api, jobId, poll = 2000, onFinished }: Props) {
  const [job, setJob] = useState<Job | null>(null);
  useEffect(() => {
    if (!jobId) { setJob(null); return; }
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const j = await api.job(jobId);
        if (stop) return;
        setJob(j);
        if (FINISHED.includes(j.state)) { onFinished?.(j); return; }
      } catch { /* the app restarting: try again */ }
      timer = setTimeout(tick, poll);
    };
    tick();
    return () => { stop = true; clearTimeout(timer); };
  }, [api, jobId, poll]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!job) return null;
  const pill = job.state === "done" ? "pill-ok" : job.state === "failed" ? "pill-bad" : job.state === "running" ? "pill-warn" : "pill-idle";
  return (
    <div className="job card" aria-label="Job">
      <div className="row">
        <strong>{job.title}</strong> <span className={`pill ${pill}`}>{job.state}</span>
        {!FINISHED.includes(job.state) && (
          <button className="discard" disabled={job.cancel_requested} onClick={() => api.cancelJob(job.id)}>
            {job.cancel_requested ? "Cancelling…" : "Cancel"}
          </button>
        )}
      </div>
      {job.state === "queued" && <p className="muted">Waiting for the worker (it runs one job at a time).</p>}
      <ol className="job-steps">
        {job.steps.map((s) => <li key={s.name} className={`job-${s.state}`}>{s.name} <span className="muted">{s.state}</span></li>)}
      </ol>
      {job.error && <p className="error">{job.error}</p>}
      {job.log && <details open={job.state === "failed"}><summary>Log</summary><pre className="job-log">{job.log.slice(-6000)}</pre></details>}
    </div>
  );
}
