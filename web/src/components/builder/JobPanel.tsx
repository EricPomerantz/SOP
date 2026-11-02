import { useEffect, useRef, useState } from "react";
import { Box, Button, LinearProgress, Typography } from "@mui/material";
import type { BuilderApi, Job } from "../../builderApi";
import { ConfirmDialog } from "../../ui/ConfirmDialog";
import { HealthDot, HealthLabel, Panel, plantColor, type Health } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";

type Props = { api: BuilderApi; jobId: string | null; poll?: number; onFinished?: (job: Job) => void };

const FINISHED = ["done", "failed", "cancelled"];

export function elapsedText(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`;
}

const stepHealth = (state: string): Health =>
  state === "done" ? "ok" : state === "running" ? "pending" : state === "failed" ? "bad" : state === "cancelled" ? "warn" : "idle";

/** A background job as it runs: how far it is, the step under way, how long it has taken, and its log when it fails. */
export function JobPanel({ api, jobId, poll = 2000, onFinished }: Props) {
  const [job, setJob] = useState<Job | null>(null);
  const [asking, setAsking] = useState(false);
  const onFinishedRef = useRef(onFinished);
  onFinishedRef.current = onFinished;
  useEffect(() => {
    if (!jobId) { setJob(null); return; }
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const j = await api.job(jobId);
        if (stop) return;
        setJob(j);
        if (FINISHED.includes(j.state)) { onFinishedRef.current?.(j); return; }
      } catch { /* the app restarting: try again */ }
      timer = setTimeout(tick, poll);
    };
    void tick();
    return () => { stop = true; clearTimeout(timer); };
  }, [api, jobId, poll]);
  if (!job) return null;

  const started = job.steps.find((s) => s.started)?.started ?? job.created;
  const ended = [...job.steps].reverse().find((s) => s.ended)?.ended;
  const finished = FINISHED.includes(job.state);
  const took = elapsedText((finished ? (ended ?? Date.now() / 1000) : Date.now() / 1000) - started);
  const done = job.steps.filter((s) => s.state === "done").length;
  const running = job.steps.find((s) => s.state === "running");
  const health: Health = job.state === "done" ? "ok" : job.state === "failed" ? "bad" : job.state === "cancelled" ? "warn" : job.state === "running" ? "pending" : "idle";
  const line =
    job.state === "queued" ? "Waiting for the worker: it runs one job at a time, so another job may be ahead of this one."
    : job.state === "running" ? `Running: ${running?.name ?? "…"} · ${took}`
    : job.state === "done" ? `Finished in ${took}`
    : job.state === "failed" ? `Failed after ${took}: the log below says why`
    : `Cancelled after ${took}`;

  return (
    <Panel title={job.title} health={job.state === "failed" ? "bad" : undefined} aria-label="Job"
      actions={!finished && (
        <Button size="small" variant="outlined" color="error" sx={plantBtn} disabled={job.cancel_requested} onClick={() => setAsking(true)}>
          {job.cancel_requested ? "Cancelling…" : "Cancel"}
        </Button>
      )}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
        <HealthLabel health={health} label={line} />
        <LinearProgress variant={job.state === "running" && !running ? "indeterminate" : "determinate"}
          value={job.steps.length ? (100 * done) / job.steps.length : 0} sx={{ height: 4 }} aria-label="Progress" />
        <Box component="ol" sx={{ m: 0, p: 0, listStyle: "none", display: "flex", flexWrap: "wrap", gap: 1.5, fontSize: 12.5, color: plantColor.muted }}>
          {job.steps.map((s) => (
            <Box component="li" key={s.name} sx={{ display: "inline-flex", alignItems: "center", gap: 0.75, color: s.state === "running" ? plantColor.ink : undefined, fontWeight: s.state === "running" ? 600 : 400 }}>
              <HealthDot health={stepHealth(s.state)} />{s.name}
            </Box>
          ))}
        </Box>
        {job.error && <Typography role="alert" sx={{ fontSize: 13, color: plantColor.ink, borderLeft: "3px solid", borderColor: "error.main", pl: 1.25, py: 0.5 }}>{job.error}</Typography>}
        {job.log && (
          <Box component="details" open={job.state === "failed"} sx={{ fontSize: 12.5, color: plantColor.muted }}>
            <summary style={{ cursor: "pointer" }}>Log</summary>
            <Box component="pre" sx={{ m: 0, mt: 0.5, p: 1, maxHeight: 260, overflow: "auto", fontSize: 12, bgcolor: plantColor.canvas, whiteSpace: "pre-wrap", color: plantColor.ink }}>{job.log.slice(-6000)}</Box>
          </Box>
        )}
      </Box>
      <ConfirmDialog open={asking} title="Cancel this job?" body="What it has done so far is thrown away." actionLabel="Cancel the job"
        onClose={() => setAsking(false)} onConfirm={async () => { await api.cancelJob(job.id); setAsking(false); }} />
    </Panel>
  );
}
