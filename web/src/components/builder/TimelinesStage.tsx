import { useEffect, useState } from "react";
import type { BuilderApi, Job, SopView } from "../../builderApi";
import { JobPanel } from "./JobPanel";

type Props = { api: BuilderApi; view: SopView; onReload: () => void };

/** Step timelines per recorded run: the tracker drafts them, a person corrects each on the timeline page. */
export function TimelinesStage({ api, view, onReload }: Props) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const runs = view.status.runs ?? [];
  useEffect(() => {
    api.jobs(view.file).then((jobs) => {
      const running = jobs.find((j) => j.kind === "draft_timelines" && !["done", "failed", "cancelled"].includes(j.state));
      if (running) setJobId(running.id);
    }).catch(() => {});
  }, [api, view.file]);
  const draft = async () => {
    setError("");
    try { setJobId((await api.startJob("draft_timelines", { sop_file: view.file })).id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const prepare = async () => {
    setError("");
    try { setJobId((await api.startJob("previews", { sop_file: view.file })).id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const finished = (job: Job) => { if (job.state === "done") onReload(); };
  return (
    <div className="stage">
      <p className="muted">
        Every run needs a step timeline: which step was under way when, idle stretches, and off-SOP actions. These
        are the decision model's labels. "Draft timelines" replays each run through the detector and tracker and
        writes a draft (and a browser-playable copy of the video); open each run to correct it and save. To label
        from scratch instead, "Prepare videos only" and open a run.
        {view.draft && " Save the draft SOP first: drafting uses its steps."}
      </p>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="row">
        <button onClick={draft} disabled={runs.length === 0 || !view.status.steps}>Draft timelines</button>
        <button onClick={prepare} disabled={runs.length === 0}>Prepare videos only</button>
        <span className="muted">{runs.filter((r) => r.steps).length} of {runs.length} runs have a saved timeline</span>
      </div>
      <JobPanel api={api} jobId={jobId} onFinished={finished} />
      <table className="grid-table" aria-label="Timelines">
        <thead><tr><th>Run</th><th>Split</th><th>Timeline</th><th></th></tr></thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id}>
              <td className="mono">{r.id}</td><td>{r.split ?? "—"}</td>
              <td>{r.steps ? <span className="step-done">✓ saved</span> : r.draft ? "draft to review" : "—"}</td>
              <td><a href={`#/sops/${view.id}/runs/${encodeURIComponent(r.id)}`}>{r.steps ? "Open" : "Label"}</a></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
