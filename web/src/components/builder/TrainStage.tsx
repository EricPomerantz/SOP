import { useCallback, useEffect, useState } from "react";
import type { BenchSummary, BuilderApi, Checkpoints, Job, SopView } from "../../builderApi";
import { JobPanel } from "./JobPanel";

type Props = { api: BuilderApi; view: SopView; onReload: () => void; confirm?: (m: string) => boolean };

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);

function Numbers({ b, title }: { b: BenchSummary; title: string }) {
  return (
    <dl className="latency-result">
      <dt>{title}</dt><dd />
      <dt>Current step right</dt><dd>{pct(b.current_step_accuracy)} of {b.labelled}</dd>
      <dt>Repeats the expected step</dt><dd>{pct(b.echo_rate)}</dd>
      <dt>Off-SOP missed</dt><dd>{pct(b.off_sop_false_negative_rate)}</dd>
      <dt>False off-SOP</dt><dd>{pct(b.off_sop_false_positive_rate)}</dd>
      <dt>False anomaly</dt><dd>{pct(b.anomaly_false_positive_rate)}</dd>
      <dt>Answer time</dt><dd>{b.request_ms_p50.toFixed(0)} ms</dd>
    </dl>
  );
}

/** Train the SOP's own decision model (a Laya checkpoint), see how it does on the test runs, and activate it. */
export function TrainStage({ api, view, confirm = (m) => window.confirm(m) }: Props) {
  const [cps, setCps] = useState<Checkpoints | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [epochs, setEpochs] = useState(12);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const load = useCallback(() => api.checkpoints(view.id).then(setCps).catch((e) => setError(e.message)), [api, view.id]);
  useEffect(() => {
    load();
    api.jobs(view.file).then((jobs) => {
      const j = jobs.find((x) => x.kind === "train_laya");
      if (j) setJobId(j.id);
    }).catch(() => {});
  }, [api, view.file, load]);

  const s = view.status;
  const ready = s.timelines.train > 0 && s.timelines.val > 0 && s.timelines.test > 0;
  const start = async () => {
    setError("");
    try { setJobId((await api.startJob("train_laya", { sop_file: view.file, epochs })).id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const activate = async (checkpoint: string | null) => {
    if (!confirm(checkpoint ? `Make ${checkpoint} answer for this SOP's runs?` : "Go back to the shared checkpoint for this SOP?")) return;
    setError("");
    try {
      const r = await api.activate(view.id, checkpoint);
      setCps(r);
      setNotice(r.warning ?? (checkpoint ? `${checkpoint} now answers for ${view.id}` : "Back to the shared checkpoint"));
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const finished = (job: Job) => { if (job.state === "done") load(); };

  return (
    <div className="stage">
      <p className="muted">
        Trains a Laya checkpoint on this SOP's train and val runs (their timelines are the labels), then measures it on
        the test runs, which it never sees: a frozen benchmark of the requests a live run makes, and a full replay with
        the alerts an operator would get. Takes about an hour on the GPU; the Laya service must be running. Nothing
        changes for live runs until you activate a checkpoint.
      </p>
      {error && <p className="error" role="alert">{error}</p>}
      {notice && <p role="status" className="notice">{notice}</p>}
      <p>
        Timelines saved: train {s.timelines.train ?? 0}/{s.recordings.train ?? 0} · val {s.timelines.val ?? 0}/{s.recordings.val ?? 0} ·
        test {s.timelines.test ?? 0}/{s.recordings.test ?? 0}
        {!ready && <span className="error"> — needs at least one of each</span>}
      </p>
      <div className="row">
        <button onClick={start} disabled={!ready || view.status.errors.length > 0}>Train a new checkpoint</button>
        <label>Epochs <input type="number" min={1} max={40} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} /></label>
        <span className="muted">Answering now: {cps?.active ?? "the shared checkpoint"}</span>
      </div>
      <JobPanel api={api} jobId={jobId} onFinished={finished} />
      {cps && cps.versions.length > 0 && (
        <table className="grid-table" aria-label="Checkpoints">
          <thead><tr><th>Checkpoint</th><th>Test benchmark</th><th>Replayed test runs</th><th></th></tr></thead>
          <tbody>
            {cps.versions.map((v) => (
              <tr key={v.checkpoint}>
                <td className="mono">{v.checkpoint}{cps.active === v.checkpoint && <span className="pill pill-ok">active</span>}</td>
                <td>{v.report?.bench ? <Numbers b={v.report.bench} title="this checkpoint" /> : <span className="muted">no report</span>}
                  {v.report?.bench_active && <Numbers b={v.report.bench_active} title={`vs ${v.report.active}`} />}</td>
                <td>{v.report?.replay.map((r) => (
                  <div key={r.run} className="mono">{r.run}: step {pct(r.current_step)}, idle/off {r.idle_or_off_sop}
                    {r.not_done.length > 0 && `, not done ${r.not_done.join(" ")}`}{r.alerts.length > 0 && `, ${r.alerts.length} alerts`}</div>
                ))}</td>
                <td>{cps.active !== v.checkpoint && <button onClick={() => activate(v.checkpoint)}>Activate</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {cps?.active && <button className="discard" onClick={() => activate(null)}>Use the shared checkpoint instead</button>}
    </div>
  );
}
