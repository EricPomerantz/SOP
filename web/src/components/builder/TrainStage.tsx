import { useCallback, useEffect, useState } from "react";
import { Box, Button, TextField, Typography } from "@mui/material";
import { Bolt as SpeedIcon, CheckCircleOutline as RightIcon, ModelTraining as TrainIcon, NotificationsOff as MissIcon, ReportProblem as AlarmIcon } from "@mui/icons-material";
import type { BenchSummary, BuilderApi, Checkpoints, Job, SopView } from "../../builderApi";
import { ConfirmDialog } from "../../ui/ConfirmDialog";
import { fieldSx } from "../../ui/dialogStyles";
import { FormBanner, FormField } from "../../ui/dialogKit";
import { useGlobalAlert } from "../../ui/GlobalAlertProvider";
import { EmptyNote, InfoRows, KpiTile, Panel, StatusPill, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";
import { JobPanel } from "./JobPanel";

type Props = { api: BuilderApi; view: SopView; onReload: () => void; flush?: () => Promise<void> };

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);

/** A checkpoint's test results in plain words: tiles for what an operator notices. */
function Results({ b }: { b: BenchSummary }) {
  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))", lg: "repeat(4, minmax(0, 1fr))" }, gap: 1 }}>
      <KpiTile icon={<RightIcon />} label="Named the right step" value={pct(b.current_step_accuracy)}
        detail={`of ${b.labelled} moments; always guessing the expected one: ${pct(b.echo_rate)}`} />
      <KpiTile icon={<MissIcon />} label="Missed off-SOP actions" value={pct(b.off_sop_false_negative_rate)} detail="lower is better"
        health={(b.off_sop_false_negative_rate ?? 0) > 0.3 ? "warn" : undefined} />
      <KpiTile icon={<AlarmIcon />} label="False alarms" value={pct((b.anomaly_false_positive_rate ?? 0) + (b.off_sop_false_positive_rate ?? 0))}
        detail="moments wrongly flagged" health={((b.anomaly_false_positive_rate ?? 0) + (b.off_sop_false_positive_rate ?? 0)) > 0.1 ? "warn" : undefined} />
      <KpiTile icon={<SpeedIcon />} label="Answer time" value={`${b.request_ms_p50.toFixed(0)} ms`} detail="typical" />
    </Box>
  );
}

/** Train the SOP's own decision model, see how it does on the test runs it never saw, and switch it on. */
export function TrainStage({ api, view, onReload, flush }: Props) {
  const alert = useGlobalAlert();
  const [cps, setCps] = useState<Checkpoints | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [epochs, setEpochs] = useState(12);
  const [error, setError] = useState("");
  const [asking, setAsking] = useState<{ checkpoint: string | null } | null>(null);
  const load = useCallback(() => api.checkpoints(view.id).then(setCps).catch((e) => setError(e.message)), [api, view.id]);
  useEffect(() => {
    load();
    api.jobs(view.file).then((jobs) => {
      const j = jobs.find((x) => x.kind === "train_laya" && !["done", "failed", "cancelled"].includes(x.state));
      if (j) setJobId(j.id);
    }).catch(() => {});
  }, [api, view.file, load]);

  const s = view.status;
  const missing = (["train", "val", "test"] as const).filter((k) => !(s.timelines[k] > 0));
  const blocker = missing.length ? `Training needs at least one ${missing.join(" and one ")} run with a timeline.`
    : s.errors.length ? "Fix the SOP's problems first (Review and publish lists them)." : "";
  const start = async () => {
    setError("");
    try { await flush?.(); setJobId((await api.startJob("train_laya", { sop_file: view.file, epochs })).id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const activate = async (checkpoint: string | null) => {
    const r = await api.activate(view.id, checkpoint);
    setCps(r);
    setAsking(null);
    alert.success(r.warning ?? (checkpoint ? `${checkpoint} now judges this SOP's runs.` : "Back to the shared model."));
    onReload();
  };
  const finished = (job: Job) => { if (job.state === "done") { load(); onReload(); } };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {error && <FormBanner tone="bad" onClose={() => setError("")}>{error}</FormBanner>}
      <Panel title="Train a new model" icon={<TrainIcon />} subtitle="Learns from the train and val runs' timelines, then is tested on the test runs it never sees">
        <InfoRows rows={[
          ["Timelines saved", ["train", "val", "test"].map((k) => `${s.timelines[k] ?? 0}/${s.recordings[k] ?? 0} ${k}`).join(" · ")],
          ["Judging runs now", cps?.active ?? "the shared model"],
          ["Takes", "about an hour on the GPU, alongside live runs"],
        ]} />
        {blocker && <Typography sx={{ fontSize: 12.5, color: plantColor.muted, mt: 1 }}>{blocker}</Typography>}
        <Box sx={{ display: "flex", gap: 1, alignItems: "flex-end", flexWrap: "wrap", mt: 1.5 }}>
          <Button variant="contained" disableElevation sx={plantBtn} startIcon={<TrainIcon />} disabled={!!blocker} onClick={() => void start()}>Train a new model</Button>
          <Typography sx={{ fontSize: 12, color: plantColor.muted }}>The Laya service must be running (<code>./start.sh --model laya</code>). Nothing changes for live runs until you switch a model on.</Typography>
        </Box>
        <Box component="details" sx={{ mt: 1, fontSize: 12.5, color: plantColor.muted }}>
          <summary style={{ cursor: "pointer" }}>Advanced</summary>
          <Box sx={{ mt: 1, maxWidth: 240 }}>
            <FormField label="Training passes (epochs)" htmlFor="epochs">
              <TextField id="epochs" type="number" size="small" value={epochs} sx={fieldSx} slotProps={{ htmlInput: { min: 1, max: 40 } }}
                onChange={(e) => setEpochs(Number(e.target.value))} helperText="12 suits a few dozen runs." />
            </FormField>
          </Box>
        </Box>
      </Panel>
      <JobPanel api={api} jobId={jobId} onFinished={finished} />
      {cps && cps.versions.length === 0 && <Panel><EmptyNote compact title="No model trained for this SOP yet">Its runs are judged by the shared model meanwhile.</EmptyNote></Panel>}
      {cps?.versions.map((v) => (
        <Panel key={v.checkpoint} title={v.checkpoint} subtitle="Tested on the test runs"
          actions={cps.active === v.checkpoint ? <StatusPill health="ok" label="Judging runs" />
            : <Button variant="outlined" sx={plantBtn} aria-label={`Switch on ${v.checkpoint}`} onClick={() => setAsking({ checkpoint: v.checkpoint })}>Switch on</Button>}>
          {v.report?.bench ? <Results b={v.report.bench} /> : <Typography sx={{ fontSize: 12.5, color: plantColor.muted }}>No test report (the job may have stopped early).</Typography>}
          {v.report?.bench_active && (
            <Box sx={{ mt: 1.5 }}>
              <Typography sx={{ fontSize: 12.5, color: plantColor.muted, mb: 0.75 }}>The model judging runs now ({v.report.active}), on the same test runs:</Typography>
              <Results b={v.report.bench_active} />
            </Box>
          )}
          {v.report?.replay && v.report.replay.length > 0 && (
            <Box component="details" sx={{ mt: 1.5, fontSize: 12.5, color: plantColor.muted }}>
              <summary style={{ cursor: "pointer" }}>Each test run replayed</summary>
              <Box component="ul" sx={{ m: 0, mt: 0.5, pl: 2.25 }}>
                {v.report.replay.map((r) => (
                  <li key={r.run}><code>{r.run}</code>: right step {pct(r.current_step)}{r.not_done.length > 0 && `, never completed ${r.not_done.join(", ")}`}
                    {r.alerts.length > 0 && `, ${r.alerts.length} alerts`}</li>
                ))}
              </Box>
            </Box>
          )}
        </Panel>
      ))}
      {cps?.active && <Box><Button variant="text" color="error" sx={plantBtn} onClick={() => setAsking({ checkpoint: null })}>Go back to the shared model</Button></Box>}
      <ConfirmDialog open={asking !== null} tone="accent" actionLabel="Switch it on"
        title={asking?.checkpoint ? `Switch on ${asking.checkpoint}?` : "Go back to the shared model?"}
        body={asking?.checkpoint ? "It judges this SOP's live runs from the next run on. You can switch back at any time." : "This SOP's runs are judged by the shared model from the next run on."}
        onClose={() => setAsking(null)} onConfirm={() => activate(asking?.checkpoint ?? null)} />
    </Box>
  );
}
