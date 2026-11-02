import { useEffect, useState } from "react";
import { Box, Button, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import { Timeline as TimelineIcon } from "@mui/icons-material";
import type { BuilderApi, Job, RunRow, SopView } from "../../builderApi";
import { FormBanner } from "../../ui/dialogKit";
import { useGlobalAlert } from "../../ui/GlobalAlertProvider";
import { EmptyNote, HealthLabel, Panel, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";
import { JobPanel } from "./JobPanel";

type Props = { api: BuilderApi; view: SopView; onReload: () => void; flush?: () => Promise<void> };

const status = (r: RunRow) =>
  r.steps ? <HealthLabel health="ok" label="Saved" /> : r.draft ? <HealthLabel health="pending" label="Draft ready to review" /> : <HealthLabel health="idle" label="Not started" strong={false} />;

/** Step timelines per recorded run: the tracker drafts them, a person checks and corrects each on the run's page. */
export function TimelinesStage({ api, view, onReload, flush }: Props) {
  const alert = useGlobalAlert();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const runs = view.status.runs ?? [];
  const missing = runs.filter((r) => !r.steps).map((r) => r.id);
  const shared = (view.data.dataset?.dir ?? view.id) !== view.id;
  useEffect(() => {
    api.jobs(view.file).then((jobs) => {
      const running = jobs.find((j) => ["draft_timelines", "previews"].includes(j.kind) && !["done", "failed", "cancelled"].includes(j.state));
      if (running) setJobId(running.id);
    }).catch((e) => alert.error(`Couldn't check for a running job: ${e instanceof Error ? e.message : String(e)}`));
  }, [api, view.file]); // eslint-disable-line react-hooks/exhaustive-deps
  const start = async (kind: "draft_timelines" | "previews", only?: string[]) => {
    setError("");
    try {
      await flush?.();
      setJobId((await api.startJob(kind, only ? { sop_file: view.file, runs: only } : { sop_file: view.file })).id);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const finished = (job: Job) => { if (job.state === "done") onReload(); };
  const noSteps = !view.status.steps;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {error && <FormBanner tone="bad" onClose={() => setError("")}>{error}</FormBanner>}
      {shared && <FormBanner tone="warn">These recordings are shared with another SOP: a timeline saved here changes its labels too.</FormBanner>}
      <Panel title="Draft timelines with the tracker" icon={<TimelineIcon />}
        subtitle="Replays runs through the detector and tracker and writes a draft timeline to check on each run's page">
        <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
          {missing.length > 0 && missing.length < runs.length && (
            <Button variant="contained" disableElevation sx={plantBtn} disabled={noSteps} onClick={() => void start("draft_timelines", missing)}>
              Draft the {missing.length} runs without a timeline
            </Button>
          )}
          <Button variant={missing.length > 0 && missing.length < runs.length ? "outlined" : "contained"} disableElevation sx={plantBtn}
            disabled={runs.length === 0 || noSteps} onClick={() => void start("draft_timelines")}>Draft all runs</Button>
          <Button variant="text" sx={plantBtn} disabled={runs.length === 0} onClick={() => void start("previews")}>Prepare videos only</Button>
          <Typography sx={{ fontSize: 13, color: plantColor.ink, ml: "auto", fontWeight: 600 }}>
            {runs.filter((r) => r.steps).length} of {runs.length} runs have a saved timeline
          </Typography>
        </Box>
        <Typography sx={{ fontSize: 12.5, color: plantColor.muted, mt: 1 }}>
          {noSteps ? "Add the steps first: the draft is labelled with them. " : ""}
          Each run without a browser-playable copy gets one made too, which takes a few minutes per run; draft one run at a time from the table to start quickly.
          To label from scratch instead, prepare the videos and open a run.
        </Typography>
      </Panel>
      <JobPanel api={api} jobId={jobId} onFinished={finished} />
      <Panel title="Runs" flush={runs.length > 0}>
        {runs.length === 0 ? (
          <EmptyNote icon={<TimelineIcon />} title="No recordings yet">Record runs in the Recordings stage first.</EmptyNote>
        ) : (
          <Table size="small" aria-label="Timelines">
            <TableHead><TableRow><TableCell>Run</TableCell><TableCell>Split</TableCell><TableCell>Timeline</TableCell><TableCell /></TableRow></TableHead>
            <TableBody>
              {runs.map((r) => {
                const verb = r.steps ? "Open" : r.draft ? "Review" : "Label";
                return (
                  <TableRow key={r.id}>
                    <TableCell sx={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12.5 }}>{r.id}</TableCell>
                    <TableCell>{r.split ?? "—"}</TableCell>
                    <TableCell>{status(r)}</TableCell>
                    <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                      <Button size="small" variant="text" sx={plantBtn} disabled={noSteps} aria-label={`Draft ${r.id}`} onClick={() => void start("draft_timelines", [r.id])}>Draft</Button>
                      <Button size="small" variant="outlined" sx={{ ...plantBtn, ml: 0.5 }} component="a" aria-label={`${verb} ${r.id}`}
                        href={`#/sops/${view.id}/runs/${encodeURIComponent(r.id)}`}>{verb}</Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </Panel>
    </Box>
  );
}
