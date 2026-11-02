import { useEffect, useState } from "react";
import { Box, Button, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import { School as TrainIcon, Tune as ValIcon, FactCheck as TestIcon, Videocam as RecordIcon } from "@mui/icons-material";
import type { BuilderApi, SopListEntry, SopView } from "../../builderApi";
import { FormBanner } from "../../ui/dialogKit";
import { EmptyNote, HealthLabel, KpiTile, Panel, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";
import { stageHash } from "./stages";

type Props = { api: BuilderApi; view: SopView };

/** The SOP's recorded runs by split, and the way to record more (the Record page, coming back here after). */
export function RecordingsStage({ api, view }: Props) {
  const runs = view.status.runs ?? [];
  const c = view.status.recordings;
  const dir = view.data.dataset?.dir ?? view.id;
  const [owner, setOwner] = useState<SopListEntry | null>(null);
  useEffect(() => {
    if (dir === view.id) { setOwner(null); return; }
    api.list().then((l) => setOwner(l.find((s) => s.id !== view.id && s.dataset_dir === dir) ?? null)).catch(() => setOwner(null));
  }, [api, dir, view.id]);
  const recordHref = `#/record?sop=${encodeURIComponent(view.file)}&back=${encodeURIComponent(stageHash(view.id, "recordings"))}`;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {owner && (
        <FormBanner tone="warn">
          <strong>Shared with {owner.name}.</strong> The folder datasets/{dir} is that SOP's too: runs recorded and timelines saved here change its recordings and labels as well.
        </FormBanner>
      )}
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(3, minmax(0, 1fr))" }, gap: 1.5 }}>
        <KpiTile icon={<TrainIcon />} label="Train runs" value={c.train ?? 0} detail="teach the model" health={(c.train ?? 0) ? undefined : "idle"} />
        <KpiTile icon={<ValIcon />} label="Val runs" value={c.val ?? 0} detail="tune it while training" />
        <KpiTile icon={<TestIcon />} label="Test runs" value={c.test ?? 0} detail="measure it; never trained on" />
      </Box>
      <Panel title="Recorded runs" subtitle="Correct runs and mistakes: a wrong order, a part taken out, idle time, off-SOP actions" icon={<RecordIcon />}
        actions={<Button component="a" href={recordHref} variant="contained" disableElevation sx={plantBtn} startIcon={<RecordIcon />}>Record a run</Button>}
        flush={runs.length > 0}>
        {runs.length === 0 ? (
          <EmptyNote icon={<RecordIcon />} title="No recordings yet" action={<Button component="a" href={recordHref} variant="outlined" sx={plantBtn}>Record a run</Button>}>
            Record the procedure with the camera a dozen times or more: most as train runs, a couple as val, several as test. Recording works once the detector stage is done.
          </EmptyNote>
        ) : (
          <Table size="small" aria-label="Recordings">
            <TableHead><TableRow><TableCell>Run</TableCell><TableCell>Split</TableCell><TableCell>Torque log</TableCell><TableCell>Timeline</TableCell></TableRow></TableHead>
            <TableBody>
              {runs.map((r) => (
                <TableRow key={r.id}>
                  <TableCell sx={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12.5 }}>{r.id}</TableCell>
                  <TableCell>{r.split ?? <HealthLabel health="warn" label="not in a split" title="Add it to datasets/<dir>/splits.yaml" />}</TableCell>
                  <TableCell>{r.torque ? "yes" : <Typography component="span" sx={{ color: plantColor.faint }}>—</Typography>}</TableCell>
                  <TableCell>{r.steps ? <HealthLabel health="ok" label="Saved" /> : r.draft ? <HealthLabel health="pending" label="Draft to review" /> : <HealthLabel health="idle" label="Not started" strong={false} />}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Panel>
    </Box>
  );
}
