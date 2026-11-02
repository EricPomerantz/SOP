import { useState } from "react";
import { Box, Button, CircularProgress, Typography } from "@mui/material";
import { Publish as PublishIcon } from "@mui/icons-material";
import type { SopData, SopView } from "../../builderApi";
import { dialogBtn } from "../../ui/dialogStyles";
import { FooterNote, FormBanner, FormDialog } from "../../ui/dialogKit";
import { InfoRows, plantColor } from "../../ui/kit";

type Props = { open: boolean; view: SopView; data: SopData; onClose: () => void; onPublish: () => Promise<void> };

const n = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;

/** What the SOP would go live with, what is still missing, and the Publish button. */
export function PublishDialog({ open, view, data, onClose, onPublish }: Props) {
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const s = view.status;
  const slots = Object.keys(data.layout.slots ?? {}).length;
  const order = data.order === "strict" ? "in the list order" : data.order === "any" ? "in any order" : `in ${n((data.groups ?? []).length, "group")}`;
  const splits = ["train", "val", "test"].map((k) => `${s.recordings[k] ?? 0} ${k}`).join(" · ");
  const timed = Object.values(s.timelines).reduce((a, b) => a + b, 0);
  const runs = Object.values(s.recordings).reduce((a, b) => a + b, 0);
  const publish = async () => {
    setBusy(true);
    setProblem(null);
    try { await onPublish(); } catch (e) { setProblem(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  return (
    <FormDialog open={open} onClose={onClose} busy={busy} icon={<PublishIcon />} title={`Publish ${data.name || view.id}?`} maxWidth="sm"
      status={problem ? <FooterNote tone="bad">{problem}</FooterNote> : null}
      actions={
        <>
          <Button sx={{ ...dialogBtn, color: plantColor.muted }} onClick={onClose} disabled={busy}>Not yet</Button>
          <Button variant="contained" disableElevation sx={{ ...dialogBtn, minWidth: 96 }} onClick={() => void publish()}
            disabled={busy || !s.publish.ready} startIcon={busy ? <CircularProgress size={14} color="inherit" /> : undefined}>
            Publish
          </Button>
        </>
      }>
      {!s.publish.ready && (
        <FormBanner tone="bad">
          <strong>Before it can be published</strong>
          <Box component="ul" sx={{ m: 0, mt: 0.5, pl: 2.25 }}>{s.publish.blockers.map((b) => <li key={b}>{b}</li>)}</Box>
        </FormBanner>
      )}
      <Typography sx={{ fontSize: 13, color: plantColor.muted, mb: 1.5 }}>
        Publishing puts the SOP in the dashboard, where runs can be started on it. {s.published ? "The published version is replaced; the old one is kept." : ""}
        {" "}Recordings, timelines and the decision model can still be worked on afterwards.
      </Typography>
      <InfoRows rows={[
        ["Detector", data.detector.classes.length ? `${n(data.detector.classes.length, "class")}, anchor ${data.layout.anchor || "—"}` : ""],
        ["Slots", slots ? n(slots, "slot") : ""],
        ["Steps", data.steps.length ? `${n(data.steps.length, "step")}, ${order}` : ""],
        ["Recordings", runs ? splits : ""],
        ["Timelines", runs ? `${timed} of ${runs} runs labelled` : ""],
        ["Decision model", s.laya_checkpoint ?? "the shared model (train one in the last stage)"],
      ]} />
    </FormDialog>
  );
}
