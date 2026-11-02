import { useEffect, useState } from "react";
import { Box, Button, CircularProgress, TextField, Typography } from "@mui/material";
import { Add as AddIcon, AssignmentTurnedIn as SopIcon } from "@mui/icons-material";
import { builderApi, type BuilderApi, type SopListEntry } from "../../builderApi";
import { AppShell } from "../../ui/AppShell";
import { dialogBtn, fieldSx } from "../../ui/dialogStyles";
import { FooterNote, FormDialog, FormField } from "../../ui/dialogKit";
import { EmptyNote, HealthLabel, Panel, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";

type Props = { api?: BuilderApi; go?: (hash: string) => void };

/** An id from a name: lower-case letters, digits and underscores, starting with a letter. */
export function idFromName(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^[^a-z]+/, "").replace(/_+$/, "").slice(0, 48);
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** Every SOP, published and draft, and the way to start a new one (#/sops). */
export function SopsPage({ api = builderApi, go = (h) => { window.location.hash = h; } }: Props) {
  const [sops, setSops] = useState<SopListEntry[] | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  useEffect(() => { api.list().then(setSops).catch((e) => setError(e.message)); }, [api]);

  return (
    <AppShell title="SOPs" current="sops"
      status={<Button variant="contained" disableElevation sx={plantBtn} startIcon={<AddIcon />} onClick={() => setOpen(true)}>New SOP</Button>}>
      {error && <Typography role="alert" color="error">{error}</Typography>}
      <Typography sx={{ fontSize: 13, color: plantColor.muted }}>
        A standard operating procedure the camera watches: its detector, where the parts go, its steps and their order,
        and the recordings its decision model learns from. Open one to work on it stage by stage.
      </Typography>
      {sops && sops.length === 0 && (
        <Panel><EmptyNote icon={<SopIcon />} title="No SOPs yet" action={<Button variant="contained" disableElevation sx={plantBtn} onClick={() => setOpen(true)}>Start the first one</Button>}>
          You need a trained detector model (.pt) to start; the builder walks through everything else.
        </EmptyNote></Panel>
      )}
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))", md: "repeat(3, minmax(0, 1fr))" }, gap: 1.5 }}>
        {sops?.map((s) => (
          <Box key={s.id} component="a" href={`#/sops/${s.id}/detector`}
            sx={{ display: "block", textDecoration: "none", color: "inherit", bgcolor: plantColor.surface, border: `1px solid ${plantColor.line}`, p: 2,
              "&:hover": { borderColor: plantColor.accent } }}>
            <Typography component="h2" sx={{ fontSize: 15, fontWeight: 700, color: plantColor.ink }}>{s.name}</Typography>
            <Typography sx={{ fontSize: 11.5, fontFamily: "monospace", color: plantColor.muted, mb: 1 }}>{s.id}</Typography>
            <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap", fontSize: 12.5 }}>
              {s.published && <HealthLabel health={s.loaded ? "ok" : "warn"} label={s.loaded ? "Published · live" : "Published"} />}
              {s.draft && <HealthLabel health="idle" label={s.published ? "Draft with changes" : "Draft"} />}
              <Typography component="span" sx={{ fontSize: 12.5, color: plantColor.muted }}>{plural(s.runs ?? 0, "recording")}</Typography>
            </Box>
          </Box>
        ))}
      </Box>
      <NewSopDialog open={open} api={api} sops={sops ?? []} onClose={() => setOpen(false)} onCreated={(id) => go(`#/sops/${id}/detector`)} />
    </AppShell>
  );
}

function NewSopDialog({ open, api, sops, onClose, onCreated }: { open: boolean; api: BuilderApi; sops: SopListEntry[]; onClose: () => void; onCreated: (id: string) => void }) {
  const [name, setName] = useState("");
  const [id, setId] = useState("");
  const [idEdited, setIdEdited] = useState(false);
  const [description, setDescription] = useState("");
  const [reuse, setReuse] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => { if (open) { setName(""); setId(""); setIdEdited(false); setDescription(""); setReuse(""); setProblem(null); } }, [open]);
  const changeName = (v: string) => { setName(v); if (!idEdited) setId(idFromName(v)); };
  const withRuns = sops.filter((s) => (s.runs ?? 0) > 0);
  const create = async () => {
    setBusy(true);
    setProblem(null);
    try {
      const view = await api.create(id.trim(), name.trim(), description.trim(), reuse || undefined);
      onClose();
      onCreated(view.id);
    } catch (e) { setProblem(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  return (
    <FormDialog open={open} onClose={onClose} busy={busy} icon={<SopIcon />} title="New SOP" maxWidth="sm"
      status={problem ? <FooterNote tone="bad">{problem}</FooterNote> : <FooterNote>Starts as a draft; nothing runs on it until it is published.</FooterNote>}
      actions={
        <>
          <Button sx={{ ...dialogBtn, color: plantColor.muted }} onClick={onClose} disabled={busy}>Cancel</Button>
          <Button variant="contained" disableElevation sx={{ ...dialogBtn, minWidth: 96 }} onClick={() => void create()} disabled={busy || !id.trim()}
            startIcon={busy ? <CircularProgress size={14} color="inherit" /> : undefined}>Create</Button>
        </>
      }>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <FormField label="Name" required htmlFor="new-sop-name">
          <TextField id="new-sop-name" size="small" fullWidth autoFocus value={name} placeholder="Wheel lug bolts" onChange={(e) => changeName(e.target.value)} sx={fieldSx} />
        </FormField>
        <FormField label="Description" htmlFor="new-sop-description">
          <TextField id="new-sop-description" size="small" fullWidth multiline minRows={2} value={description} onChange={(e) => setDescription(e.target.value)} sx={fieldSx}
            placeholder="Put five lug bolts into the hub, then torque them in star sequence." helperText="What the procedure is, in a sentence. The decision model reads it." />
        </FormField>
        <FormField label="Reuse recordings from" htmlFor="new-sop-reuse">
          <TextField id="new-sop-reuse" select size="small" fullWidth value={reuse} onChange={(e) => setReuse(e.target.value)} sx={fieldSx}
            slotProps={{ select: { native: true } }}
            helperText="For a new version of a procedure that was already recorded. The recordings folder is shared: timelines saved here change that SOP's labels too.">
            <option value="">No: it gets its own recordings folder</option>
            {withRuns.map((s) => <option key={s.id} value={s.id}>{s.name} — {plural(s.runs ?? 0, "recording")}</option>)}
          </TextField>
        </FormField>
        <FormField label="Id" htmlFor="new-sop-id">
          <TextField id="new-sop-id" size="small" fullWidth value={id} onChange={(e) => { setIdEdited(true); setId(e.target.value); }} sx={fieldSx}
            slotProps={{ htmlInput: { style: { fontFamily: "ui-monospace, Menlo, monospace" } } }}
            helperText="From the name. Lower-case letters, digits and _; it names the files and can't change later." />
        </FormField>
      </Box>
    </FormDialog>
  );
}
