import { useEffect, useState } from "react";
import { Box, Button, CircularProgress, FormControlLabel, Radio, RadioGroup, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from "@mui/material";
import { CenterFocusStrong as DetectorIcon, UploadFile as UploadIcon } from "@mui/icons-material";
import type { BuilderApi, ModelEntry, SopData, SopView } from "../../builderApi";
import { fieldSx } from "../../ui/dialogStyles";
import { FormBanner } from "../../ui/dialogKit";
import { useGlobalAlert } from "../../ui/GlobalAlertProvider";
import { EmptyNote, Panel, Tag, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";

export type Role = "anchor" | "part" | "empty" | "tool" | "hand" | "ignore";

export const ROLE_LABELS: Record<Role, string> = {
  anchor: "Anchor: the work piece the slots sit on", part: "Part: fills a slot", empty: "Empty-slot marker",
  tool: "Tool (a torque wrench)", hand: "Hand", ignore: "Ignore",
};

export function roleOf(data: SopData, cls: string): Role {
  if (cls === "hand") return "hand";
  if (data.layout.anchor === cls) return "anchor";
  if ((data.layout.tools ?? []).includes(cls)) return "tool";
  return data.detector.roles?.[cls] ?? "ignore";
}

/** The draft with `cls` given `role`: one anchor at most, tools in layout.tools, parts and markers in detector.roles. */
export function setRole(data: SopData, cls: string, role: Role): SopData {
  const roles = { ...(data.detector.roles ?? {}) };
  delete roles[cls];
  if (role === "part" || role === "empty") roles[cls] = role;
  let tools = (data.layout.tools ?? []).filter((t) => t !== cls);
  if (role === "tool") tools = [...tools, cls];
  let anchor = data.layout.anchor === cls ? "" : data.layout.anchor;
  if (role === "anchor") anchor = cls;
  return { ...data, detector: { ...data.detector, roles }, layout: { ...data.layout, anchor, tools } };
}

/** "models/lug_bolts/best.pt" → "lug_bolts / best.pt". */
const modelName = (path: string) => path.replace(/^models\//, "").split("/").join(" / ");

type Props = { api: BuilderApi; view: SopView; data: SopData; onChange: (d: SopData) => void; onView: (v: SopView) => void; flush?: () => Promise<void> };

/** Pick the trained detector (one already under models/, or an upload) and say what each of its classes is. */
export function DetectorStage({ api, view, data, onChange, onView, flush }: Props) {
  const alert = useGlobalAlert();
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [pick, setPick] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const ro = !view.draft;

  useEffect(() => {
    api.models().then((m) => { setModels(m); setPick((p) => p || m[0]?.path || ""); })
      .catch((e) => alert.error(`Couldn't list the models under models/: ${e instanceof Error ? e.message : String(e)}`));
  }, [api]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = async (label: string, action: () => Promise<SopView>) => {
    setBusy(label);
    setError("");
    try { await flush?.(); onView(await action()); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(""); }
  };
  const classes = data.detector.classes ?? [];

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {error && <FormBanner tone="bad" onClose={() => setError("")}>{error}</FormBanner>}
      {view.warnings.map((w) => <FormBanner key={w} tone="warn">{w}</FormBanner>)}
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "3fr 2fr" }, gap: 1.5 }}>
        <Panel title="Pick a trained model" subtitle="Detection models already on this machine" icon={<DetectorIcon />}>
          {models.length === 0 ? (
            <EmptyNote compact title="No models found under models/">Upload one on the right.</EmptyNote>
          ) : (
            <RadioGroup value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Models">
              {models.map((m) => (
                <FormControlLabel key={m.path} value={m.path} disabled={ro || !!busy} sx={{ alignItems: "flex-start", mx: 0, mb: 1, "& .MuiFormControlLabel-label": { pt: 0.75 } }}
                  control={<Radio size="small" />}
                  label={
                    <Box>
                      <Typography sx={{ fontSize: 13.5, fontWeight: 600, color: plantColor.ink }}>{modelName(m.path)}</Typography>
                      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.5, alignItems: "center" }}>
                        <Typography component="span" sx={{ fontSize: 12, color: plantColor.muted, mr: 0.5 }}>{m.classes.length} classes</Typography>
                        {m.classes.map((c) => <Tag key={c} tone="dark">{c}</Tag>)}
                      </Box>
                    </Box>
                  } />
              ))}
            </RadioGroup>
          )}
          <Box sx={{ mt: 1 }}>
            <Button variant="contained" disableElevation sx={plantBtn} disabled={ro || !pick || !!busy} onClick={() => run("Loading…", () => api.useModel(view.id, pick))}
              startIcon={busy === "Loading…" ? <CircularProgress size={14} color="inherit" /> : undefined}>Use this model</Button>
          </Box>
        </Panel>
        <Panel title="Upload a model" subtitle="An Ultralytics detection model (.pt)" icon={<UploadIcon />}>
          <Typography sx={{ fontSize: 13, color: plantColor.muted, mb: 1.5 }}>
            Trained outside the app, on this procedure's parts. Only upload files you trust: a .pt file can run code when it is loaded.
          </Typography>
          <Button component="label" variant="outlined" sx={plantBtn} disabled={ro || !!busy}
            startIcon={busy === "Uploading…" ? <CircularProgress size={14} color="inherit" /> : <UploadIcon />}>
            {busy === "Uploading…" ? "Uploading…" : "Choose a .pt file"}
            <input type="file" accept=".pt" hidden aria-label="Upload a .pt file" onChange={(e) => { const f = e.target.files?.[0]; if (f) void run("Uploading…", () => api.uploadDetector(view.id, f)); }} />
          </Button>
        </Panel>
      </Box>
      <Panel title="What each class is" subtitle="Exactly one anchor; the parts fill the slots; the tool is what torques">
        {classes.length === 0 ? (
          <EmptyNote compact icon={<DetectorIcon />} title="No detector yet">Pick or upload a model above: its classes appear here.</EmptyNote>
        ) : (
          <Table size="small" aria-label="Classes">
            <TableHead><TableRow><TableCell>Class</TableCell><TableCell>Role</TableCell></TableRow></TableHead>
            <TableBody>
              {classes.map((c) => (
                <TableRow key={c}>
                  <TableCell sx={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12.5 }}>{c}</TableCell>
                  <TableCell sx={{ width: { md: 420 } }}>
                    <TextField select size="small" fullWidth value={roleOf(data, c)} disabled={ro || c === "hand"} sx={fieldSx}
                      slotProps={{ select: { native: true }, htmlInput: { "aria-label": `Role of ${c}` } }}
                      onChange={(e) => onChange(setRole(data, c, e.target.value as Role))}>
                      {(Object.keys(ROLE_LABELS) as Role[]).filter((r) => r !== "hand" || c === "hand")
                        .map((r) => <option key={r} value={r}>{ROLE_LABELS[r]}</option>)}
                    </TextField>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
        <Box component="details" sx={{ mt: 1.5, fontSize: 12.5, color: plantColor.muted }}>
          <summary style={{ cursor: "pointer" }}>Advanced</summary>
          <Typography sx={{ fontSize: 12.5, mt: 0.75 }}>Weights file: <code>{data.detector.weights}</code>. The scene recognises hands only from a class literally named <code>hand</code>.</Typography>
        </Box>
      </Panel>
    </Box>
  );
}
