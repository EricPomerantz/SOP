import { useState } from "react";
import { Box, Button, Checkbox, IconButton, Table, TableBody, TableCell, TableHead, TableRow, TextField, Tooltip, Typography } from "@mui/material";
import { Add as AddIcon, ArrowDownward as DownIcon, ArrowUpward as UpIcon, AutoFixHigh as GenerateIcon, Close as DeleteIcon, ListAlt as StepsIcon } from "@mui/icons-material";
import type { BuilderApi, Evidence, SopData, StepData, SopView } from "../../builderApi";
import { ConfirmDialog } from "../../ui/ConfirmDialog";
import { fieldSx } from "../../ui/dialogStyles";
import { FormBanner, FormField } from "../../ui/dialogKit";
import { EmptyNote, Panel, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";

type Props = { api: BuilderApi; view: SopView; data: SopData; onChange: (d: SopData) => void; flush?: () => Promise<void> };

type GroupKind = "any" | "sequence";

/** Which group each step is in (index) and each group's kind, from the draft's `groups`. */
function groupsOf(data: SopData): { of: Record<string, number>; kinds: GroupKind[] } {
  const of: Record<string, number> = {};
  const kinds: GroupKind[] = [];
  (data.groups ?? []).forEach((g, i) => {
    const ids = Array.isArray(g) ? g : g.sequence;
    kinds.push(Array.isArray(g) ? "any" : "sequence");
    ids.forEach((id) => { of[id] = i; });
  });
  return { of, kinds };
}

/** Groups rebuilt from per-step group numbers; a sequence group runs in the steps' list order. */
function buildGroups(steps: StepData[], of: Record<string, number>, kinds: GroupKind[]): SopData["groups"] {
  return kinds.map((kind, i) => {
    const ids = steps.filter((s) => (of[s.id] ?? 0) === i).map((s) => s.id);
    return kind === "sequence" ? { sequence: ids } : ids;
  });
}

/** The steps with `id` moved to `position` (1-based) among the steps of its group; the others keep their places. */
export function moveWithinGroup(steps: StepData[], of: Record<string, number>, id: string, position: number): StepData[] {
  const g = of[id] ?? 0;
  const slots = steps.map((s, i) => [s, i] as const).filter(([s]) => (of[s.id] ?? 0) === g);
  const members = slots.map(([s]) => s).filter((s) => s.id !== id);
  members.splice(Math.max(0, Math.min(position - 1, members.length)), 0, steps.find((s) => s.id === id)!);
  const next = [...steps];
  slots.forEach(([, i], k) => { next[i] = members[k]; });
  return next;
}

const STATE_LABELS: Record<Evidence, string> = { installed: "has a part installed", empty: "is empty", torqued: "has been torqued" };
const DECISIONS: { key: string; label: string; help: string; step: number; fallback: number }[] = [
  { key: "accept_confidence", label: "How sure the model must be before a step counts (0-1)", help: "Higher: fewer wrong ticks, slower ones.", step: 0.05, fallback: 0.8 },
  { key: "hold_seconds", label: "How long it must stay that sure (s)", help: "Rides out a single odd answer.", step: 0.5, fallback: 1 },
  { key: "evidence_only_after", label: "How long the camera must see the evidence (s)", help: "A part must sit in its slot this long before the slot counts as filled.", step: 0.5, fallback: 3 },
  { key: "hands_off_seconds", label: "How long hands must be away (s)", help: "For steps marked 'hands off': the work piece must be left alone this long.", step: 0.1, fallback: 0.5 },
];

/** The steps, what completes each one, and the order they must happen in. */
export function StepsStage({ api, view, data, onChange, flush }: Props) {
  const slots = Object.keys(data.layout.slots ?? {});
  const steps = data.steps ?? [];
  const { of, kinds } = groupsOf(data);
  const ro = !view.draft;
  const [asking, setAsking] = useState<null | "generate" | number>(null);
  const [error, setError] = useState("");

  const setSteps = (next: StepData[], renamed?: [string, string]) => {
    const map = { ...of };
    if (renamed) { map[renamed[1]] = map[renamed[0]] ?? 0; delete map[renamed[0]]; }
    onChange({ ...data, steps: next, groups: data.order === "groups" ? buildGroups(next, map, kinds) : data.groups });
  };
  const update = (i: number, patch: Partial<StepData>) => {
    const next = steps.map((s, j) => (j === i ? { ...s, ...patch } : s));
    setSteps(next, patch.id !== undefined ? [steps[i].id, patch.id] : undefined);
  };
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= steps.length) return;
    const next = [...steps];
    [next[i], next[j]] = [next[j], next[i]];
    setSteps(next);
  };
  const setOrder = (order: SopData["order"]) => {
    onChange({ ...data, order, groups: order === "groups" ? (data.groups?.length ? data.groups : [steps.map((s) => s.id)]) : null });
  };
  const setGroupOf = (id: string, g: number) => onChange({ ...data, groups: buildGroups(steps, { ...of, [id]: g }, kinds) });
  const setKind = (g: number, kind: GroupKind) => onChange({ ...data, groups: buildGroups(steps, of, kinds.map((k, i) => (i === g ? kind : k))) });
  const addGroup = () => onChange({ ...data, groups: [...(buildGroups(steps, of, kinds) ?? []), []] });
  const generate = async () => {
    setError("");
    await flush?.();  // the server generates from the saved slots
    const g = await api.generateSteps(view.id);
    onChange({ ...data, ...g });
    setAsking(null);
  };
  const askGenerate = () => {
    if (steps.length) { setAsking("generate"); return; }
    generate().catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };
  const setDecision = (key: string, value: string) => {
    const d = { ...(data.decisions ?? {}) };
    if (value === "") delete d[key]; else d[key] = Number(value);
    onChange({ ...data, decisions: d });
  };
  const evidenceSlot = (s: StepData) => Object.keys(s.evidence ?? {})[0] ?? "";
  const evidenceState = (s: StepData): Evidence => Object.values(s.evidence ?? {})[0] ?? "installed";
  const label = (s: StepData) => s.label || s.id;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {error && <FormBanner tone="bad" onClose={() => setError("")}>{error}</FormBanner>}
      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
        <Tooltip title={slots.length === 0 ? "Add slots first (the Slots stage)" : "A step to place a part in each slot, then torque steps when there is a tool"}>
          <span><Button variant="contained" disableElevation sx={plantBtn} startIcon={<GenerateIcon />} disabled={ro || slots.length === 0} onClick={askGenerate}>Generate from slots</Button></span>
        </Tooltip>
        <Button variant="outlined" sx={plantBtn} startIcon={<AddIcon />} disabled={ro}
          onClick={() => setSteps([...steps, { id: `step_${steps.length + 1}`, text: "", evidence: {} }])}>Add step</Button>
        <Box sx={{ ml: "auto", minWidth: 280 }}>
          <TextField select size="small" fullWidth value={data.order} disabled={ro} sx={fieldSx}
            slotProps={{ select: { native: true }, htmlInput: { "aria-label": "Order" } }}
            onChange={(e) => setOrder(e.target.value as SopData["order"])}>
            <option value="strict">Steps happen in the list order</option>
            <option value="any">Steps happen in any order</option>
            <option value="groups">Steps happen in groups, one group after another</option>
          </TextField>
        </Box>
      </Box>

      {data.order === "groups" && (
        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" }, gap: 1.5 }}>
          {kinds.map((k, g) => {
            const members = steps.filter((s) => (of[s.id] ?? 0) === g);
            return (
              <Box component="section" key={g} aria-label={`Group ${g + 1}: ${k === "sequence" ? "in this exact order" : "in any order"}`}
                sx={{ border: `1px solid ${plantColor.line}`, bgcolor: plantColor.surface }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 1.5, py: 0.75, borderBottom: `1px solid ${plantColor.line}`, bgcolor: plantColor.header }}>
                  <Typography sx={{ fontSize: 13.5, fontWeight: 700, flex: 1 }}>Group {g + 1}</Typography>
                  <TextField select size="small" value={k} disabled={ro} sx={{ ...fieldSx, minWidth: 190 }}
                    slotProps={{ select: { native: true }, htmlInput: { "aria-label": `Group ${g + 1} kind` } }}
                    onChange={(e) => setKind(g, e.target.value as GroupKind)}>
                    <option value="any">In any order</option>
                    <option value="sequence">In this exact order</option>
                  </TextField>
                </Box>
                <Box component="ol" sx={{ m: 0, p: 1.5, pl: 1.5, listStyle: "none", display: "flex", flexDirection: "column", gap: 0.75 }}>
                  {members.length === 0 && <Typography sx={{ fontSize: 12.5, color: plantColor.muted }}>No steps yet: pick this group in a step's Group column.</Typography>}
                  {members.map((s, pos) => (
                    <Box component="li" key={s.id} sx={{ display: "flex", alignItems: "center", gap: 1, fontSize: 13 }}>
                      {k === "sequence" ? (
                        <TextField select size="small" value={pos + 1} disabled={ro} sx={{ ...fieldSx, width: 72 }}
                          slotProps={{ select: { native: true }, htmlInput: { "aria-label": `Position of ${s.id}` } }}
                          onChange={(e) => setSteps(moveWithinGroup(steps, of, s.id, Number(e.target.value)))}>
                          {members.map((_, n) => <option key={n} value={n + 1}>{n + 1}</option>)}
                        </TextField>
                      ) : <Box component="span" aria-hidden sx={{ width: 6, height: 6, bgcolor: plantColor.faint, mx: 1 }} />}
                      <span>{label(s)}</span>
                    </Box>
                  ))}
                </Box>
                {k === "sequence" && <Typography sx={{ fontSize: 12, color: plantColor.muted, px: 1.5, pb: 1 }}>Change a position to reorder; the others shift.</Typography>}
              </Box>
            );
          })}
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Button variant="text" sx={plantBtn} startIcon={<AddIcon />} disabled={ro} onClick={addGroup}>Add group</Button>
          </Box>
        </Box>
      )}

      <Panel title="Steps" subtitle="Each one completes when its slot shows the evidence" icon={<StepsIcon />} flush={steps.length > 0}>
        {steps.length === 0 ? (
          <EmptyNote icon={<StepsIcon />} title="No steps yet">Generate them from the slots, then edit the wording; or add them one by one.</EmptyNote>
        ) : (
          <Table size="small" aria-label="Steps">
            <TableHead><TableRow><TableCell /><TableCell>Label</TableCell><TableCell>Instruction</TableCell><TableCell>Completes when the slot…</TableCell>
              <TableCell>Hands off</TableCell>{data.order === "groups" && <TableCell>Group</TableCell>}<TableCell /></TableRow></TableHead>
            <TableBody>
              {steps.map((s, i) => (
                <TableRow key={i}>
                  <TableCell sx={{ whiteSpace: "nowrap", px: 0.5 }}>
                    <IconButton size="small" aria-label={`Move ${s.id} up`} disabled={ro || i === 0} onClick={() => move(i, -1)}><UpIcon fontSize="small" /></IconButton>
                    <IconButton size="small" aria-label={`Move ${s.id} down`} disabled={ro || i === steps.length - 1} onClick={() => move(i, 1)}><DownIcon fontSize="small" /></IconButton>
                  </TableCell>
                  <TableCell sx={{ minWidth: 120 }}><TextField size="small" fullWidth value={s.label ?? ""} disabled={ro} sx={fieldSx} placeholder={s.id}
                    slotProps={{ htmlInput: { "aria-label": `Label of ${s.id}` } }} onChange={(e) => update(i, { label: e.target.value || undefined })} /></TableCell>
                  <TableCell sx={{ minWidth: 240, width: "40%" }}><TextField size="small" fullWidth value={s.text} disabled={ro} sx={fieldSx}
                    slotProps={{ htmlInput: { "aria-label": `Instruction of ${s.id}` } }} onChange={(e) => update(i, { text: e.target.value })} /></TableCell>
                  <TableCell sx={{ whiteSpace: "nowrap" }}>
                    <Box sx={{ display: "flex", gap: 0.5 }}>
                      <TextField select size="small" value={evidenceSlot(s)} disabled={ro} sx={{ ...fieldSx, minWidth: 110 }}
                        slotProps={{ select: { native: true }, htmlInput: { "aria-label": `Evidence slot of ${s.id}` } }}
                        onChange={(e) => update(i, { evidence: e.target.value ? { [e.target.value]: evidenceState(s) } : {} })}>
                        <option value="">no slot</option>
                        {slots.map((sl) => <option key={sl} value={sl}>{sl}</option>)}
                      </TextField>
                      <TextField select size="small" value={evidenceState(s)} disabled={ro || !evidenceSlot(s)} sx={{ ...fieldSx, minWidth: 170 }}
                        slotProps={{ select: { native: true }, htmlInput: { "aria-label": `Evidence state of ${s.id}` } }}
                        onChange={(e) => update(i, { evidence: { [evidenceSlot(s)]: e.target.value as Evidence } })}>
                        {(Object.keys(STATE_LABELS) as Evidence[]).map((k) => <option key={k} value={k}>{STATE_LABELS[k]}</option>)}
                      </TextField>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Tooltip title="The step only counts once hands have left the work piece (e.g. a cover closed and let go)">
                      <Checkbox size="small" checked={!!s.hands_off} disabled={ro} slotProps={{ input: { "aria-label": `Hands off ${s.id}` } }}
                        onChange={(e) => update(i, { hands_off: e.target.checked || undefined })} />
                    </Tooltip>
                  </TableCell>
                  {data.order === "groups" && (
                    <TableCell><TextField select size="small" value={of[s.id] ?? 0} disabled={ro} sx={{ ...fieldSx, width: 72 }}
                      slotProps={{ select: { native: true }, htmlInput: { "aria-label": `Group of ${s.id}` } }}
                      onChange={(e) => setGroupOf(s.id, Number(e.target.value))}>
                      {kinds.map((_, g) => <option key={g} value={g}>{g + 1}</option>)}
                    </TextField></TableCell>
                  )}
                  <TableCell sx={{ px: 0.5 }}><IconButton size="small" aria-label={`Delete ${s.id}`} disabled={ro} onClick={() => setAsking(i)}><DeleteIcon fontSize="small" /></IconButton></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Panel>

      <Box component="details" sx={{ border: `1px solid ${plantColor.line}`, bgcolor: plantColor.surface, px: 1.5, py: 1 }}>
        <Box component="summary" sx={{ cursor: "pointer", fontSize: 13.5, fontWeight: 700 }}>Advanced</Box>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 1.5 }}>
          <Typography sx={{ fontSize: 12.5, color: plantColor.muted }}>
            Step ids name the steps in the recordings' timelines: change one and the timelines that use it must be labelled again.
          </Typography>
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(3, minmax(0, 1fr))" }, gap: 1 }}>
            {steps.map((s, i) => (
              <FormField key={i} label={label(s)} htmlFor={`step-id-${i}`}>
                <TextField id={`step-id-${i}`} size="small" value={s.id} disabled={ro} sx={fieldSx}
                  slotProps={{ htmlInput: { "aria-label": `Id of step ${i + 1}`, style: { fontFamily: "ui-monospace, Menlo, monospace" } } }}
                  onChange={(e) => update(i, { id: e.target.value })} />
              </FormField>
            ))}
          </Box>
          <FormField label="Off-SOP means" htmlFor="off-sop">
            <TextField id="off-sop" size="small" fullWidth value={data.off_sop} disabled={ro} sx={fieldSx} onChange={(e) => onChange({ ...data, off_sop: e.target.value })} />
          </FormField>
          <FormField label="Idle means" htmlFor="idle">
            <TextField id="idle" size="small" fullWidth value={data.idle} disabled={ro} sx={fieldSx} onChange={(e) => onChange({ ...data, idle: e.target.value })} />
          </FormField>
          <Typography sx={{ fontSize: 13, fontWeight: 700 }}>When a step counts as done</Typography>
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))" }, gap: 2 }}>
            {DECISIONS.map((d) => (
              <FormField key={d.key} label={d.label} htmlFor={`decision-${d.key}`}>
                <TextField id={`decision-${d.key}`} type="number" size="small" value={data.decisions?.[d.key] ?? ""} placeholder={String(d.fallback)} disabled={ro} sx={fieldSx}
                  slotProps={{ htmlInput: { step: d.step, min: 0 } }} helperText={d.help} onChange={(e) => setDecision(d.key, e.target.value)} />
              </FormField>
            ))}
          </Box>
        </Box>
      </Box>

      <ConfirmDialog open={asking === "generate"} tone="accent" title="Replace the steps?" actionLabel="Replace the steps"
        body="The steps, their order and groups are replaced by ones generated from the slots. Edits made to the current steps are lost."
        onClose={() => setAsking(null)} onConfirm={generate} />
      <ConfirmDialog open={typeof asking === "number"} title={`Delete ${typeof asking === "number" ? label(steps[asking] ?? { id: "", text: "" }) : ""}?`}
        body="The step goes from the SOP and its group. Timelines that use it must be labelled again."
        onClose={() => setAsking(null)} onConfirm={() => { if (typeof asking === "number") setSteps(steps.filter((_, j) => j !== asking)); setAsking(null); }} />
    </Box>
  );
}
