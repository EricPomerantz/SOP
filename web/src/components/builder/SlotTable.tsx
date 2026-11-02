import { useState } from "react";
import { Box, Button, Checkbox, FormControlLabel, IconButton, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from "@mui/material";
import { Close as DeleteIcon, DoneAll as AllIcon } from "@mui/icons-material";
import type { SlotData, SopData } from "../../builderApi";
import { ConfirmDialog } from "../../ui/ConfirmDialog";
import { fieldSx } from "../../ui/dialogStyles";
import { FormBanner } from "../../ui/dialogKit";
import { EmptyNote, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";

type Props = { data: SopData; readOnly: boolean; onChange: (d: SopData) => void; selected?: string; onSelect?: (s: string) => void };

export const partClasses = (d: SopData) => Object.entries(d.detector.roles ?? {}).filter(([, r]) => r === "part").map(([c]) => c);
export const emptyClasses = (d: SopData) => Object.entries(d.detector.roles ?? {}).filter(([, r]) => r === "empty").map(([c]) => c);
const fills = (s: SlotData) => (Array.isArray(s.filled_by) ? s.filled_by : [s.filled_by]);
const mono = { fontFamily: "ui-monospace, Menlo, monospace" };

/** The slots: name, what fills each one, its empty marker, its position on the anchor (0-1) and its radius. */
export function SlotTable({ data, readOnly, onChange, selected, onSelect }: Props) {
  const slots = data.layout.slots ?? {};
  const parts = partClasses(data);
  const empties = emptyClasses(data);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [problem, setProblem] = useState("");
  const setSlots = (next: Record<string, SlotData>) => onChange({ ...data, layout: { ...data.layout, slots: next } });
  const patch = (name: string, p: Partial<SlotData>) => setSlots({ ...slots, [name]: { ...slots[name], ...p } });
  const rename = (from: string, to: string): boolean => {
    if (to === from) return true;
    if (!to) { setProblem("A slot needs a name."); return false; }
    if (slots[to]) { setProblem(`${to} is already a slot: pick another name.`); return false; }
    setProblem("");
    const steps = data.steps.map((s) => (s.evidence && from in s.evidence
      ? { ...s, evidence: Object.fromEntries(Object.entries(s.evidence).map(([k, v]) => [k === from ? to : k, v])) } : s));
    onChange({ ...data, steps, layout: { ...data.layout, slots: Object.fromEntries(Object.entries(slots).map(([k, v]) => [k === from ? to : k, v])) } });
    return true;
  };
  const remove = (name: string) => {
    const { [name]: _gone, ...rest } = slots;
    const steps = data.steps.map((s) => (s.evidence && name in s.evidence
      ? { ...s, evidence: Object.fromEntries(Object.entries(s.evidence).filter(([k]) => k !== name)) } : s));
    onChange({ ...data, steps, layout: { ...data.layout, slots: rest } });
  };
  const toggleFill = (name: string, cls: string) => {
    const cur = fills(slots[name]).filter(Boolean);
    const next = cur.includes(cls) ? cur.filter((c) => c !== cls) : [...cur, cls];
    patch(name, { filled_by: next.length === 1 ? next[0] : next });
  };
  const allowAll = () => setSlots(Object.fromEntries(Object.entries(slots).map(([k, v]) => [k, { ...v, filled_by: parts.length === 1 ? parts[0] : [...parts] }])));
  const num = (v: string) => Math.min(1, Math.max(0, Number(v)));
  const usedBy = (name: string) => data.steps.filter((s) => s.evidence && name in s.evidence).map((s) => s.label || s.id);
  const names = Object.keys(slots);

  return (
    <Box>
      {problem && <FormBanner tone="bad" onClose={() => setProblem("")}>{problem}</FormBanner>}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
        <Typography sx={{ fontSize: 12.5, color: plantColor.muted, flex: 1 }}>
          Name the slots as the procedure numbers them (hole_1, hole_2, …): the steps are named after them.
        </Typography>
        <Button size="small" variant="outlined" sx={plantBtn} startIcon={<AllIcon />} disabled={readOnly || !names.length || !parts.length} onClick={allowAll}>
          Allow every part in every slot
        </Button>
      </Box>
      {names.length === 0 ? (
        <EmptyNote compact title="No slots yet">Suggest them from the recordings, or add one and drag it into place.</EmptyNote>
      ) : (
        <Table size="small" aria-label="Slots">
          <TableHead><TableRow><TableCell>Slot</TableCell><TableCell>Filled by</TableCell><TableCell>Empty marker</TableCell>
            <TableCell>Across</TableCell><TableCell>Down</TableCell><TableCell>Radius</TableCell><TableCell /></TableRow></TableHead>
          <TableBody>
            {Object.entries(slots).map(([name, s]) => (
              <TableRow key={name} selected={selected === name} onClick={() => onSelect?.(name)}>
                <TableCell sx={{ minWidth: 120 }}>
                  <TextField size="small" defaultValue={name} disabled={readOnly} sx={fieldSx}
                    slotProps={{ htmlInput: { "aria-label": `Name of ${name}`, style: mono } }}
                    onBlur={(e) => { if (!rename(name, e.target.value.trim())) e.target.value = name; }}
                    onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
                </TableCell>
                <TableCell>
                  {parts.length === 0 ? <Typography sx={{ fontSize: 12.5, color: plantColor.muted }}>Mark part classes in the Detector stage</Typography>
                    : parts.map((c) => (
                      <FormControlLabel key={c} sx={{ mr: 1, "& .MuiFormControlLabel-label": { fontSize: 12.5 } }} label={c}
                        control={<Checkbox size="small" checked={fills(s).includes(c)} disabled={readOnly} onChange={() => toggleFill(name, c)} />} />
                    ))}
                </TableCell>
                <TableCell>
                  <TextField select size="small" value={s.empty_as ?? ""} disabled={readOnly} sx={{ ...fieldSx, minWidth: 150 }}
                    slotProps={{ select: { native: true }, htmlInput: { "aria-label": `Empty marker of ${name}` } }}
                    onChange={(e) => patch(name, { empty_as: e.target.value || null })}>
                    <option value="">none (unseen = unknown)</option>
                    {empties.map((c) => <option key={c} value={c}>{c}</option>)}
                  </TextField>
                </TableCell>
                <TableCell><TextField type="number" size="small" value={s.at[0]} disabled={readOnly} sx={{ ...fieldSx, width: 88 }}
                  slotProps={{ htmlInput: { "aria-label": `x of ${name}`, step: 0.01, min: 0, max: 1 } }}
                  onChange={(e) => patch(name, { at: [num(e.target.value), s.at[1]] })} /></TableCell>
                <TableCell><TextField type="number" size="small" value={s.at[1]} disabled={readOnly} sx={{ ...fieldSx, width: 88 }}
                  slotProps={{ htmlInput: { "aria-label": `y of ${name}`, step: 0.01, min: 0, max: 1 } }}
                  onChange={(e) => patch(name, { at: [s.at[0], num(e.target.value)] })} /></TableCell>
                <TableCell><TextField type="number" size="small" value={s.match_radius ?? ""} placeholder={String(data.layout.match_radius)} disabled={readOnly} sx={{ ...fieldSx, width: 88 }}
                  slotProps={{ htmlInput: { "aria-label": `Radius of ${name}`, step: 0.01, min: 0, max: 1 } }}
                  onChange={(e) => patch(name, { match_radius: e.target.value === "" ? null : num(e.target.value) })} /></TableCell>
                <TableCell sx={{ px: 0.5 }}>
                  <IconButton size="small" aria-label={`Delete ${name}`} disabled={readOnly} onClick={(e) => { e.stopPropagation(); setDeleting(name); }}><DeleteIcon fontSize="small" /></IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      <ConfirmDialog open={deleting !== null} title={`Delete ${deleting ?? ""}?`} onClose={() => setDeleting(null)}
        onConfirm={() => { if (deleting) remove(deleting); setDeleting(null); }}
        body={deleting && usedBy(deleting).length
          ? <>These steps are completed by this slot and lose their evidence: <strong>{usedBy(deleting).join(", ")}</strong>. Give them another slot in the Steps stage.</>
          : "No step uses it."} />
    </Box>
  );
}

/** A new slot at (x, y) with the draft's parts and first empty marker. */
export function addSlot(data: SopData, x = 0.5, y = 0.5): { data: SopData; name: string } {
  const slots = data.layout.slots ?? {};
  let n = Object.keys(slots).length + 1;
  while (slots[`slot_${n}`]) n += 1;
  const parts = partClasses(data);
  const slot: SlotData = { at: [Number(x.toFixed(3)), Number(y.toFixed(3))], filled_by: parts.length === 1 ? parts[0] : parts,
    empty_as: emptyClasses(data)[0] ?? null };
  return { data: { ...data, layout: { ...data.layout, slots: { ...slots, [`slot_${n}`]: slot } } }, name: `slot_${n}` };
}
