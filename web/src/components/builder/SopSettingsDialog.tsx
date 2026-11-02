import { useEffect, useState } from "react";
import { Box, Button, TextField, Typography } from "@mui/material";
import { Settings as SettingsIcon } from "@mui/icons-material";
import type { BuilderApi, SopData, SopListEntry } from "../../builderApi";
import { dialogBtn, fieldSx } from "../../ui/dialogStyles";
import { FormDialog, FormField } from "../../ui/dialogKit";
import { plantColor } from "../../ui/kit";

type Props = { open: boolean; api: BuilderApi; data: SopData; onClose: () => void; onChange: (d: SopData) => void };

/** The SOP's name, its description, and which recordings folder it works from. */
export function SopSettingsDialog({ open, api, data, onClose, onChange }: Props) {
  const [name, setName] = useState(data.name ?? "");
  const [description, setDescription] = useState(data.description ?? "");
  const [dir, setDir] = useState(data.dataset?.dir ?? data.sop);
  const [sops, setSops] = useState<SopListEntry[]>([]);
  useEffect(() => {
    if (!open) return;
    setName(data.name ?? "");
    setDescription(data.description ?? "");
    setDir(data.dataset?.dir ?? data.sop);
    api.list().then(setSops).catch(() => setSops([]));
  }, [open, api, data]);
  const own = data.sop;
  const others = sops.filter((s) => s.id !== data.sop && s.dataset_dir && s.dataset_dir !== own);
  const apply = () => {
    onChange({ ...data, name: name.trim() || data.sop, description: description.trim() || name.trim() || data.sop,
      dataset: { ...(data.dataset ?? { format: "timeline" }), dir } });
    onClose();
  };
  return (
    <FormDialog open={open} onClose={onClose} icon={<SettingsIcon />} title="SOP settings" maxWidth="sm"
      actions={
        <>
          <Button sx={{ ...dialogBtn, color: plantColor.muted }} onClick={onClose}>Cancel</Button>
          <Button variant="contained" disableElevation sx={{ ...dialogBtn, minWidth: 96 }} onClick={apply}>Apply</Button>
        </>
      }>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <FormField label="Name" htmlFor="sop-name">
          <TextField id="sop-name" size="small" fullWidth value={name} onChange={(e) => setName(e.target.value)} sx={fieldSx} />
        </FormField>
        <FormField label="Description" htmlFor="sop-description">
          <TextField id="sop-description" size="small" fullWidth multiline minRows={2} value={description} onChange={(e) => setDescription(e.target.value)} sx={fieldSx}
            helperText="What the procedure is, in a sentence: the decision model reads it." />
        </FormField>
        <FormField label="Recordings" htmlFor="sop-recordings">
          <TextField id="sop-recordings" select size="small" fullWidth value={dir} onChange={(e) => setDir(e.target.value)} sx={fieldSx}
            slotProps={{ select: { native: true } }}
            helperText="Another SOP's recordings can be reused, for a new version of the same procedure. Timelines saved here then change that SOP's labels too.">
            <option value={own}>Its own folder (datasets/{own})</option>
            {others.map((s) => <option key={s.id} value={s.dataset_dir}>{s.name} — {s.runs ?? 0} recordings (datasets/{s.dataset_dir})</option>)}
          </TextField>
        </FormField>
        <Typography sx={{ fontSize: 12, color: plantColor.muted }}>The SOP's id, <code>{data.sop}</code>, can't change.</Typography>
      </Box>
    </FormDialog>
  );
}
