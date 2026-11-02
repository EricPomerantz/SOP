import { useEffect, useRef, useState, type MouseEvent } from "react";
import { Box, Button, TextField, Typography } from "@mui/material";
import { AddLocationAlt as AddIcon, AutoFixHigh as SuggestIcon, PlaceOutlined as SlotsIcon } from "@mui/icons-material";
import type { AutofitResult, BuilderApi, Job, SlotData, SopData, SopView } from "../../builderApi";
import { ConfirmDialog } from "../../ui/ConfirmDialog";
import { fieldSx } from "../../ui/dialogStyles";
import { FormBanner, FormField } from "../../ui/dialogKit";
import { useGlobalAlert } from "../../ui/GlobalAlertProvider";
import { EmptyNote, Panel, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";
import { JobPanel } from "./JobPanel";
import { addSlot, SlotTable } from "./SlotTable";

type Props = { api: BuilderApi; view: SopView; data: SopData; onChange: (d: SopData) => void; flush?: () => Promise<void> };
type Sample = AutofitResult["samples"][string];

const COLOURS = ["#2f7cf6", "#e5484d", "#30a46c", "#f5a524", "#8e4ec6", "#12a594", "#d6409f", "#978365"];

/** Slots drawn on one recorded frame: the anchor box, each slot's ring (its radius) and the auto-fit point clouds. */
function FrameView({ src, sample, data, fitResult, selected, onSelect, onMove, onAdd, small }: {
  src: string; sample: Sample; data: SopData; fitResult?: AutofitResult | null; selected?: string;
  onSelect?: (s: string) => void; onMove?: (s: string, at: [number, number]) => void; onAdd?: (at: [number, number]) => void; small?: boolean;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<string | null>(null);
  const [x1, y1, x2, y2] = sample.anchor;
  const w = x2 - x1, h = y2 - y1;
  const px = (at: [number, number]) => [x1 + at[0] * w, y1 + at[1] * h];
  const toAt = (e: MouseEvent): [number, number] | null => {
    const m = svg.current?.getScreenCTM?.();
    if (!svg.current || !m) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    return [Number(((p.x - x1) / w).toFixed(3)), Number(((p.y - y1) / h).toFixed(3))];
  };
  const slots = Object.entries(data.layout.slots ?? {});
  return (
    <svg ref={svg} className={small ? "frame frame-small" : "frame"} viewBox={`0 0 ${sample.width} ${sample.height}`}
      onPointerMove={(e) => { if (drag) { const at = toAt(e); if (at) onMove?.(drag, at); } }}
      onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}
      onClick={(e) => { if (!drag && onAdd && e.target === e.currentTarget.querySelector("image")) { const at = toAt(e); if (at) onAdd(at); } }}>
      <image href={src} width={sample.width} height={sample.height} />
      <rect x={x1} y={y1} width={w} height={h} fill="none" stroke="#fff" strokeDasharray="8 6" strokeWidth={small ? 3 : 1.5} />
      {!small && fitResult?.slots.map((c, i) => c.points.map(([u, v], j) => (
        <circle key={`${i}-${j}`} cx={x1 + u * w} cy={y1 + v * h} r={2} fill={COLOURS[i % COLOURS.length]} opacity={0.5} />
      )))}
      {slots.map(([name, s], i) => {
        const [cx, cy] = px(s.at);
        const r = s.match_radius ?? data.layout.match_radius;
        const on = name === selected;
        return (
          <g key={name} className="slot-ring" onPointerDown={(e) => { e.stopPropagation(); onSelect?.(name); if (onMove) setDrag(name); }}>
            <ellipse cx={cx} cy={cy} rx={r * w} ry={r * h} fill={on ? "rgba(47,124,246,0.15)" : "transparent"}
              stroke={COLOURS[i % COLOURS.length]} strokeWidth={small ? 4 : on ? 3 : 2} />
            <circle cx={cx} cy={cy} r={small ? 6 : 3} fill={COLOURS[i % COLOURS.length]} />
            {!small && <text x={cx + r * w + 4} y={cy + 4} className="slot-label">{name}</text>}
          </g>
        );
      })}
    </svg>
  );
}

/** Slot positions on the anchor's box: suggested from the recordings, adjusted on a frame, checked on the others. */
export function LayoutStage({ api, view, data, onChange, flush }: Props) {
  const alert = useGlobalAlert();
  const ro = !view.draft;
  const layout = data.layout;
  const [fitResult, setFitResult] = useState<AutofitResult | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.08);
  const [run, setRun] = useState("");
  const [selected, setSelected] = useState<string | undefined>();
  const [adding, setAdding] = useState(false);
  const [replacing, setReplacing] = useState(false);
  const [error, setError] = useState("");
  const say = (what: string) => (e: unknown) => alert.error(`${what}: ${e instanceof Error ? e.message : String(e)}`);

  const load = () => api.autofit(view.id)
    .then((r) => { setFitResult(r); setRun((cur) => cur || Object.keys(r?.samples ?? {})[0] || ""); })
    .catch(say("Couldn't load the last slot suggestion"));
  useEffect(() => {
    load();
    api.jobs(view.file).then((jobs) => {
      const running = jobs.find((j) => j.kind === "autofit" && !["done", "failed", "cancelled"].includes(j.state));
      if (running) setJobId(running.id);
    }).catch(say("Couldn't check for a running job"));
  }, [api, view.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const set = (patch: Partial<SopData["layout"]>) => onChange({ ...data, layout: { ...layout, ...patch } });
  const runs = view.status.runs ?? [];
  const suggestBlocker = !layout.anchor ? "Pick the anchor class in the Detector stage first" : runs.length === 0 ? "Record some runs first" : "";
  const suggest = async () => {
    setError("");
    try { await flush?.(); setJobId((await api.startJob("autofit", { sop_file: view.file, threshold })).id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const apply = () => {
    if (!fitResult) return;
    const slots: Record<string, SlotData> = {};
    for (const c of fitResult.slots) {
      slots[c.name] = { at: c.at, filled_by: c.filled_by.length === 1 ? c.filled_by[0] : c.filled_by, empty_as: c.empty_as,
        match_radius: Math.abs(c.radius - fitResult.match_radius) > 0.01 ? c.radius : null };
    }
    onChange({ ...data, layout: { ...layout, slots, match_radius: fitResult.match_radius } });
    setReplacing(false);
  };
  const askApply = () => (Object.keys(layout.slots ?? {}).length ? setReplacing(true) : apply());
  const move = (name: string, at: [number, number]) => {
    const s = layout.slots[name];
    set({ slots: { ...layout.slots, [name]: { ...s, at: [Math.min(1, Math.max(0, at[0])), Math.min(1, Math.max(0, at[1]))] } } });
  };
  const add = (at: [number, number]) => {
    const r = addSlot(data, at[0], at[1]);
    onChange(r.data);
    setSelected(r.name);
    setAdding(false);
  };
  const finished = (job: Job) => { if (job.state === "done") load(); };
  const samples = fitResult?.samples ?? {};
  const sample = samples[run];

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {error && <FormBanner tone="bad" onClose={() => setError("")}>{error}</FormBanner>}
      <Panel title="Suggest slots from the recordings" icon={<SuggestIcon />}
        subtitle="Runs the detector over the recordings and finds where parts and empty-slot markers sit on the anchor">
        <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
          <Button variant="contained" disableElevation sx={plantBtn} startIcon={<SuggestIcon />} disabled={ro || !!suggestBlocker} onClick={() => void suggest()}>Suggest slots</Button>
          {fitResult && <Button variant="outlined" sx={plantBtn} disabled={ro || fitResult.slots.length === 0} onClick={askApply}>Use the {fitResult.slots.length} suggested slots</Button>}
          <Button variant={adding ? "contained" : "outlined"} disableElevation sx={plantBtn} startIcon={<AddIcon />} disabled={ro}
            onClick={() => (sample ? setAdding(!adding) : onChange(addSlot(data).data))}>
            {adding ? "Click the frame…" : "Add slot"}
          </Button>
          {suggestBlocker && <Typography sx={{ fontSize: 12.5, color: plantColor.muted }}>{suggestBlocker}.</Typography>}
        </Box>
        {fitResult && (
          <Typography sx={{ fontSize: 12.5, color: plantColor.muted, mt: 1 }}>
            Last suggestion: {fitResult.slots.length} slots from {fitResult.frames_used} frames
            {fitResult.frames_dropped ? ` (${fitResult.frames_dropped} left out: the anchor looked an unusual size)` : ""}.
          </Typography>
        )}
        {fitResult?.warnings.map((w) => <Box key={w} sx={{ mt: 1 }}><FormBanner tone="warn">{w}</FormBanner></Box>)}
        <Box component="details" sx={{ mt: 1, fontSize: 12.5, color: plantColor.muted }}>
          <summary style={{ cursor: "pointer" }}>Advanced</summary>
          <Box sx={{ mt: 1, maxWidth: 360 }}>
            <FormField label="Cluster distance (anchor widths)" htmlFor="cluster-distance">
              <TextField id="cluster-distance" type="number" size="small" value={threshold} sx={fieldSx}
                slotProps={{ htmlInput: { step: 0.01, min: 0.01, max: 0.5 } }} onChange={(e) => setThreshold(Number(e.target.value))}
                helperText="Detections closer than this are one slot. Smaller splits close slots apart." />
            </FormField>
          </Box>
        </Box>
      </Panel>
      <JobPanel api={api} jobId={jobId} onFinished={finished} />
      <Panel title="Slots on a recorded frame" icon={<SlotsIcon />} subtitle="Drag a ring to move its slot; check it on the other runs at the right. The ring is the slot's radius.">
        {sample ? (
          <div className="layout-editor">
            <FrameView src={api.frameUrl(view.id, run)} sample={sample} data={data} fitResult={fitResult} selected={selected}
              onSelect={setSelected} onMove={ro ? undefined : move} onAdd={adding && !ro ? add : undefined} />
            <div className="frame-strip" aria-label="Other runs">
              {Object.entries(samples).map(([r, s]) => (
                <button key={r} className={`frame-pick ${r === run ? "tab-on" : ""}`} onClick={() => setRun(r)} aria-label={`Show ${r}`}>
                  <FrameView src={api.frameUrl(view.id, r)} sample={s} data={data} small />
                  <span className="mono">{r}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <EmptyNote compact icon={<SlotsIcon />} title="No frame to draw on yet">Frames from the recordings appear here after "Suggest slots".</EmptyNote>
        )}
      </Panel>
      <Panel title="Slots">
        <SlotTable data={data} readOnly={ro} onChange={onChange} selected={selected} onSelect={setSelected} />
      </Panel>
      <Box component="details" sx={{ border: `1px solid ${plantColor.line}`, bgcolor: plantColor.surface, px: 1.5, py: 1 }}>
        <Box component="summary" sx={{ cursor: "pointer", fontSize: 13.5, fontWeight: 700 }}>Advanced</Box>
        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(3, minmax(0, 1fr))" }, gap: 2, mt: 1.5 }}>
          <FormField label="Default radius" htmlFor="default-radius">
            <TextField id="default-radius" type="number" size="small" value={layout.match_radius} disabled={ro} sx={fieldSx}
              slotProps={{ htmlInput: { step: 0.01, min: 0.01, max: 1 } }} onChange={(e) => set({ match_radius: Number(e.target.value) })}
              helperText="How far from a slot's centre a part still counts as in it, in anchor widths." />
          </FormField>
          <FormField label="Anchor follow" htmlFor="anchor-follow">
            <TextField id="anchor-follow" type="number" size="small" placeholder="0.1" value={layout.anchor_follow ?? ""} disabled={ro} sx={fieldSx}
              slotProps={{ htmlInput: { step: 0.05, min: 0.05, max: 1 } }}
              onChange={(e) => set({ anchor_follow: e.target.value === "" ? undefined : Number(e.target.value) })}
              helperText="How closely the slots follow the anchor from frame to frame (1 = every frame, for a shaking camera)." />
          </FormField>
          <FormField label="Removal wait (s)" htmlFor="removal-wait">
            <TextField id="removal-wait" type="number" size="small" placeholder="0.5" value={layout.removal_seconds ?? ""} disabled={ro} sx={fieldSx}
              slotProps={{ htmlInput: { step: 0.5, min: 0.5 } }}
              onChange={(e) => set({ removal_seconds: e.target.value === "" ? null : Number(e.target.value) })}
              helperText="How long a slot must look empty before its part counts as removed. Longer rides out detector flicker; alerts come later." />
          </FormField>
        </Box>
      </Box>
      <ConfirmDialog open={replacing} tone="accent" title="Replace the slots?" actionLabel="Use the suggested slots"
        body="The current slots are replaced by the suggested ones (named slot_1, slot_2, …). Rename them afterwards; steps that used the old names lose their evidence."
        onClose={() => setReplacing(false)} onConfirm={apply} />
    </Box>
  );
}
