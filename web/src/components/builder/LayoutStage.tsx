import { useEffect, useRef, useState, type MouseEvent } from "react";
import type { AutofitResult, BuilderApi, Job, SlotData, SopData, SopView } from "../../builderApi";
import { JobPanel } from "./JobPanel";
import { addSlot, SlotTable } from "./SlotTable";

type Props = { api: BuilderApi; view: SopView; data: SopData; onChange: (d: SopData) => void; confirm?: (m: string) => boolean };
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
export function LayoutStage({ api, view, data, onChange, confirm = (m) => window.confirm(m) }: Props) {
  const ro = !view.draft;
  const layout = data.layout;
  const [fitResult, setFitResult] = useState<AutofitResult | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.08);
  const [run, setRun] = useState("");
  const [selected, setSelected] = useState<string | undefined>();
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState("");

  const load = () => api.autofit(view.id).then((r) => { setFitResult(r); setRun((cur) => cur || Object.keys(r?.samples ?? {})[0] || ""); }).catch(() => {});
  useEffect(() => {
    load();
    api.jobs(view.file).then((jobs) => {
      const running = jobs.find((j) => j.kind === "autofit" && !["done", "failed", "cancelled"].includes(j.state));
      if (running) setJobId(running.id);
    }).catch(() => {});
  }, [api, view.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const set = (patch: Partial<SopData["layout"]>) => onChange({ ...data, layout: { ...layout, ...patch } });
  const suggest = async () => {
    setError("");
    try { setJobId((await api.startJob("autofit", { sop_file: view.file, threshold })).id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const apply = () => {
    if (!fitResult) return;
    if (Object.keys(layout.slots ?? {}).length && !confirm("Replace the current slots with the suggested ones?")) return;
    const slots: Record<string, SlotData> = {};
    for (const c of fitResult.slots) {
      slots[c.name] = { at: c.at, filled_by: c.filled_by.length === 1 ? c.filled_by[0] : c.filled_by, empty_as: c.empty_as,
        match_radius: Math.abs(c.radius - fitResult.match_radius) > 0.01 ? c.radius : null };
    }
    onChange({ ...data, layout: { ...layout, slots, match_radius: fitResult.match_radius } });
  };
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
    <div className="stage">
      <p className="muted">
        Slot positions are fractions of the anchor's box (0-1 across and down), so they move with it. "Suggest slots"
        runs the detector over the recordings and clusters where parts and empty-slot markers sit; drag the rings to
        adjust, and check them on the other runs' frames below. The ring is the match radius.
      </p>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="row">
        <button disabled={ro || (view.status.runs ?? []).length === 0 || !layout.anchor} onClick={suggest}>Suggest slots</button>
        <label>Cluster distance <input type="number" step="0.01" min="0.01" max="0.5" value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))} /></label>
        {fitResult && <button disabled={ro || fitResult.slots.length === 0} onClick={apply}>Use the {fitResult.slots.length} suggested slots</button>}
        <button disabled={ro} className={adding ? "tab-on" : undefined} onClick={() => (sample ? setAdding(!adding) : onChange(addSlot(data).data))}>
          {adding ? "Click the frame…" : "Add slot"}
        </button>
      </div>
      {!layout.anchor && <p className="muted">Pick the anchor class in the detector stage first.</p>}
      <JobPanel api={api} jobId={jobId} onFinished={finished} />
      {fitResult && (
        <p className="muted">
          Last suggestion: {fitResult.slots.length} slots from {fitResult.frames_used} frames
          {fitResult.frames_dropped ? ` (${fitResult.frames_dropped} dropped: anchor box an unusual size)` : ""}.
          {fitResult.slots.map((c) => ` ${c.name} ±${c.spread.toFixed(3)}`).join(",")}
        </p>
      )}
      {fitResult?.warnings.map((w) => <p key={w} className="pill-warn warn-line">{w}</p>)}
      {sample && (
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
      )}
      <SlotTable data={data} readOnly={ro} onChange={onChange} selected={selected} onSelect={setSelected} />
      <div className="row">
        <label>Default radius <input type="number" step="0.01" min="0.01" max="1" value={layout.match_radius} disabled={ro}
          onChange={(e) => set({ match_radius: Number(e.target.value) })} /></label>
        <label>Anchor follow <input type="number" step="0.05" min="0.05" max="1" placeholder="0.1" value={layout.anchor_follow ?? ""} disabled={ro}
          onChange={(e) => set({ anchor_follow: e.target.value === "" ? undefined : Number(e.target.value) })} /></label>
        <label>Removal wait (s) <input type="number" step="0.5" min="0.5" placeholder="0.5" value={layout.removal_seconds ?? ""} disabled={ro}
          onChange={(e) => set({ removal_seconds: e.target.value === "" ? null : Number(e.target.value) })} /></label>
      </div>
      <p className="muted">
        Anchor follow: how closely the anchor box tracks each detection (1 = every frame, for a shaking camera).
        Removal wait: how long a slot must read empty before its part counts as removed (longer rides out detector
        flicker, but alerts come later).
      </p>
    </div>
  );
}
