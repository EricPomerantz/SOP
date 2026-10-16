import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { builderApi, type BuilderApi, type Job, type TimelineView } from "../../builderApi";
import { normalize, problem, removeRow, rowAt, setBoundary, setStep, splitAt, type Row } from "../../timelineEdit";
import { JobPanel } from "./JobPanel";

type Props = { sopId: string; run: string; api?: BuilderApi };

const PALETTE = ["#2f7cf6", "#30a46c", "#f5a524", "#8e4ec6", "#12a594", "#d6409f", "#978365", "#0090ff", "#46a758", "#e54666"];
const colour = (step: string, ids: string[]) =>
  step === "idle" ? "#9aa1ad" : step === "off_sop" ? "#e5484d" : PALETTE[Math.max(0, ids.indexOf(step)) % PALETTE.length];
const KEYS = "123456789qwertyu";
const SPEEDS = [0.25, 0.5, 1, 1.5, 2];
const WINDOW = 20; // seconds shown in the zoomed bar

const fmt = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, "0")}`;

/** A bar of the timeline between `from` and `to` seconds: segments, draggable boundaries, torque ticks, playhead. */
function Bar({ rows, from, to, t, ids, torque, selected, label, onSeek, onSelect, onBoundary, tall, aria }: {
  rows: Row[]; from: number; to: number; t: number; ids: string[]; torque: TimelineView["torque"]; selected: number | null;
  label: (s: string) => string; onSeek: (t: number) => void; onSelect: (i: number) => void;
  onBoundary: (i: number, t: number) => void; tall?: boolean; aria: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const span = Math.max(to - from, 0.001);
  const x = (time: number) => `${((time - from) / span) * 100}%`;
  const timeAt = (e: ReactPointerEvent) => {
    const box = ref.current?.getBoundingClientRect();
    return box && box.width ? from + ((e.clientX - box.left) / box.width) * span : t;
  };
  const visible = rows.map((r, i) => [r, i] as const).filter(([[, a, b]]) => b > from && a < to);
  return (
    <div ref={ref} className={`label-bar ${tall ? "label-bar-tall" : ""}`} aria-label={aria}
      onPointerDown={(e) => { if (drag === null) onSeek(timeAt(e)); }}
      onPointerMove={(e) => { if (drag !== null) onBoundary(drag, timeAt(e)); }}
      onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
      {visible.map(([[s, a, b], i]) => (
        <div key={i} className={`label-seg ${selected === i ? "label-seg-on" : ""}`} title={`${label(s)} ${a}-${b} s`}
          style={{ left: x(Math.max(a, from)), width: `${((Math.min(b, to) - Math.max(a, from)) / span) * 100}%`, background: colour(s, ids) }}
          onPointerDown={() => onSelect(i)}>
          {tall && <span className="label-seg-text">{label(s)}</span>}
        </div>
      ))}
      {visible.filter(([[, , b], i]) => i < rows.length - 1 && b < to).map(([[, , b], i]) => (
        <div key={`h${i}`} className="label-handle" style={{ left: x(b) }} aria-label={`Boundary ${i + 1}`}
          onPointerDown={(e) => { e.stopPropagation(); setDrag(i); }} />
      ))}
      {torque.filter((p) => p.time_s >= from && p.time_s <= to).map((p) => (
        <div key={p.time_s} className={`label-torque ${p.ok ? "" : "label-torque-bad"}`} style={{ left: x(p.time_s) }}
          title={`torque ${p.ok ? "reached" : "not reached"} at ${p.time_s} s`} />
      ))}
      <div className="label-playhead" style={{ left: x(t) }} />
    </div>
  );
}

/**
 * Label one recorded run's step timeline while watching it (#/sops/<id>/runs/<run>). Play the video (slowed down if
 * it helps) and press the key of the step that starts the moment it starts; it runs until the next press. The
 * zoomed bar follows the playhead for fine-tuning boundaries by drag or frame-by-frame. Saves steps.csv.
 */
export function TimelinePage({ sopId, run, api = builderApi }: Props) {
  const [view, setView] = useState<TimelineView | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [history, setHistory] = useState<Row[][]>([]);
  const [dirty, setDirty] = useState(false);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const video = useRef<HTMLVideoElement>(null);

  const load = useCallback(() => api.timeline(sopId, run).then((v) => {
    setView(v);
    setRows((cur) => (cur.length ? cur : normalize((v.steps ?? v.draft ?? [["idle", 0, v.duration]]).map((r) => [...r] as Row))));
  }).catch((e) => setError(e.message)), [api, sopId, run]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (video.current) video.current.playbackRate = speed; }, [speed, view?.preview]);

  // While playing, follow the video every animation frame: the playhead and the zoomed bar keep up with it.
  useEffect(() => {
    if (!playing) return;
    let id = 0;
    const tick = () => { if (video.current) setT(video.current.currentTime); id = requestAnimationFrame(tick); };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing]);

  const steps = view ? [...view.step_ids, "idle", "off_sop"] : [];
  const label = (s: string) => (s === "idle" ? "Idle" : s === "off_sop" ? "Off-SOP" : view?.labels[s] ?? s);
  const duration = view?.duration || rows[rows.length - 1]?.[2] || 1;
  const now = () => video.current?.currentTime ?? t;  // the frame on screen, not the last rendered state
  const edit = (next: Row[]) => {
    setHistory((h) => [...h.slice(-99), rows]);
    setRows(next);
    setDirty(true);
    setNotice("");
  };
  const undo = () => {
    if (!history.length) return;
    setRows(history[history.length - 1]);
    setHistory(history.slice(0, -1));
    setDirty(true);
  };
  const seek = (time: number) => {
    const at = Math.min(Math.max(time, 0), duration);
    if (video.current) video.current.currentTime = at;
    setT(at);
  };
  const toggle = () => {
    const v = video.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {}); else v.pause();
  };
  const mark = (step: string) => edit(splitAt(rows, now(), step));
  const boundaries = rows.slice(0, -1).map((r) => r[2]);
  const jump = (dir: 1 | -1) => {
    const cur = now();
    const target = dir > 0 ? boundaries.find((b) => b > cur + 0.01) : [...boundaries].reverse().find((b) => b < cur - 0.01);
    if (target !== undefined) seek(target);
  };
  const nudge = (d: number) => {
    // Move the boundary nearest the playhead by d seconds, and the playhead with it.
    const cur = now();
    let i = -1, best = Infinity;
    boundaries.forEach((b, j) => { if (Math.abs(b - cur) < best) { best = Math.abs(b - cur); i = j; } });
    if (i < 0 || best > 2) return;
    const next = setBoundary(rows, i, boundaries[i] + d);
    edit(next);
    seek(next[i][2]);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      const frame = 1 / (view?.fps || 25);
      if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); undo(); return; }
      if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); save(); return; }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === " " || e.key === "k") { e.preventDefault(); toggle(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); seek(now() - (e.shiftKey ? 1 : frame)); }
      else if (e.key === "ArrowRight") { e.preventDefault(); seek(now() + (e.shiftKey ? 1 : frame)); }
      else if (e.key === "j") seek(now() - 5);
      else if (e.key === "l") seek(now() + 5);
      else if (e.key === "[") jump(-1);
      else if (e.key === "]") jump(1);
      else if (e.key === ",") nudge(-frame);
      else if (e.key === ".") nudge(frame);
      else if (e.key === "-") setSpeed((s) => SPEEDS[Math.max(0, SPEEDS.indexOf(s) - 1)]);
      else if (e.key === "=" || e.key === "+") setSpeed((s) => SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(s) + 1)]);
      else if (e.key === "i") mark("idle");
      else if (e.key === "o") mark("off_sop");
      else if ((e.key === "Delete" || e.key === "Backspace") && selected !== null) { edit(removeRow(rows, selected)); setSelected(null); }
      else {
        const k = KEYS.indexOf(e.key);
        if (k >= 0 && view && k < view.step_ids.length) mark(view.step_ids[k]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const save = async () => {
    setError("");
    try {
      const v = await api.saveTimeline(sopId, run, rows);
      setView(v);
      setDirty(false);
      setNotice("Saved steps.csv");
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const prepare = async () => {
    setError("");
    try { setJobId((await api.startJob("previews", { sop_file: view?.sop_file, runs: [run] })).id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const prepared = (job: Job) => { if (job.state === "done") load(); };

  if (!view) return <main className="app builder">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</main>;
  const issue = problem(rows, view.duration);
  const current = rowAt(rows, t);
  const currentStep = rows[current]?.[0] ?? "";
  const from = Math.max(0, Math.min(t - WINDOW / 2, duration - WINDOW));
  const to = Math.min(duration, from + WINDOW);

  return (
    <main className="app builder timeline-page">
      <header className="topbar card">
        <div className="topbar-title"><span className="name">Label run {run}</span>
          {view.steps ? <span className="pill pill-ok">saved</span> : <span className="pill pill-idle">not saved yet</span>}
          {dirty && <span className="pill pill-warn">unsaved</span>}</div>
        <div className="topbar-right">
          <button onClick={undo} disabled={!history.length}>Undo</button>
          <button onClick={save} disabled={!!issue || (!dirty && !!view.steps)}>Save</button>
          <a className="muted" href={`#/sops/${sopId}`}>Back to the SOP</a>
        </div>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {notice && <p role="status" className="notice">{notice}</p>}
      <div className="label-grid">
        <section className="card label-video">
          {view.preview ? (
            <div className="label-video-wrap">
              <video ref={video} src={api.previewUrl(sopId, run)} preload="auto" aria-label="Run video" onClick={toggle}
                onPlay={() => setPlaying(true)} onPause={() => { setPlaying(false); setT(video.current?.currentTime ?? 0); }}
                onSeeked={(e) => setT(e.currentTarget.currentTime)} onLoadedMetadata={(e) => { e.currentTarget.playbackRate = speed; }} />
              <div className="label-now" style={{ borderLeftColor: colour(currentStep, view.step_ids) }}>{label(currentStep)}</div>
            </div>
          ) : (
            <div className="label-novideo">
              <p className="muted">The recording needs a browser-playable copy first (a minute or two).</p>
              {!jobId && <button onClick={prepare}>Prepare the video</button>}
              <JobPanel api={api} jobId={jobId} poll={1500} onFinished={prepared} />
            </div>
          )}
          <div className="transport">
            <button onClick={() => seek(now() - 5)} aria-label="Back 5 s">⏪</button>
            <button onClick={() => seek(now() - 1 / (view.fps || 25))} aria-label="Previous frame">◀︎</button>
            <button onClick={toggle} aria-label={playing ? "Pause" : "Play"} className="transport-play">{playing ? "❚❚" : "▶"}</button>
            <button onClick={() => seek(now() + 1 / (view.fps || 25))} aria-label="Next frame">▶︎</button>
            <button onClick={() => seek(now() + 5)} aria-label="Forward 5 s">⏩</button>
            <span className="mono">{fmt(t)} / {fmt(duration)}</span>
            <label>Speed <select aria-label="Speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
              {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
            </select></label>
          </div>
        </section>
        <section className="card latency-panel" aria-label="Steps">
          <div className="palette">
            {steps.map((s, k) => (
              <button key={s} onClick={() => mark(s)} style={{ borderLeft: `6px solid ${colour(s, view.step_ids)}` }}
                className={currentStep === s ? "tab-on" : undefined} title={view.labels[s] ? `${s}` : undefined}>
                <kbd>{s === "idle" ? "i" : s === "off_sop" ? "o" : KEYS[k] ?? ""}</kbd> {label(s)}
              </button>
            ))}
          </div>
          <details className="muted keys">
            <summary>How to label</summary>
            <p>Play the run and, the moment a step starts, press its key: that step then runs until the next press.
              Fetching the next part already counts as its step; idle is nobody working, off-SOP is anything that isn't a
              step (a part taken out again, a retorque, fiddling). A torque step ends at the torque press (▼ on the bar).</p>
            <ul>
              <li><kbd>space</kbd>/<kbd>k</kbd> play/pause · <kbd>-</kbd>/<kbd>=</kbd> slower/faster</li>
              <li><kbd>←</kbd>/<kbd>→</kbd> one frame · <kbd>shift</kbd> 1 s · <kbd>j</kbd>/<kbd>l</kbd> 5 s</li>
              <li><kbd>[</kbd>/<kbd>]</kbd> previous/next step change · <kbd>,</kbd>/<kbd>.</kbd> move the nearest change one frame</li>
              <li>Drag a change on either bar · <kbd>Delete</kbd> removes the selected row · <kbd>ctrl</kbd>+<kbd>z</kbd> undo · <kbd>ctrl</kbd>+<kbd>s</kbd> save</li>
            </ul>
          </details>
          <div className="row">
            {view.draft && <button onClick={() => edit(normalize(view.draft!.map((r) => [...r] as Row)))}>Start from the tracker's draft</button>}
            <button onClick={() => edit([["idle", 0, view.duration]])}>Start from scratch</button>
          </div>
          {view.draft_info && (view.draft_info.not_done.length > 0 || view.draft_info.alerts.length > 0) && (
            <details><summary>What the tracker flagged in its draft</summary><ul className="mono">
              {view.draft_info.not_done.length > 0 && <li>never completed: {view.draft_info.not_done.join(", ")}</li>}
              {view.draft_info.alerts.map((a) => <li key={a}>{a}</li>)}
            </ul></details>
          )}
        </section>
      </div>
      <section className="card timeline">
        <div className="muted bar-caption">Around the playhead ({WINDOW} s)</div>
        <Bar rows={rows} from={from} to={to} t={t} ids={view.step_ids} torque={view.torque} selected={selected} label={label}
          onSeek={seek} onSelect={setSelected} onBoundary={(i, time) => edit(setBoundary(rows, i, time))} tall aria="Zoomed timeline" />
        <div className="muted bar-caption">Whole run</div>
        <Bar rows={rows} from={0} to={duration} t={t} ids={view.step_ids} torque={view.torque} selected={selected} label={label}
          onSeek={seek} onSelect={setSelected} onBoundary={(i, time) => edit(setBoundary(rows, i, time))} aria="Timeline bar" />
        {issue && <p className="error">{issue}</p>}
        <table className="grid-table" aria-label="Rows">
          <thead><tr><th>Step</th><th>Start</th><th>End</th><th>Length</th><th></th></tr></thead>
          <tbody>
            {rows.map(([s, a, b], i) => (
              <tr key={i} className={i === current ? "selected" : undefined} onClick={() => setSelected(i)}>
                <td><select aria-label={`Step of row ${i + 1}`} value={s} onChange={(e) => edit(setStep(rows, i, e.target.value))}>
                  {steps.map((x) => <option key={x} value={x}>{label(x)}</option>)}
                </select></td>
                <td><button className="link" onClick={() => seek(a)}>{a.toFixed(2)}</button></td>
                <td>{i < rows.length - 1
                  ? <input aria-label={`End of row ${i + 1}`} type="number" step="0.04" value={b}
                      onChange={(e) => edit(setBoundary(rows, i, Number(e.target.value)))} />
                  : b.toFixed(2)}</td>
                <td className="mono">{(b - a).toFixed(1)} s</td>
                <td><button className="icon" aria-label={`Remove row ${i + 1}`} onClick={() => edit(removeRow(rows, i))}>✕</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
