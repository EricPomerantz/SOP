import { useCallback, useEffect, useState, type ReactNode } from "react";
import { builderApi, type BuilderApi, type SopData, type SopView } from "../../builderApi";
import { DetectorStage } from "./DetectorStage";
import { LayoutStage } from "./LayoutStage";
import { RecordingsStage } from "./RecordingsStage";
import { StepsStage } from "./StepsStage";
import { TimelinesStage } from "./TimelinesStage";
import { TrainStage } from "./TrainStage";

type Props = { sopId: string; api?: BuilderApi; confirm?: (m: string) => boolean; go?: (hash: string) => void };

const STAGES = ["detector", "recordings", "layout", "steps", "timelines", "train"] as const;
type Stage = (typeof STAGES)[number];
const TITLES: Record<Stage, string> = {
  detector: "1 Detector", recordings: "2 Recordings", layout: "3 Slots", steps: "4 Steps",
  timelines: "5 Timelines", train: "6 Decision model",
};

function done(view: SopView, stage: Stage): boolean {
  const s = view.status;
  switch (stage) {
    case "detector": return s.detector && s.roles;
    case "recordings": return s.recordings.train > 0 && s.recordings.val > 0 && s.recordings.test > 0;
    case "layout": return s.layout;
    case "steps": return s.steps;
    case "timelines": return Object.values(s.recordings).reduce((a, b) => a + b, 0) > 0
      && Object.keys(s.recordings).every((k) => s.timelines[k] === s.recordings[k]);
    case "train": return !!s.laya_checkpoint;
  }
}

/** One SOP from detector to decision model (#/sops/<id>): edits a draft, publishes it into the app's SOPs. */
export function SopBuilder({ sopId, api = builderApi, confirm = (m) => window.confirm(m), go = (h) => { window.location.hash = h; } }: Props) {
  const [view, setView] = useState<SopView | null>(null);
  const [data, setData] = useState<SopData | null>(null);
  const [dirty, setDirty] = useState(false);
  const [stage, setStage] = useState<Stage>("detector");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const show = useCallback((v: SopView) => { setView(v); setData(v.data); setDirty(false); }, []);
  const reload = useCallback(() => api.get(sopId).then(show).catch((e) => setError(e.message)), [api, sopId, show]);
  useEffect(() => { reload(); }, [reload]);

  const act = async (action: () => Promise<void>) => {
    setError("");
    setNotice("");
    try { await action(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const save = () => act(async () => { if (data) show(await api.save(sopId, data)); });
  const change = (d: SopData) => { setData(d); setDirty(true); };
  const viewChange = (v: SopView) => show(v);

  if (!view || !data) {
    return <main className="app builder">{error ? <p className="error" role="alert">{error}</p> : <p className="muted">Loading…</p>}</main>;
  }
  const s = view.status;
  const publish = () => act(async () => {
    if (dirty) await api.save(sopId, data);
    show(await api.publish(sopId));
    setNotice("Published: the SOP is in the dashboard's SOP list.");
  });
  const discard = () => act(async () => {
    if (!confirm(s.published ? "Throw away the changes in this draft? The published SOP stays as it is." : "Delete this draft SOP?")) return;
    const r = await api.discard(sopId);
    if ("deleted" in r) go("#/sops"); else show(r);
  });
  const edit = () => act(async () => show(await api.edit(sopId)));

  const stages: Record<Stage, ReactNode> = {
    detector: <DetectorStage api={api} view={view} data={data} onChange={change} onView={viewChange} />,
    recordings: <RecordingsStage view={view} />,
    layout: <LayoutStage api={api} view={view} data={data} onChange={change} />,
    steps: <StepsStage api={api} view={view} data={data} onChange={change} />,
    timelines: <TimelinesStage api={api} view={view} onReload={reload} />,
    train: <TrainStage api={api} view={view} onReload={reload} />,
  };

  return (
    <main className="app builder">
      <header className="topbar card">
        <div className="topbar-title">
          <span className="name">{data.name || view.id}</span>
          <span className="muted mono">{view.id} v{data.version}</span>
          {view.draft ? <span className="pill pill-idle">draft</span> : <span className="pill pill-ok">published</span>}
          {dirty && <span className="pill pill-warn">unsaved</span>}
        </div>
        <div className="topbar-right">
          {view.draft ? (
            <>
              <button onClick={save} disabled={!dirty}>Save</button>
              <button onClick={publish} disabled={s.errors.length > 0 && !dirty}>Publish</button>
              <button className="discard" onClick={discard}>{s.published ? "Discard changes" : "Delete draft"}</button>
            </>
          ) : <button onClick={edit}>Edit</button>}
          <a className="muted" href="#/sops">All SOPs</a>
        </div>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {notice && <p role="status" className="notice">{notice}</p>}
      {view.draft && s.errors.length > 0 && !dirty && (
        <details className="card errors"><summary>{s.errors.length} thing{s.errors.length > 1 ? "s" : ""} to fix before publishing</summary>
          <ul>{s.errors.map((e) => <li key={e}>{e}</li>)}</ul></details>
      )}
      <nav className="tabs card" aria-label="Stages">
        {STAGES.map((st) => (
          <button key={st} className={`tab ${stage === st ? "tab-on" : ""}`} aria-pressed={stage === st} onClick={() => setStage(st)}>
            <span className={done(view, st) ? "step-done" : "muted"}>{done(view, st) ? "✓" : "○"}</span> {TITLES[st]}
          </button>
        ))}
      </nav>
      <section className="card stage-card" aria-label={TITLES[stage]}>
        {view.draft || stage === "recordings" || stage === "timelines" || stage === "train" ? null
          : <p className="muted">Published SOPs are read-only here: press Edit to change a draft copy.</p>}
        {stages[stage]}
      </section>
    </main>
  );
}
