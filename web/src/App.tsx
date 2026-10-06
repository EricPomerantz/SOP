import { useCallback, useEffect, useMemo, useState } from "react";
import { connectSocket, httpApi, type Api } from "./api";
import { Checklist } from "./components/Checklist";
import { Controls } from "./components/Controls";
import { EventLog } from "./components/EventLog";
import { JevStrip } from "./components/JevStrip";
import { Timeline } from "./components/Timeline";
import { TopBar } from "./components/TopBar";
import { VideoPanel } from "./components/VideoPanel";
import { type DashboardState, decisionView, initial, reduce } from "./state";
import type { RunInfo, RunSnapshot, Sop, SourceSpec, WsMessage } from "./types";

type Props = { api?: Api; connect?: (onMessage: (m: WsMessage) => void) => () => void };

function fromSnapshot(sop: Sop, snap: RunSnapshot, decider: string): DashboardState {
  const base = initial(sop.steps, decider);
  const byId = new Map(snap.steps.map((s) => [s.id, s]));
  return {
    ...base,
    steps: base.steps.map((s) => ({ ...s, ...(byId.has(s.id) ? { status: byId.get(s.id)!.status, duration: byId.get(s.id)!.duration ?? null, source: byId.get(s.id)!.source ?? null } : {}) })),
    log: [...snap.alerts].reverse().map((a) => ({ t: a.t, level: a.level, text: a.message })),
    status: snap.alerts.some((a) => a.level === "violation") ? "Deviation" : "On track",
    decision: snap.last_decision ? decisionView(snap.last_decision) : undefined,
    elapsed: snap.last_decision?.t ?? 0,
  };
}

export default function App({ api = httpApi, connect = connectSocket }: Props) {
  const [sop, setSop] = useState<Sop>();
  const [sources, setSources] = useState<SourceSpec[]>([]);
  const [selected, setSelected] = useState("");
  const [run, setRun] = useState<RunInfo | null>(null);
  const [state, setState] = useState<DashboardState>(initial([]));
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.getSops(), api.getSources(), api.getCurrent(), api.getInfo()]).then(([sops, srcs, current, info]) => {
      const first = sops[0];
      setSop(first);
      setSources(srcs);
      setSelected(current?.source ?? srcs[0]?.id ?? "");
      if (current && first) {
        setRun(current);
        setState(fromSnapshot(first, current, info.decider));
      } else if (first) {
        setState(initial(first.steps, info.decider));
      }
    }).catch((e) => setError(String(e.message ?? e)));
  }, [api]);

  useEffect(() => connect((m) => setState((s) => reduce(s, m))), [connect]);

  const labels = useMemo(() => Object.fromEntries((sop?.steps ?? []).map((s) => [s.id, s.label])), [sop]);
  const running = run !== null && !state.ended;
  const source = sources.find((s) => s.id === (run?.source ?? selected));

  const guard = useCallback(async (action: () => Promise<void>) => {
    setError("");
    try { await action(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }, []);

  const start = (sourceId: string) => guard(async () => {
    if (!sop) return;
    const info = await api.startRun(sop.id, sourceId);
    setState((s) => initial(sop.steps, s.decider));
    setRun(info);
    setPaused(false);
    setSpeed(1);
  });

  return (
    <main className="app">
      <TopBar title={sop?.name ?? "SOP Vision"} version={sop?.version} source={source?.label}
              elapsed={state.elapsed} status={state.status} running={running} />
      {error && <p className="error" role="alert">{error}</p>}
      <div className="main-grid">
        <VideoPanel src={api.videoUrl} running={running} chassisVisible={state.chassisVisible} />
        <Checklist steps={state.steps} decision={state.decision} />
      </div>
      <JevStrip decision={state.decision} labels={labels} jevOffline={state.jevOffline} name={state.decider} />
      <Timeline segments={state.timeline} offSopMarks={state.offSopMarks} elapsed={state.elapsed} labels={labels} />
      <EventLog entries={state.log} />
      <Controls
        sources={sources} selected={selected} onSelect={setSelected} running={running}
        pausable={source?.kind === "file"} paused={paused} speed={speed}
        reportHref={run ? api.reportUrl(run.run_id) : undefined}
        onStart={() => start(selected)}
        onStop={() => guard(async () => { if (run) await api.stopRun(run.run_id); })}
        onRestart={() => guard(async () => { if (run) { await api.stopRun(run.run_id); } await start(run?.source ?? selected); })}
        onPauseToggle={() => guard(async () => { if (run) { await api.control(run.run_id, paused ? "resume" : "pause"); setPaused(!paused); } })}
        onSpeed={(s) => guard(async () => { if (run) { await api.control(run.run_id, undefined, s); setSpeed(s); } })}
      />
    </main>
  );
}
