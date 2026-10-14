import { clock } from "../format";
import type { RunStatus } from "../state";

type Props = { title: string; version?: number; source?: string; elapsed: number; status: RunStatus; running: boolean };

export function TopBar({ title, version, source, elapsed, status, running }: Props) {
  const tone = status === "On track" ? "ok" : status === "Deviation" ? "bad" : "warn";
  return (
    <header className="topbar card">
      <div className="topbar-title">
        <span className="logo" aria-hidden="true">◧</span>
        <span className="name">{title}</span>
        {version !== undefined && <span className="muted">v{version}</span>}
        {source && <span className="muted">· {source}</span>}
      </div>
      <div className="topbar-right">
        <a className="muted" href="#/record">Record</a>
        <a className="muted" href="#/latency">Camera latency</a>
        <span className="timer" aria-label="Run time">{clock(elapsed)}</span>
        <span role="status" className={`pill pill-${running || status !== "On track" ? tone : "idle"}`}>
          {running || status !== "On track" ? status : "No run"}
        </span>
      </div>
    </header>
  );
}
