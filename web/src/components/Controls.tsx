import type { SourceSpec } from "../types";

type Props = {
  sources: SourceSpec[];
  selected: string;
  onSelect(id: string): void;
  running: boolean;
  pausable: boolean;
  paused: boolean;
  speed: number;
  reportHref?: string;
  onStart(): void;
  onStop(): void;
  onRestart(): void;
  onPauseToggle(): void;
  onSpeed(speed: number): void;
};

export function Controls(p: Props) {
  return (
    <section className="controls">
      <select aria-label="Source" value={p.selected} onChange={(e) => p.onSelect(e.target.value)} disabled={p.running}>
        {p.sources.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
      </select>
      {p.running ? (
        <>
          <button onClick={p.onStop}>Stop</button>
          <button onClick={p.onRestart}>Restart</button>
          {p.pausable && <button onClick={p.onPauseToggle}>{p.paused ? "Resume" : "Pause"}</button>}
          {p.pausable && (
            <button onClick={() => p.onSpeed(p.speed === 1 ? 2 : 1)} aria-label="Playback speed">{p.speed}×</button>
          )}
        </>
      ) : (
        <button className="primary" onClick={p.onStart} disabled={!p.selected}>Start</button>
      )}
      {p.reportHref && <a className="button" href={p.reportHref} download>Download run report</a>}
    </section>
  );
}
