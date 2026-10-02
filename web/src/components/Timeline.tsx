import type { Segment } from "../state";

type Props = { segments: Segment[]; offSopMarks: number[]; elapsed: number; labels: Record<string, string> };

export function Timeline({ segments, offSopMarks, elapsed, labels }: Props) {
  const span = Math.max(elapsed, ...segments.map((s) => s.end ?? s.start), 1);
  const pct = (t: number) => `${(100 * t) / span}%`;
  return (
    <section className="timeline card" aria-label="Timeline">
      <div className="track">
        {segments.map((s, i) => (
          <div key={i} className={`seg ${s.end === undefined ? "seg-active" : "seg-done"}`}
               style={{ left: pct(s.start), width: pct((s.end ?? elapsed) - s.start) }}
               title={labels[s.stepId] ?? s.stepId} />
        ))}
        {offSopMarks.map((t, i) => <div key={`o${i}`} className="mark" style={{ left: pct(t) }} title="Off-SOP activity" />)}
      </div>
    </section>
  );
}
