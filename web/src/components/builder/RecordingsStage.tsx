import type { SopView } from "../../builderApi";

/** The SOP's recorded runs by split, and a link to record more (the Record page, this SOP preselected). */
export function RecordingsStage({ view }: { view: SopView }) {
  const runs = view.status.runs ?? [];
  const c = view.status.recordings;
  return (
    <div className="stage">
      <p className="muted">
        Record runs of the procedure with the camera: train and val runs teach the decision model, test runs measure
        it and are never trained on. Record correct runs and deviations (wrong order, a part taken out, idle, off-SOP
        actions). Recording needs the detector stage done (the SOP must load).
      </p>
      <p>
        train {c.train ?? 0} · val {c.val ?? 0} · test {c.test ?? 0}{" "}
        <a href={`#/record?sop=${encodeURIComponent(view.file)}`}>Record a run</a>
      </p>
      {runs.length > 0 && (
        <table className="grid-table" aria-label="Recordings">
          <thead><tr><th>Run</th><th>Split</th><th>Torque log</th><th>Timeline</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td className="mono">{r.id}</td><td>{r.split ?? <span className="error">not in splits.yaml</span>}</td>
                <td>{r.torque ? "yes" : "—"}</td><td>{r.steps ? "✓" : r.draft ? "draft" : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
