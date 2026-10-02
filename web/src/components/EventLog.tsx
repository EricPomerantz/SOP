import { stamp } from "../format";
import type { LogEntry } from "../state";

export function EventLog({ entries }: { entries: LogEntry[] }) {
  return (
    <section className="log card">
      <h2>Events</h2>
      <ol aria-label="Event log">
        {entries.length === 0 && <li className="muted">Nothing yet</li>}
        {entries.map((e, i) => (
          <li key={i} className={`log-${e.level}`}>
            <span className="mono muted">{stamp(e.t)}</span>
            <span>{e.text}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
