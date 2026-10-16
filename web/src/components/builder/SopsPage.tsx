import { useEffect, useState } from "react";
import { builderApi, type BuilderApi, type SopListEntry } from "../../builderApi";

type Props = { api?: BuilderApi; go?: (hash: string) => void };

/** Every SOP, published and draft, and a form to start a new one (#/sops). */
export function SopsPage({ api = builderApi, go = (h) => { window.location.hash = h; } }: Props) {
  const [sops, setSops] = useState<SopListEntry[]>([]);
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState("");

  useEffect(() => { api.list().then(setSops).catch((e) => setError(e.message)); }, [api]);

  const create = async () => {
    setError("");
    try {
      const view = await api.create(id.trim(), name.trim(), description.trim());
      go(`#/sops/${view.id}`);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  return (
    <main className="app builder">
      <header className="topbar card">
        <div className="topbar-title"><span className="name">SOPs</span></div>
        <div className="topbar-right"><a className="muted" href="#/">Back to the dashboard</a></div>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="latency-grid">
        <section className="card latency-panel" aria-label="SOP list">
          {sops.length === 0 && <p className="muted">No SOPs yet.</p>}
          <ul className="sop-list">
            {sops.map((s) => (
              <li key={s.id}>
                <a href={`#/sops/${s.id}`}>{s.name}</a> <span className="muted mono">{s.id}</span>
                {s.published && <span className={`pill ${s.loaded ? "pill-ok" : "pill-warn"}`}>{s.loaded ? "live" : "published"}</span>}
                {s.draft && <span className="pill pill-idle">draft</span>}
              </li>
            ))}
          </ul>
        </section>
        <section className="card latency-panel" aria-label="New SOP">
          <h3>New SOP</h3>
          <p className="muted">
            Start from a trained YOLO detection model. The builder walks through the detector, recordings, slot
            layout, steps, step timelines and training the decision model, then publishes the SOP.
          </p>
          <label>Id <input value={id} placeholder="e.g. wheel_lug_bolts" onChange={(e) => setId(e.target.value)} /></label>
          <label>Name <input value={name} placeholder="Wheel lug bolts" onChange={(e) => setName(e.target.value)} /></label>
          <label>Description <textarea value={description} rows={3} onChange={(e) => setDescription(e.target.value)} /></label>
          <button onClick={create} disabled={!id.trim()}>Create draft</button>
        </section>
      </div>
    </main>
  );
}
