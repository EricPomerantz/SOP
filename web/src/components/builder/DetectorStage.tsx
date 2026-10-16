import { useEffect, useState } from "react";
import type { BuilderApi, ModelEntry, SopData, SopView } from "../../builderApi";

export type Role = "anchor" | "part" | "empty" | "tool" | "hand" | "ignore";

export const ROLE_LABELS: Record<Role, string> = {
  anchor: "Anchor (slots are placed on it)", part: "Part (fills slots)", empty: "Empty-slot marker",
  tool: "Tool (e.g. torque wrench)", hand: "Hand", ignore: "Ignore",
};

export function roleOf(data: SopData, cls: string): Role {
  if (cls === "hand") return "hand";
  if (data.layout.anchor === cls) return "anchor";
  if ((data.layout.tools ?? []).includes(cls)) return "tool";
  return data.detector.roles?.[cls] ?? "ignore";
}

/** The draft with `cls` given `role`: one anchor at most, tools in layout.tools, parts and markers in detector.roles. */
export function setRole(data: SopData, cls: string, role: Role): SopData {
  const roles = { ...(data.detector.roles ?? {}) };
  delete roles[cls];
  if (role === "part" || role === "empty") roles[cls] = role;
  let tools = (data.layout.tools ?? []).filter((t) => t !== cls);
  if (role === "tool") tools = [...tools, cls];
  let anchor = data.layout.anchor === cls ? "" : data.layout.anchor;
  if (role === "anchor") anchor = cls;
  return { ...data, detector: { ...data.detector, roles }, layout: { ...data.layout, anchor, tools } };
}

type Props = { api: BuilderApi; view: SopView; data: SopData; onChange: (d: SopData) => void; onView: (v: SopView) => void };

/** Import the trained detector (upload or one already under models/) and say what each of its classes is. */
export function DetectorStage({ api, view, data, onChange, onView }: Props) {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [pick, setPick] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => { api.models().then((m) => { setModels(m); setPick(m[0]?.path ?? ""); }).catch(() => {}); }, [api]);

  const run = async (label: string, action: () => Promise<SopView>) => {
    setBusy(label);
    setError("");
    try { onView(await action()); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(""); }
  };
  const classes = data.detector.classes ?? [];

  return (
    <div className="stage">
      <p className="muted">
        A trained Ultralytics detection model (.pt). Its classes become the SOP's: one is the anchor (the work piece the
        slots sit on, e.g. a hub or a chassis), others are parts and the markers the model gives an empty slot.
        Only import models you trust: a .pt file can run code when it is loaded.
      </p>
      {error && <p className="error" role="alert">{error}</p>}
      {view.warnings.map((w) => <p key={w} className="pill-warn warn-line">{w}</p>)}
      <div className="row">
        <label>Upload <input type="file" accept=".pt" disabled={!view.draft || !!busy}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) run("Uploading…", () => api.uploadDetector(view.id, f)); }} /></label>
        {models.length > 0 && (
          <>
            <span className="muted">or</span>
            <select aria-label="Existing model" value={pick} onChange={(e) => setPick(e.target.value)}>
              {models.map((m) => <option key={m.path} value={m.path}>{m.path} ({m.classes.length} classes)</option>)}
            </select>
            <button disabled={!view.draft || !pick || !!busy} onClick={() => run("Loading…", () => api.useModel(view.id, pick))}>Use this model</button>
          </>
        )}
        {busy && <span className="muted">{busy}</span>}
      </div>
      <p className="muted mono">Weights: {data.detector.weights}</p>
      {classes.length > 0 && (
        <table className="grid-table" aria-label="Classes">
          <thead><tr><th>Class</th><th>Role</th></tr></thead>
          <tbody>
            {classes.map((c) => (
              <tr key={c}>
                <td className="mono">{c}</td>
                <td>
                  <select aria-label={`Role of ${c}`} value={roleOf(data, c)} disabled={!view.draft || c === "hand"}
                    onChange={(e) => onChange(setRole(data, c, e.target.value as Role))}>
                    {(Object.keys(ROLE_LABELS) as Role[]).filter((r) => r !== "hand" || c === "hand")
                      .map((r) => <option key={r} value={r}>{ROLE_LABELS[r]}</option>)}
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
