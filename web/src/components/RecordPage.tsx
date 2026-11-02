import { AppShell } from "../ui/AppShell";
import { useCallback, useEffect, useState } from "react";
import { httpApi, type Api } from "../api";
import type { RecordingSop, RecordingStatus, RecordingSummary, SourceSpec } from "../types";
import { TorqueButtons } from "./TorqueButtons";

type Props = { api?: Api; poll?: number; confirm?: (message: string) => boolean; initialSop?: string; back?: string };

/**
 * Record a camera into an SOP's dataset (no detector needed): video plus "torque reached" presses on the video's
 * timeline, written as runs/<name>/{video.mp4, torque.csv} and listed in splits.yaml.
 */
export function RecordPage({ api = httpApi, poll = 1000, confirm = (m) => window.confirm(m), initialSop, back }: Props) {
  const [sops, setSops] = useState<RecordingSop[]>([]);
  const [sources, setSources] = useState<SourceSpec[]>([]);
  const [sopFile, setSopFile] = useState("");
  const [sourceId, setSourceId] = useState("");
  const [split, setSplit] = useState("train");
  const [name, setName] = useState("");
  const [status, setStatus] = useState<RecordingStatus | null>(null);
  const [saved, setSaved] = useState<RecordingSummary | null>(null);
  const [discarded, setDiscarded] = useState("");
  const [error, setError] = useState("");
  const [sopErrors, setSopErrors] = useState<{ file: string; error: string }[]>([]);

  const refresh = useCallback(() => api.getRecording?.().then(setStatus).catch(() => {}), [api]);

  useEffect(() => {
    api.getRecordingSops?.().then((s) => { setSops(s); setSopFile((f) => f || s.find((x) => x.file === initialSop || x.file.endsWith(`/${initialSop}`))?.file || s[0]?.file || ""); }).catch((e) => setError(String(e.message ?? e)));
    api.getRecordingSopErrors?.().then(setSopErrors).catch(() => {});
    api.getSources().then((all) => {
      const live = all.filter((s) => s.kind !== "file");
      setSources(live);
      setSourceId((id) => id || live[0]?.id || "");
    }).catch(() => {});
    refresh();
  }, [api, refresh]);

  useEffect(() => {
    if (!status) return;
    const timer = window.setInterval(refresh, poll);
    return () => window.clearInterval(timer);
  }, [status !== null, refresh, poll]); // eslint-disable-line react-hooks/exhaustive-deps

  const guard = async (action: () => Promise<void>) => {
    setError("");
    try { await action(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const start = () => guard(async () => {
    if (!api.startRecording) return;
    setSaved(null);
    setDiscarded("");
    setStatus(await api.startRecording(sopFile, sourceId, split, name.trim() || undefined));
  });
  const stop = () => guard(async () => {
    if (!api.stopRecording) return;
    setSaved(await api.stopRecording());
    setStatus(null);
    setName("");
  });
  const discard = () => guard(async () => {
    if (!api.discardRecording || !status) return;
    if (!confirm(`Discard ${status.run_id}? The video is deleted and nothing is saved.`)) return;
    await api.discardRecording();
    setDiscarded(`Discarded ${status.run_id}: nothing was saved. You can record it again under the same name.`);
    setStatus(null);
  });
  const torque = async (ok: boolean) => {
    if (!api.recordingTorque) return "";
    const { time_s } = await api.recordingTorque(ok);
    refresh();
    return `${ok ? "Torque reached" : "Not reached"} at ${time_s.toFixed(2)} s`;
  };
  const sop = sops.find((s) => s.file === sopFile);

  return (
    <AppShell title="Record a run" current="record" back={back ? { href: back, label: "Back to the SOP" } : undefined}>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="latency-grid">
        <section className="card record-video" aria-label="Preview">
          {status ? <img src={api.videoUrl} alt="Camera preview" /> : <p className="muted">The camera preview shows here while recording.</p>}
        </section>
        <section className="card latency-panel">
          {!status ? (
            <>
              <p className="muted">
                Records the camera into the SOP's dataset with no detection, for training a new SOP. Press the torque
                button the moment the wrench reaches torque. Stop any run first.
              </p>
              {sops.length === 0 && sopErrors.length === 0 && <p className="error">No SOP with a timeline dataset (sops/ or sops/drafts/).</p>}
              {sopErrors.map((e) => <p key={e.file} className="error">{e.file} could not be loaded: {e.error}</p>)}
              {sources.length === 0 && <p className="error">No live camera configured. Add one with SOPVISION_RTSP_URLS in .env.</p>}
              <label>SOP <select value={sopFile} onChange={(e) => setSopFile(e.target.value)}>
                {sops.map((s) => <option key={s.file} value={s.file}>{s.name}</option>)}
              </select></label>
              <label>Camera <select value={sourceId} onChange={(e) => setSourceId(e.target.value)}>
                {sources.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
              </select></label>
              <label>Split <select value={split} onChange={(e) => setSplit(e.target.value)}>
                <option value="train">train</option><option value="val">val</option><option value="test">test (never trained on)</option>
              </select></label>
              <label>Name (optional) <input value={name} placeholder="e.g. r07-wrong-order" onChange={(e) => setName(e.target.value)} /></label>
              {sop && <p className="muted">Saved to {sop.dataset}/runs/&lt;name&gt;</p>}
              <button onClick={start} disabled={!sopFile || !sourceId}>Start recording</button>
            </>
          ) : (
            <>
              <dl className="latency-result" aria-label="Recording">
                <dt>Run</dt><dd>{status.run_id} ({status.split})</dd>
                <dt>Recorded</dt><dd>{status.seconds.toFixed(1)} s · {status.frames} frames</dd>
                <dt>Camera delay</dt><dd>{status.camera_delay_ms} ms</dd>
                <dt>Torque presses</dt><dd>{status.presses.map((p) => `${p.time_s.toFixed(1)}${p.ok ? "" : "✗"}`).join(", ") || "none yet"}</dd>
              </dl>
              {status.error && <p className="error">{status.error}</p>}
              {sop?.torque !== false && <TorqueButtons onPress={torque} />}
              <div className="record-actions">
                <button onClick={stop}>Stop and save</button>
                <button className="discard" onClick={discard}>Discard (don't save)</button>
              </div>
            </>
          )}
          {discarded && <p role="status">{discarded}</p>}
          {saved && (
            <p role="status">
              Saved {saved.run_id} ({saved.split}): {saved.seconds} s, {saved.presses} torque presses.
              {sop && <> Next: <a href={`#/sops/${sop.id}/timelines`}>label its timeline</a> in the SOP builder.</>}
            </p>
          )}
        </section>
      </div>
    </AppShell>
  );
}
