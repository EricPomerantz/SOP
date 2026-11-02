import { AppShell } from "../ui/AppShell";
import QRCode from "qrcode";
import { useCallback, useEffect, useRef, useState } from "react";
import { httpApi, type Api } from "../api";
import type { LatencyResult, SourceSpec } from "../types";

export const QR_PREFIX = "SVL:";
/** How long each code stays on screen: long enough that a camera frame (~40 ms exposure) sees one clean code. */
export const HOLD_MS = 50;
const WRAP_MS = 100_000_000;

/** The QR payload: prefix, hold time, and the app-clock time modulo WRAP_MS (short, so the code has large modules). */
export function clockText(appNowMs: number): string {
  return `${QR_PREFIX}${HOLD_MS}:${Math.round(appNowMs % WRAP_MS)}`;
}

/** Offset to add to this browser's clock to get the app's clock, from one round trip (NTP-style midpoint). */
export function clockOffset(sentMs: number, serverMs: number, receivedMs: number): number {
  return serverMs - (sentMs + receivedMs) / 2;
}

type Props = { api?: Api };

/**
 * Point a live camera at this page: the QR code always shows the current time on the app's clock, so every frame the
 * app receives tells it how long ago it was filmed.
 */
export function LatencyPage({ api = httpApi }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const offset = useRef(0);
  const [sources, setSources] = useState<SourceSpec[]>([]);
  const [sourceId, setSourceId] = useState("");
  const [seconds, setSeconds] = useState(10);
  const [save, setSave] = useState(true);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<LatencyResult | null>(null);
  const [saved, setSaved] = useState<Record<string, LatencyResult>>({});
  const [error, setError] = useState("");

  const sync = useCallback(async () => {
    if (!api.latencyClock) return;
    const sent = Date.now();
    const { now_ms } = await api.latencyClock();
    offset.current = clockOffset(sent, now_ms, Date.now());
  }, [api]);

  useEffect(() => {
    api.getSources().then((all) => {
      const live = all.filter((s) => s.kind !== "file");
      setSources(live);
      setSourceId((id) => id || live[0]?.id || "");
    }).catch((e) => setError(String(e.message ?? e)));
    api.getLatency?.().then(setSaved).catch(() => {});
    sync().catch(() => {});
    const timer = window.setInterval(() => { sync().catch(() => {}); }, 5000);
    return () => window.clearInterval(timer);
  }, [api, sync]);

  useEffect(() => {
    let frame = 0;
    let shown = 0;
    const draw = () => {
      const el = canvas.current;
      const now = Date.now();
      if (el && now - shown >= HOLD_MS) {
        shown = now;
        QRCode.toCanvas(el, clockText(now + offset.current), { errorCorrectionLevel: "M", margin: 2, width: el.clientWidth || 480 }, () => {});
      }
      frame = window.requestAnimationFrame(draw);
    };
    frame = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(frame);
  }, []);

  const run = async () => {
    if (!api.measureLatency || !sourceId) return;
    setBusy(true); setError(""); setResult(null);
    try {
      setResult(await api.measureLatency(sourceId, seconds, save));
      api.getLatency?.().then(setSaved).catch(() => {});
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AppShell title="Camera latency check" current="latency">
      <div className="latency-grid">
        <section className="card latency-qr" aria-label="QR clock">
          <canvas ref={canvas} />
        </section>
        <section className="card latency-panel">
          <p className="muted">
            Point the camera at the code so it fills a good part of the picture, then measure. The delay is how long a
            moment takes to reach the app from the camera (encoder, network and decoding). Torque results are matched to
            the video using it. Stop any run first.
          </p>
          {sources.length === 0 ? (
            <p className="error">No live camera configured. Add one with SOPVISION_RTSP_URLS in .env.</p>
          ) : (
            <>
              <label>Camera{" "}
                <select value={sourceId} onChange={(e) => setSourceId(e.target.value)}>
                  {sources.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
                </select>
              </label>
              <label>Seconds{" "}
                <input type="number" min={2} max={60} value={seconds} onChange={(e) => setSeconds(Number(e.target.value))} />
              </label>
              <label><input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)} /> Save as this camera's delay</label>
              <button onClick={run} disabled={busy}>{busy ? "Measuring…" : "Measure"}</button>
            </>
          )}
          {error && <p className="error" role="alert">{error}</p>}
          {result && (
            <dl className="latency-result" aria-label="Result">
              <dt>Delay (median)</dt><dd>{result.delay_ms} ms</dd>
              <dt>Spread (10th–90th)</dt><dd>{result.p10_ms}–{result.p90_ms} ms</dd>
              <dt>Frames read / with a readable code</dt><dd>{result.frames} / {result.decoded} at {result.fps} fps</dd>
              <dt>Saved</dt><dd>{result.saved ? "yes" : "no"}</dd>
            </dl>
          )}
          {Object.keys(saved).length > 0 && (
            <>
              <h2>Saved delays</h2>
              <ul>{Object.entries(saved).map(([id, r]) => <li key={id}>{id}: {r.delay_ms} ms ({r.measured_at})</li>)}</ul>
            </>
          )}
        </section>
      </div>
    </AppShell>
  );
}
