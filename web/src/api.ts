import type { Info, RunInfo, RunSnapshot, Sop, SourceSpec, WsMessage } from "./types";

export type Api = {
  getInfo(): Promise<Info>;
  getSops(): Promise<Sop[]>;
  getSources(): Promise<SourceSpec[]>;
  getCurrent(): Promise<RunSnapshot | null>;
  startRun(sop: string, sourceId: string): Promise<RunInfo>;
  stopRun(runId: string): Promise<void>;
  control(runId: string, action?: "pause" | "resume", speed?: number): Promise<void>;
  reportUrl(runId: string): string;
  videoUrl: string;
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return (res.status === 204 ? null : await res.json()) as T;
}

export const httpApi: Api = {
  getInfo: () => call("/info"),
  getSops: () => call("/sops"),
  getSources: () => call("/sources"),
  getCurrent: () => call("/runs/current"),
  startRun: (sop, sourceId) => call("/runs", { method: "POST", body: JSON.stringify({ sop, source_id: sourceId }) }),
  stopRun: async (runId) => { await call(`/runs/${runId}/stop`, { method: "POST" }); },
  control: async (runId, action, speed) => {
    await call(`/runs/${runId}/control`, { method: "POST", body: JSON.stringify({ action, speed }) });
  },
  reportUrl: (runId) => `/runs/${runId}/report?format=csv`,
  videoUrl: "/video.mjpg",
};

/** Subscribe to run messages; reconnects every 2 s if the server restarts. Returns an unsubscribe function. */
export function connectSocket(onMessage: (m: WsMessage) => void): () => void {
  let socket: WebSocket | null = null;
  let closed = false;
  let retry: ReturnType<typeof setTimeout> | undefined;
  const open = () => {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.onmessage = (e) => onMessage(JSON.parse(e.data));
    socket.onclose = () => { if (!closed) retry = setTimeout(open, 2000); };
  };
  open();
  return () => { closed = true; clearTimeout(retry); socket?.close(); };
}
