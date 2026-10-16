/** The SOP builder's API (/builder/..., /jobs). Kept apart from the dashboard's Api so its fakes stay small. */

export type SopData = {
  sop: string; name?: string; version: number; description: string;
  detector: { weights: string; classes: string[]; roles?: Record<string, "part" | "empty"> };
  layout: {
    anchor: string; match_radius: number; tools?: string[]; anchor_follow?: number; removal_seconds?: number | null;
    slots: Record<string, SlotData>;
  };
  steps: StepData[];
  order: "strict" | "any" | "groups";
  groups?: (string[] | { sequence: string[] })[] | null;
  off_sop: string; idle: string;
  decisions?: Record<string, number>;
  dataset?: { format?: string; dir?: string };
  [key: string]: unknown;
};
export type SlotData = { at: [number, number]; filled_by: string | string[]; empty_as?: string | null; match_radius?: number | null };
export type Evidence = "installed" | "empty" | "torqued";
export type StepData = { id: string; text: string; label?: string; evidence?: Record<string, Evidence>; hands_off?: boolean };
export type RunRow = { id: string; split: string | null; steps: boolean; draft: boolean; torque: boolean };
export type BuilderStatus = {
  errors: string[]; detector: boolean; roles: boolean; layout: boolean; steps: boolean;
  recordings: Record<string, number>; timelines: Record<string, number>; laya_checkpoint: string | null;
  published: boolean; loaded: boolean; draft: boolean; runs?: RunRow[];
};
export type SopView = { id: string; file: string; draft: boolean; data: SopData; warnings: string[]; status: BuilderStatus };
export type SopListEntry = { id: string; name: string; draft: boolean; published: boolean; loaded: boolean };
export type ModelEntry = { path: string; classes: string[] };
export type JobStep = { name: string; argv: string[]; state: string; started: number | null; ended: number | null };
export type Job = {
  id: string; kind: string; title: string; sop: string | null; state: string; error: string | null; created: number;
  steps: JobStep[]; result: Record<string, unknown>; params: Record<string, unknown>; cancel_requested: boolean; log?: string;
};

export type AutofitSlot = {
  name: string; at: [number, number]; spread: number; share: number; radius: number; nearest: number;
  classes: Record<string, number>; filled_by: string[]; empty_as: string | null; points: [number, number][];
};
export type AutofitResult = {
  sop: string; threshold: number; slots: AutofitSlot[]; match_radius: number; frames_used: number; frames_dropped: number;
  warnings: string[]; samples: Record<string, { t: number; width: number; height: number; anchor: [number, number, number, number] }>;
};

export type TimelineView = {
  run: string; sop_file: string; fps: number; duration: number; preview: boolean;
  steps: [string, number, number][] | null; draft: [string, number, number][] | null;
  draft_info: { alerts: string[]; not_done: string[] } | null;
  torque: { time_s: number; ok: boolean }[]; step_ids: string[]; labels: Record<string, string>;
};

export type BenchSummary = {
  requests: number; errors: number; current_step_accuracy: number; labelled: number; echo_rate: number;
  accuracy_when_acting: number; off_sop_false_negative_rate: number | null; off_sop_false_positive_rate: number | null;
  anomaly_false_positive_rate: number | null; request_ms_p50: number;
};
export type TrainReport = {
  sop: string; checkpoint: string; active: string | null; training_set: Record<string, number> | null;
  bench: BenchSummary | null; bench_active: BenchSummary | null;
  replay: { run: string; done: string[]; not_done: string[]; alerts: string[]; current_step: number | null; idle_or_off_sop: string }[];
};
export type Checkpoints = {
  active: string | null; versions: { checkpoint: string; report: TrainReport | null }[]; shared: string[]; warning?: string | null;
};

export type BuilderApi = {
  list(): Promise<SopListEntry[]>;
  create(id: string, name: string, description: string): Promise<SopView>;
  get(id: string): Promise<SopView>;
  save(id: string, data: SopData): Promise<SopView>;
  edit(id: string): Promise<SopView>;
  discard(id: string): Promise<SopView | { id: string; deleted: true }>;
  publish(id: string): Promise<SopView>;
  generateSteps(id: string): Promise<Pick<SopData, "steps" | "order" | "groups">>;
  models(): Promise<ModelEntry[]>;
  uploadDetector(id: string, file: Blob): Promise<SopView>;
  useModel(id: string, path: string): Promise<SopView>;
  autofit(id: string): Promise<AutofitResult | null>;
  frameUrl(id: string, run: string): string;
  timeline(id: string, run: string): Promise<TimelineView>;
  saveTimeline(id: string, run: string, rows: [string, number, number][]): Promise<TimelineView>;
  previewUrl(id: string, run: string): string;
  checkpoints(id: string): Promise<Checkpoints>;
  activate(id: string, checkpoint: string | null): Promise<Checkpoints>;
  jobs(sopFile?: string): Promise<Job[]>;
  job(id: string): Promise<Job>;
  startJob(kind: string, params: Record<string, unknown>): Promise<Job>;
  cancelJob(id: string): Promise<Job>;
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

const post = (body?: unknown): RequestInit => ({ method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const builderApi: BuilderApi = {
  list: () => call("/builder/sops"),
  create: (id, name, description) => call("/builder/sops", post({ id, name, description })),
  get: (id) => call(`/builder/sops/${id}`),
  save: (id, data) => call(`/builder/sops/${id}`, { method: "PUT", body: JSON.stringify({ data }) }),
  edit: (id) => call(`/builder/sops/${id}/edit`, post()),
  discard: (id) => call(`/builder/sops/${id}/discard`, post()),
  publish: (id) => call(`/builder/sops/${id}/publish`, post()),
  generateSteps: (id) => call(`/builder/sops/${id}/generate-steps`, post()),
  models: () => call("/builder/models"),
  uploadDetector: (id, file) => call(`/builder/sops/${id}/detector`, {
    method: "PUT", body: file, headers: { "Content-Type": "application/octet-stream" },
  }),
  useModel: (id, path) => call(`/builder/sops/${id}/detector`, post({ path })),
  autofit: (id) => call(`/builder/sops/${id}/autofit`),
  frameUrl: (id, run) => `/builder/sops/${id}/frames/${encodeURIComponent(run)}.jpg`,
  timeline: (id, run) => call(`/builder/sops/${id}/runs/${encodeURIComponent(run)}/timeline`),
  saveTimeline: (id, run, rows) => call(`/builder/sops/${id}/runs/${encodeURIComponent(run)}/timeline`, {
    method: "PUT", body: JSON.stringify({ rows }),
  }),
  previewUrl: (id, run) => `/builder/sops/${id}/runs/${encodeURIComponent(run)}/preview.webm`,
  checkpoints: (id) => call(`/builder/sops/${id}/checkpoints`),
  activate: (id, checkpoint) => call(`/builder/sops/${id}/activate`, post({ checkpoint })),
  jobs: (sopFile) => call(`/jobs${sopFile ? `?sop=${encodeURIComponent(sopFile)}` : ""}`),
  job: (id) => call(`/jobs/${id}`),
  startJob: (kind, params) => call("/jobs", post({ kind, params })),
  cancelJob: (id) => call(`/jobs/${id}/cancel`, post()),
};
