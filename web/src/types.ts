// Mirrors src/sopvision/messages.py and the REST payloads in src/sopvision/server/app.py.

export type SceneMsg = { type: "scene"; t: number; scene: { chassis: { visible: boolean; conf: number | null } } & Record<string, unknown> };
export type DecisionMsg = {
  type: "decision"; t: number; current_step: string; confidence: number; active_step_id: string;
  active_step_done: number; anomaly: number; off_sop: number; latency_ms: number;
  provider?: string; // which model answered: jev | decider | laya
};
export type StepUpdateMsg = { type: "step_update"; t: number; step_id: string; status: StepStatus; source: string | null; duration: number | null };
export type AlertMsg = { type: "alert"; t: number; level: "info" | "warning" | "violation"; kind: string; message: string; step_id: string | null };
export type RunSummaryMsg = { type: "run_summary"; t: number; report: { steps: { id: string; status: StepStatus }[] } & Record<string, unknown> };
export type WsMessage = SceneMsg | DecisionMsg | StepUpdateMsg | AlertMsg | RunSummaryMsg;

export type StepStatus = "pending" | "active" | "done";
export type SopStep = { id: string; label: string; text: string };
export type Sop = { id: string; name: string; version: number; description: string; steps: SopStep[]; torque?: boolean };
export type SourceSpec = { id: string; kind: "file" | "rtsp"; label: string };
export type LatencyResult = {
  frames: number; fps: number; decoded: number; delay_ms?: number; p10_ms?: number; p90_ms?: number;
  min_ms?: number; max_ms?: number; source_id?: string; saved?: boolean; measured_at?: string;
};
export type RecordingSop = { file: string; id: string; name: string; dataset: string; torque: boolean };
export type RecordingStatus = {
  run_id: string; sop: string; split: string; source: string; folder: string; seconds: number; frames: number;
  camera_delay_ms: number; presses: { time_s: number; ok: boolean }[]; error: string | null;
};
export type RecordingSummary = { run_id: string; folder: string; frames: number; seconds: number; presses: number; split: string };
export type AppConfig = { decision_model: string; decision_mode: string };
export type RunInfo = { run_id: string; sop: string; source: string; decision_model?: string };
export type RunSnapshot = RunInfo & {
  steps: { id: string; status: StepStatus; duration: number | null; source: string | null }[];
  alerts: { t: number; level: AlertMsg["level"]; kind: string; message: string; step_id: string | null }[];
  last_decision: DecisionMsg | null;
};
