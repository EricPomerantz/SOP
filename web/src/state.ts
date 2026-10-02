import type { AlertMsg, DecisionMsg, StepStatus, WsMessage } from "./types";

export type StepView = { id: string; label: string; status: StepStatus; duration: number | null; source: string | null };
export type LogEntry = { t: number; level: AlertMsg["level"] | "step"; text: string };
export type DecisionView = {
  t: number; currentStep: string; confidence: number; activeStepId: string; activeStepDone: number;
  anomaly: number; offSop: number; latencyMs: number;
};
export type Segment = { stepId: string; start: number; end?: number };
export type RunStatus = "On track" | "Deviation" | "Incomplete";

export type DashboardState = {
  steps: StepView[];
  log: LogEntry[];
  decision?: DecisionView;
  status: RunStatus;
  elapsed: number;
  timeline: Segment[];
  offSopMarks: number[];
  jevOffline: boolean;
  chassisVisible: boolean;
  ended: boolean;
};

const MAX_LOG = 200;
const SOURCE_NAMES: Record<string, string> = { jev: "Jev", evidence: "evidence" };

export function initial(steps: { id: string; label: string }[]): DashboardState {
  return {
    steps: steps.map((s) => ({ ...s, status: "pending", duration: null, source: null })),
    log: [],
    status: "On track",
    elapsed: 0,
    timeline: [],
    offSopMarks: [],
    jevOffline: false,
    chassisVisible: true,
    ended: false,
  };
}

export function decisionView(m: DecisionMsg): DecisionView {
  return {
    t: m.t, currentStep: m.current_step, confidence: m.confidence, activeStepId: m.active_step_id,
    activeStepDone: m.active_step_done, anomaly: m.anomaly, offSop: m.off_sop, latencyMs: m.latency_ms,
  };
}

function closeSegment(timeline: Segment[], stepId: string, t: number): Segment[] {
  const i = timeline.findIndex((s) => s.stepId === stepId && s.end === undefined);
  return i < 0 ? timeline : timeline.map((s, j) => (j === i ? { ...s, end: t } : s));
}

const withLog = (log: LogEntry[], entry: LogEntry) => [entry, ...log].slice(0, MAX_LOG);

export function reduce(state: DashboardState, msg: WsMessage): DashboardState {
  const elapsed = Math.max(state.elapsed, msg.t);
  switch (msg.type) {
    case "scene":
      return { ...state, elapsed, chassisVisible: msg.scene.chassis.visible };
    case "decision":
      return { ...state, elapsed, decision: decisionView(msg), jevOffline: false };
    case "step_update": {
      const label = state.steps.find((s) => s.id === msg.step_id)?.label ?? msg.step_id;
      const steps = state.steps.map((s) =>
        s.id === msg.step_id ? { ...s, status: msg.status, duration: msg.duration, source: msg.source } : s);
      let timeline = state.timeline;
      let log = state.log;
      if (msg.status === "active") {
        timeline = [...closeSegment(timeline, msg.step_id, msg.t), { stepId: msg.step_id, start: msg.t }];
      } else {
        const hadOpen = timeline.some((s) => s.stepId === msg.step_id && s.end === undefined);
        timeline = hadOpen ? closeSegment(timeline, msg.step_id, msg.t)
          : msg.status === "done" ? [...timeline, { stepId: msg.step_id, start: msg.t - (msg.duration ?? 0), end: msg.t }]
          : timeline;
      }
      if (msg.status === "done") {
        log = withLog(log, { t: msg.t, level: "step", text: `${label} complete · by ${SOURCE_NAMES[msg.source ?? ""] ?? msg.source}` });
      }
      return { ...state, elapsed, steps, timeline, log };
    }
    case "alert":
      return {
        ...state,
        elapsed,
        status: msg.level === "violation" ? "Deviation" : state.status,
        jevOffline: state.jevOffline || msg.kind === "jev_offline",
        offSopMarks: msg.kind === "off_sop" ? [...state.offSopMarks, msg.t] : state.offSopMarks,
        log: withLog(state.log, { t: msg.t, level: msg.level, text: msg.message }),
      };
    case "run_summary": {
      const incomplete = msg.report.steps.some((s) => s.status !== "done");
      return { ...state, elapsed, ended: true, status: incomplete ? "Incomplete" : state.status };
    }
  }
}
