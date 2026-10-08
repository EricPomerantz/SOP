import { initial, reduce } from "./state";
import type { WsMessage } from "./types";

const STEPS = [
  { id: "fan_1", label: "Fan 1" },
  { id: "fan_2", label: "Fan 2" },
];
const run = (msgs: WsMessage[]) => msgs.reduce(reduce, initial(STEPS));

test("step_update marks a step done with its duration", () => {
  const s = run([{ type: "step_update", t: 9.1, step_id: "fan_1", status: "done", source: "jev", duration: 9.1 }]);
  expect(s.steps[0]).toMatchObject({ id: "fan_1", status: "done", duration: 9.1, source: "jev" });
  expect(s.log[0].text).toBe("Fan 1 complete · by Jev");
});

test("active then done builds a timeline segment", () => {
  const s = run([
    { type: "step_update", t: 2, step_id: "fan_1", status: "active", source: null, duration: null },
    { type: "step_update", t: 9, step_id: "fan_1", status: "done", source: "evidence", duration: 7 },
  ]);
  expect(s.timeline).toEqual([{ stepId: "fan_1", start: 2, end: 9 }]);
});

test("a violation flips status to Deviation", () => {
  const s = run([{ type: "alert", t: 5, level: "violation", kind: "order_violation", message: "fan_2 completed before fan_1", step_id: "fan_2" }]);
  expect(s.status).toBe("Deviation");
  expect(s.log[0]).toMatchObject({ t: 5, level: "violation", text: "fan_2 completed before fan_1" });
});

test("run_summary with missing steps → Incomplete", () => {
  const s = run([{ type: "run_summary", t: 40, report: { steps: [{ id: "fan_1", status: "done" }, { id: "fan_2", status: "pending" }] } }]);
  expect(s.status).toBe("Incomplete");
  expect(s.ended).toBe(true);
});

test("jev_offline alert sets the badge until the next decision", () => {
  let s = run([{ type: "alert", t: 3, level: "warning", kind: "jev_offline", message: "Jev offline — using evidence only", step_id: null }]);
  expect(s.jevOffline).toBe(true);
  expect(s.status).toBe("On track");
  s = reduce(s, { type: "decision", t: 4, current_step: "fan_1", confidence: 0.9, active_step_id: "fan_1", active_step_done: 0.1, anomaly: 0, off_sop: 0.02, latency_ms: 140 });
  expect(s.jevOffline).toBe(false);
  expect(s.decision).toMatchObject({ currentStep: "fan_1", confidence: 0.9, offSop: 0.02, latencyMs: 140 });
});

test("scene without chassis shows the banner flag and advances the clock", () => {
  const s = run([{ type: "scene", t: 12.5, scene: { chassis: { visible: false, conf: null } } }]);
  expect(s.chassisVisible).toBe(false);
  expect(s.elapsed).toBe(12.5);
});

test("the deciding model names the strip and the completion source", () => {
  let s = initial(STEPS, "laya");
  expect(s.model).toBe("laya");
  s = reduce(s, { type: "step_update", t: 9.1, step_id: "fan_1", status: "done", source: "jev", duration: 9.1 });
  expect(s.log[0].text).toBe("Fan 1 complete · by Laya");
  s = reduce(s, { type: "decision", t: 10, current_step: "fan_2", confidence: 0.9, active_step_id: "fan_2",
                  active_step_done: 0.1, anomaly: 0, off_sop: 0, latency_ms: 85, provider: "decider" });
  expect(s.model).toBe("decider");
  expect(s.decision?.provider).toBe("decider");
});
