import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import type { BuilderApi, SopData, SopView } from "../../builderApi";
const lastData = (fn: ReturnType<typeof vi.fn>) => fn.mock.calls[fn.mock.calls.length - 1]![0] as SopData;
import { Providers } from "../../ui/Providers";
import { StepsStage } from "./StepsStage";

const step = (id: string, slot: string, ev: "installed" | "torqued") => ({ id, label: id, text: `${id}.`, evidence: { [slot]: ev } });
const DATA: SopData = {
  sop: "wheel", name: "Wheel", version: 1, description: "", detector: { weights: "", classes: [] },
  layout: { anchor: "hub", match_radius: 0.15, tools: ["wrench"], slots: { hole_1: { at: [0, 0], filled_by: "b" }, hole_2: { at: [1, 0], filled_by: "b" }, hole_3: { at: [1, 1], filled_by: "b" } } },
  steps: [step("place_1", "hole_1", "installed"), step("place_2", "hole_2", "installed"), step("place_3", "hole_3", "installed"),
          step("torque_1", "hole_1", "torqued"), step("torque_2", "hole_2", "torqued"), step("torque_3", "hole_3", "torqued")],
  order: "groups", groups: [["place_1", "place_2", "place_3"], { sequence: ["torque_1", "torque_2", "torque_3"] }],
  off_sop: "Other", idle: "Idle", decisions: { accept_confidence: 0.8, hold_seconds: 1, evidence_only_after: 1 },
};
const view = { id: "wheel", file: "f", draft: true, data: DATA, warnings: [], status: {} } as unknown as SopView;

function Harness({ api, onData }: { api: BuilderApi; onData: (d: SopData) => void }) {
  const [data, setData] = useState(DATA);
  return <Providers><StepsStage api={api} view={view} data={data} onChange={(d) => { setData(d); onData(d); }} /></Providers>;
}
const api = (over: Partial<BuilderApi> = {}) => ({ generateSteps: vi.fn(async () => ({ steps: [step("place_1", "hole_1", "installed")], order: "strict", groups: null })), ...over } as unknown as BuilderApi);

test("each group is a card; a sequence group's positions set the order within it", () => {
  const onData = vi.fn();
  render(<Harness api={api()} onData={onData} />);
  const seq = screen.getByRole("region", { name: /Group 2/ });
  expect(within(seq).getByText(/in this exact order/i)).toBeInTheDocument();
  fireEvent.change(within(seq).getByLabelText("Position of torque_3"), { target: { value: "2" } });
  const last = lastData(onData);
  expect(last.groups?.[1]).toEqual({ sequence: ["torque_1", "torque_3", "torque_2"] });
  expect(last.groups?.[0]).toEqual(["place_1", "place_2", "place_3"]);
});

test("generating saves pending slot edits first: the server generates from the saved draft", async () => {
  const order: string[] = [];
  const a = api({ generateSteps: vi.fn(async () => { order.push("generate"); return { steps: [], order: "strict" as const, groups: null }; }) });
  render(<Providers><StepsStage api={a} view={view} data={{ ...DATA, steps: [] }} onChange={() => {}} flush={async () => { order.push("flush"); }} /></Providers>);
  fireEvent.click(screen.getByRole("button", { name: "Generate from slots" }));
  await waitFor(() => expect(order).toEqual(["flush", "generate"]));
});

test("generating over existing steps asks first, and a failure is shown", async () => {
  const failing = api({ generateSteps: vi.fn(async () => { throw new Error("no slots"); }) });
  render(<Harness api={failing} onData={() => {}} />);
  fireEvent.click(screen.getByRole("button", { name: "Generate from slots" }));
  const dialog = await screen.findByRole("alertdialog");
  fireEvent.click(within(dialog).getByRole("button", { name: "Replace the steps" }));
  expect(await screen.findByText("no slots")).toBeInTheDocument();
});

test("deleting a step asks first", async () => {
  const onData = vi.fn();
  render(<Harness api={api()} onData={onData} />);
  fireEvent.click(screen.getByRole("button", { name: "Delete place_2" }));
  fireEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Delete" }));
  await waitFor(() => expect((lastData(onData)).steps.map((s) => s.id)).not.toContain("place_2"));
  expect((lastData(onData)).groups?.[0]).toEqual(["place_1", "place_3"]);
});

test("the advanced settings hold the step ids and the decision thresholds, in plain words", () => {
  const onData = vi.fn();
  render(<Harness api={api()} onData={onData} />);
  fireEvent.click(screen.getByText("Advanced"));
  fireEvent.change(screen.getByLabelText(/How long hands must be away/), { target: { value: "0.8" } });
  expect((lastData(onData)).decisions).toEqual({ accept_confidence: 0.8, hold_seconds: 1, evidence_only_after: 1, hands_off_seconds: 0.8 });
  expect(screen.getByLabelText("Id of step 1")).toHaveValue("place_1");
});
