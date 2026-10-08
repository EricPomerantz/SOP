import { act, render, screen, within } from "@testing-library/react";
import App from "./App";
import type { Api } from "./api";
import stream from "./fixtures/run_stream.json";
import type { RunSnapshot, Sop, WsMessage } from "./types";

const STEPS = ["Fan 1", "Fan 2", "Fan 3", "Fan 4", "Fan 5", "Fan 6", "PSU 1", "PSU 2", "Cover"];
const SOP: Sop = {
  id: "server_fan_install", name: "Bench assembly", version: 1, description: "Install six fans, two power supplies and the cover on a 1U server.",
  steps: STEPS.map((label) => ({ id: label.toLowerCase().replace(" ", "_"), label, text: label })),
};
const RUNNING: RunSnapshot = { run_id: "r1", sop: SOP.id, source: "test:Install_12", steps: [], alerts: [], last_decision: null };

function fakeApi(current: RunSnapshot | null, model?: string): Api {
  return {
    ...(model ? { getConfig: async () => ({ decision_model: model, decision_mode: model }) } : {}),
    getSops: async () => [SOP],
    getSources: async () => [{ id: "test:Install_12", kind: "file", label: "Install_12 (test run)" }],
    getCurrent: async () => current,
    startRun: vi.fn(async () => ({ run_id: "r2", sop: SOP.id, source: "test:Install_12" })),
    stopRun: vi.fn(async () => {}),
    control: vi.fn(async () => {}),
    reportUrl: (id: string) => `/runs/${id}/report?format=csv`,
    videoUrl: "/video.mjpg",
  };
}

function fakeSocket() {
  let handler: (m: WsMessage) => void = () => {};
  return {
    connect: (onMessage: (m: WsMessage) => void) => { handler = onMessage; return () => {}; },
    push: (m: WsMessage) => act(() => handler(m)),
  };
}

test("App renders a recorded stream", async () => {
  const sock = fakeSocket();
  render(<App api={fakeApi(RUNNING)} connect={sock.connect} />);
  const checklist = await screen.findByRole("list", { name: "Checklist" });
  expect(within(checklist).getAllByRole("listitem")).toHaveLength(9);
  for (const m of stream as WsMessage[]) sock.push(m);
  expect(within(checklist).getByText("Fan 1").closest("li")).toHaveAttribute("data-status", "done");
  expect(within(checklist).getByText("7.1 s")).toBeInTheDocument();
  expect(screen.getByLabelText("Jev now")).toHaveTextContent("Fan 2");
  expect(screen.getByLabelText("Jev now")).toHaveTextContent("0.91");
  const log = screen.getByRole("list", { name: "Event log" });
  expect(log).toHaveTextContent("Off-SOP activity");
  expect(log).toHaveTextContent("Fan 1 complete · by Jev");
  expect(screen.getByRole("status")).toHaveTextContent("On track");
});

test("shows a banner when the chassis is out of view", async () => {
  const sock = fakeSocket();
  render(<App api={fakeApi(RUNNING)} connect={sock.connect} />);
  await screen.findByRole("list", { name: "Checklist" });
  sock.push({ type: "scene", t: 3, scene: { chassis: { visible: false, conf: null } } });
  expect(screen.getByText("Chassis not visible")).toBeInTheDocument();
});

test("with no run in progress it offers sources to start one", async () => {
  render(<App api={fakeApi(null)} connect={fakeSocket().connect} />);
  expect(await screen.findByRole("option", { name: "Install_12 (test run)" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Start" })).toBeEnabled();
});

test("the top bar shows the SOP's own name", async () => {
  render(<App api={fakeApi(null)} connect={fakeSocket().connect} />);
  expect(await screen.findByText("Bench assembly")).toBeInTheDocument();
});


test("before any run the strip names the configured model, not Jev", async () => {
  render(<App api={fakeApi(null, "laya")} connect={fakeSocket().connect} />);
  expect(await screen.findByLabelText("Laya now")).toHaveTextContent("Waiting for the first decision");
  expect(screen.queryByLabelText("Jev now")).toBeNull();
});
