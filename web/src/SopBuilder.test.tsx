import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { BuilderApi, SopData, SopView } from "./builderApi";
import { roleOf, setRole } from "./components/builder/DetectorStage";
import { SopBuilder } from "./components/builder/SopBuilder";
import { SopsPage } from "./components/builder/SopsPage";

const DATA: SopData = {
  sop: "wheel", name: "Wheel", version: 1, description: "",
  detector: { weights: "models/wheel/detector-1.pt", classes: ["hub", "hole_empty", "bolt", "wrench", "hand"], roles: {} },
  layout: { anchor: "", match_radius: 0.15, tools: [], slots: {} },
  steps: [], order: "strict", off_sop: "Other", idle: "Idle",
};

const STATUS = {
  errors: ["layout: anchor '' is not a detector class"], detector: true, roles: false, layout: false, steps: false,
  recordings: { train: 0, val: 0, test: 0 }, timelines: { train: 0, val: 0, test: 0 }, laya_checkpoint: null,
  published: false, loaded: false, draft: true, runs: [],
};

const TIMELINE = {
  run: "r1", sop_file: "sops/drafts/wheel.yaml", fps: 10, duration: 20, preview: true, steps: null, draft: [["place_hole_1", 0, 12], ["idle", 12, 20]] as [string, number, number][],
  draft_info: { alerts: [], not_done: ["torque_hole_1"] }, torque: [{ time_s: 15, ok: true }],
  step_ids: ["place_hole_1", "torque_hole_1"], labels: { place_hole_1: "Bolt 1", torque_hole_1: "Torque 1" },
};

function fakeApi(): BuilderApi & { saved: SopData[] } {
  let view: SopView = { id: "wheel", file: "sops/drafts/wheel.yaml", draft: true, data: DATA, warnings: [], status: STATUS };
  const saved: SopData[] = [];
  return {
    saved,
    list: async () => [{ id: "wheel", name: "Wheel", draft: true, published: false, loaded: false }],
    create: vi.fn(async () => view),
    get: async () => view,
    save: vi.fn(async (_id, data) => { saved.push(data); view = { ...view, data }; return view; }),
    edit: async () => view, discard: async () => view,
    publish: vi.fn(async () => (view = { ...view, draft: false, status: { ...STATUS, errors: [], published: true, loaded: true } })),
    generateSteps: async () => ({
      steps: [{ id: "place_hole_1", text: "Install the part in hole 1.", evidence: { hole_1: "installed" } },
              { id: "torque_hole_1", text: "Torque it.", evidence: { hole_1: "torqued" } }],
      order: "groups", groups: [["place_hole_1"], { sequence: ["torque_hole_1"] }],
    }),
    models: async () => [{ path: "models/lug_bolts/best.pt", classes: ["hub"] }],
    uploadDetector: vi.fn(async () => view), useModel: vi.fn(async () => view),
    autofit: async () => null, frameUrl: () => "",
    timeline: vi.fn(async () => TIMELINE), saveTimeline: vi.fn(async (_id, _run, rows) => ({ ...TIMELINE, steps: rows })),
    previewUrl: () => "/p.webm",
    checkpoints: async () => ({ active: null, versions: [], shared: [] }),
    activate: vi.fn(async () => ({ active: "wheel-v1", versions: [], shared: [] })),
    jobs: async () => [], job: vi.fn(), startJob: vi.fn(), cancelJob: vi.fn(),
  };
}

test("roles: one anchor, tools in the layout, parts and markers in the detector", () => {
  let d = setRole(DATA, "hub", "anchor");
  d = setRole(d, "bolt", "part");
  d = setRole(d, "hole_empty", "empty");
  d = setRole(d, "wrench", "tool");
  expect(d.layout.anchor).toBe("hub");
  expect(d.layout.tools).toEqual(["wrench"]);
  expect(d.detector.roles).toEqual({ bolt: "part", hole_empty: "empty" });
  d = setRole(d, "bolt", "anchor");
  expect([d.layout.anchor, d.detector.roles]).toEqual(["bolt", { hole_empty: "empty" }]);
  expect(roleOf(d, "hub")).toBe("ignore");
  expect(roleOf(d, "hand")).toBe("hand");
});

test("builder: set class roles, generate steps, save and publish", async () => {
  const api = fakeApi();
  render(<SopBuilder sopId="wheel" api={api} />);
  await screen.findByLabelText("Role of hub");
  fireEvent.change(screen.getByLabelText("Role of hub"), { target: { value: "anchor" } });
  fireEvent.change(screen.getByLabelText("Role of bolt"), { target: { value: "part" } });
  expect(screen.getByText("unsaved")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(api.saved.at(-1)?.layout.anchor).toBe("hub"));
  expect(api.saved.at(-1)?.detector.roles).toEqual({ bolt: "part" });

  fireEvent.click(screen.getByRole("button", { name: /3 Slots/ }));
  fireEvent.click(screen.getByRole("button", { name: "Add slot" }));
  expect(screen.getByLabelText("x of slot_1")).toHaveValue(0.5);

  fireEvent.click(screen.getByRole("button", { name: /4 Steps/ }));
  fireEvent.click(screen.getByRole("button", { name: "Generate from slots" }));
  await screen.findByDisplayValue("place_hole_1");
  expect(screen.getByLabelText("Order")).toHaveValue("groups");
  expect(screen.getByLabelText("Group 2 kind")).toHaveValue("sequence");

  fireEvent.click(screen.getByRole("button", { name: "Publish" }));
  await screen.findByText(/Published: the SOP is in the dashboard/);
  expect(api.saved.at(-1)?.steps.map((s) => s.id)).toEqual(["place_hole_1", "torque_hole_1"]);
  expect(screen.getByRole("button", { name: "Edit" })).toBeTruthy();
});

test("sops page: create a draft and open it", async () => {
  const api = fakeApi();
  const go = vi.fn();
  render(<SopsPage api={api} go={go} />);
  await screen.findByText("Wheel");
  fireEvent.change(screen.getByPlaceholderText(/wheel_lug_bolts/), { target: { value: "wheel" } });
  fireEvent.click(screen.getByRole("button", { name: "Create draft" }));
  await waitFor(() => expect(go).toHaveBeenCalledWith("#/sops/wheel"));
  expect(api.create).toHaveBeenCalledWith("wheel", "", "");
});

test("timeline page: starts from the draft, a key press marks a step change, save writes the rows", async () => {
  const { TimelinePage } = await import("./components/builder/TimelinePage");
  const api = fakeApi();
  render(<TimelinePage sopId="wheel" run="r1" api={api} />);
  await screen.findByLabelText("Step of row 2");
  expect(screen.getByLabelText("Step of row 1")).toHaveValue("place_hole_1");
  fireEvent.click(screen.getByRole("button", { name: /Torque 1/ }));  // at t=0: the whole first row becomes torque
  expect(screen.getByLabelText("Step of row 1")).toHaveValue("torque_hole_1");
  fireEvent.keyDown(window, { key: "z", ctrlKey: true });  // undo
  expect(screen.getByLabelText("Step of row 1")).toHaveValue("place_hole_1");
  fireEvent.keyDown(window, { key: "2" });  // the second step's key
  expect(screen.getByLabelText("Step of row 1")).toHaveValue("torque_hole_1");
  fireEvent.change(screen.getByLabelText("End of row 1"), { target: { value: "8" } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(api.saveTimeline).toHaveBeenCalledWith("wheel", "r1", [["torque_hole_1", 0, 8], ["idle", 8, 20]]));
  await screen.findByText("Saved steps.csv");
});

test("timeline page: a run without a browser copy offers to prepare one", async () => {
  const { TimelinePage } = await import("./components/builder/TimelinePage");
  const api = fakeApi();
  api.timeline = vi.fn(async () => ({ ...TIMELINE, preview: false }));
  api.startJob = vi.fn(async () => ({ id: "j1" } as never));
  api.job = vi.fn(async () => ({ id: "j1", title: "Prepare video", state: "running", steps: [], cancel_requested: false } as never));
  render(<TimelinePage sopId="wheel" run="r1" api={api} />);
  fireEvent.click(await screen.findByRole("button", { name: "Prepare the video" }));
  await waitFor(() => expect(api.startJob).toHaveBeenCalledWith("previews", { sop_file: "sops/drafts/wheel.yaml", runs: ["r1"] }));
});
