import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import type { BuilderApi, SopData, SopView, StageStatus } from "./builderApi";
import { roleOf, setRole } from "./components/builder/DetectorStage";
import { SopBuilder } from "./components/builder/SopBuilder";
import { SopsPage } from "./components/builder/SopsPage";
import { Providers } from "./ui/Providers";

const wrap = (ui: ReactNode) => render(<Providers>{ui}</Providers>);
const settle = () => act(async () => { vi.advanceTimersByTime(1000); });

const DATA: SopData = {
  sop: "wheel", name: "Wheel", version: 1, description: "",
  detector: { weights: "models/wheel/detector-1.pt", classes: ["hub", "hole_empty", "bolt", "wrench", "hand"], roles: {} },
  layout: { anchor: "", match_radius: 0.15, tools: [], slots: {} },
  steps: [], order: "strict", off_sop: "Other", idle: "Idle",
};

const STAGES: StageStatus[] = [
  { id: "detector", title: "Detector", done: false, blockers: ["Pick the anchor class", "Mark at least one class as a part"] },
  { id: "recordings", title: "Recordings", done: false, blockers: ["Record at least one train run"] },
  { id: "layout", title: "Slots", done: false, blockers: ["Add at least one slot"] },
  { id: "steps", title: "Steps", done: false, blockers: ["Add at least one step"] },
  { id: "timelines", title: "Timelines", done: false, blockers: ["Record some runs first"] },
  { id: "train", title: "Decision model", done: false, blockers: ["Train a decision model"] },
];
const STATUS = {
  errors: ["layout: anchor '' is not a detector class"], detector: true, roles: false, layout: false, steps: false,
  recordings: { train: 0, val: 0, test: 0 }, timelines: { train: 0, val: 0, test: 0 }, laya_checkpoint: null,
  published: false, loaded: false, draft: true, runs: [], stages: STAGES,
  publish: { ready: false, blockers: ["layout: anchor '' is not a detector class", "Add at least one step"] },
};
const READY = { ...STATUS, errors: [], stages: STAGES.map((s) => ({ ...s, done: true, blockers: [] })), publish: { ready: true, blockers: [] } };

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
    list: async () => [{ id: "wheel", name: "Wheel", draft: true, published: false, loaded: false, runs: 0, dataset_dir: "wheel" },
      { id: "lug_bolts_torque", name: "Wheel lug bolts", draft: false, published: true, loaded: true, runs: 25, dataset_dir: "lug_bolts" }],
    create: vi.fn(async () => view),
    get: async () => view,
    save: vi.fn(async (_id, data) => {
      saved.push(data);
      view = { ...view, data, status: data.steps.length ? READY : STATUS };
      return view;
    }),
    edit: async () => view, discard: vi.fn(async () => ({ id: "wheel", deleted: true as const })),
    publish: vi.fn(async () => (view = { ...view, draft: false, status: { ...READY, published: true, loaded: true } })),
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

test("builder: edits save by themselves, the stepper walks the stages, and publishing is reviewed first", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const api = fakeApi();
  const go = vi.fn();
  wrap(<SopBuilder sopId="wheel" api={api} go={go} />);
  await screen.findByLabelText("Role of hub");
  // The stepper shows every stage and what each one still needs.
  const stepper = screen.getByRole("navigation", { name: "Stages" });
  expect(within(stepper).getAllByRole("button").map((b) => b.textContent)).toEqual(["Detector", "Recordings", "Slots", "Steps", "Timelines", "Decision model"]);
  expect(screen.getByText("Pick the anchor class")).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText("Role of hub"), { target: { value: "anchor" } });
  fireEvent.change(screen.getByLabelText("Role of bolt"), { target: { value: "part" } });
  expect(screen.getByText("Saving…")).toBeInTheDocument();
  await settle();
  expect(api.save).toHaveBeenCalledTimes(1);
  expect(api.saved.at(-1)?.layout.anchor).toBe("hub");
  expect(api.saved.at(-1)?.detector.roles).toEqual({ bolt: "part" });
  expect(screen.getByText("All changes saved")).toBeInTheDocument();

  // Next goes to the next stage (and into the URL); the stepper jumps anywhere.
  fireEvent.click(screen.getByRole("button", { name: /^Next/ }));
  expect(go).toHaveBeenLastCalledWith("#/sops/wheel/recordings");
  expect(screen.getByRole("heading", { name: "Recordings" })).toBeInTheDocument();
  fireEvent.click(within(stepper).getByRole("button", { name: "Slots" }));
  fireEvent.click(screen.getByRole("button", { name: "Add slot" }));
  expect(screen.getByLabelText("x of slot_1")).toHaveValue(0.5);
  fireEvent.click(screen.getByRole("button", { name: /^Back/ }));
  expect(screen.getByRole("heading", { name: "Recordings" })).toBeInTheDocument();

  fireEvent.click(within(stepper).getByRole("button", { name: "Steps" }));
  fireEvent.click(screen.getByRole("button", { name: "Generate from slots" }));
  await screen.findByDisplayValue("place_hole_1");
  expect(screen.getByLabelText("Order")).toHaveValue("groups");
  expect(screen.getByLabelText("Group 2 kind")).toHaveValue("sequence");
  await settle();

  // Review and publish: a summary, then the SOP goes live.
  fireEvent.click(screen.getByRole("button", { name: "Review and publish" }));
  const dialog = await screen.findByRole("dialog", { name: "Publish Wheel?" });
  expect(within(dialog).getByText(/2 steps/)).toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole("button", { name: "Publish" }));
  await waitFor(() => expect(api.publish).toHaveBeenCalled());
  expect(await screen.findByText(/Wheel is published/)).toBeInTheDocument();
  expect(api.saved.at(-1)?.steps.map((s) => s.id)).toEqual(["place_hole_1", "torque_hole_1"]);
  expect(await screen.findByRole("button", { name: "Edit" })).toBeInTheDocument();  // once the dialog has faded out
  vi.useRealTimers();
});

test("builder: the publish review lists what is still missing and keeps Publish disabled", async () => {
  const api = fakeApi();
  wrap(<SopBuilder sopId="wheel" api={api} />);
  await screen.findByLabelText("Role of hub");
  fireEvent.click(screen.getByRole("button", { name: "Review and publish" }));
  const dialog = await screen.findByRole("dialog", { name: "Publish Wheel?" });
  expect(within(dialog).getByText("Add at least one step")).toBeInTheDocument();
  expect(within(dialog).getByRole("button", { name: "Publish" })).toBeDisabled();
});

test("builder: deleting a draft asks for its id first, then goes back to the list", async () => {
  const api = fakeApi();
  const go = vi.fn();
  wrap(<SopBuilder sopId="wheel" api={api} go={go} />);
  await screen.findByLabelText("Role of hub");
  fireEvent.click(screen.getByRole("button", { name: "Delete draft" }));
  const dialog = await screen.findByRole("alertdialog");
  expect(within(dialog).getByRole("button", { name: "Delete" })).toBeDisabled();
  fireEvent.change(within(dialog).getByLabelText("Type wheel to confirm"), { target: { value: "wheel" } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }));
  await waitFor(() => expect(api.discard).toHaveBeenCalledWith("wheel"));
  expect(go).toHaveBeenCalledWith("#/sops");
});

test("builder: settings change the name, the description and the recordings folder", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const api = fakeApi();
  wrap(<SopBuilder sopId="wheel" api={api} />);
  await screen.findByLabelText("Role of hub");
  fireEvent.click(screen.getByRole("button", { name: "Settings" }));
  const dialog = await screen.findByRole("dialog", { name: "SOP settings" });
  fireEvent.change(within(dialog).getByLabelText("Name"), { target: { value: "Wheel bolts" } });
  fireEvent.change(within(dialog).getByLabelText("Description"), { target: { value: "Five bolts." } });
  fireEvent.change(within(dialog).getByLabelText("Recordings"), { target: { value: "lug_bolts" } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply" }));
  await settle();
  expect(api.saved.at(-1)?.name).toBe("Wheel bolts");
  expect(api.saved.at(-1)?.description).toBe("Five bolts.");
  expect(api.saved.at(-1)?.dataset?.dir).toBe("lug_bolts");
  expect(screen.getByRole("heading", { name: "Wheel bolts" })).toBeInTheDocument();
  vi.useRealTimers();
});

test("sops page: lists the SOPs and starts a new one that reuses another SOP's recordings", async () => {
  const api = fakeApi();
  const go = vi.fn();
  wrap(<SopsPage api={api} go={go} />);
  await screen.findByText("Wheel lug bolts");
  expect(screen.getByText(/25 recordings/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /Wheel lug bolts/ })).toHaveAttribute("href", "#/sops/lug_bolts_torque/detector");
  fireEvent.click(screen.getByRole("button", { name: "New SOP" }));
  const dialog = await screen.findByRole("dialog", { name: "New SOP" });
  fireEvent.change(within(dialog).getByLabelText(/^Name/), { target: { value: "Wheel Bolts v2" } });
  expect(within(dialog).getByLabelText("Id")).toHaveValue("wheel_bolts_v2");
  fireEvent.change(within(dialog).getByLabelText("Reuse recordings from"), { target: { value: "lug_bolts_torque" } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Create" }));
  await waitFor(() => expect(api.create).toHaveBeenCalledWith("wheel_bolts_v2", "Wheel Bolts v2", "", "lug_bolts_torque"));
  expect(go).toHaveBeenCalledWith("#/sops/wheel/detector");
});

test("timeline page: starts from the draft, a key press marks a step change, save writes the rows", async () => {
  const { TimelinePage } = await import("./components/builder/TimelinePage");
  const api = fakeApi();
  wrap(<TimelinePage sopId="wheel" run="r1" api={api} />);
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
  await screen.findByText("Timeline saved");
  expect(screen.getByRole("link", { name: "Back to the timelines" })).toHaveAttribute("href", "#/sops/wheel/timelines");
});

test("timeline page: a run without a browser copy offers to prepare one", async () => {
  const { TimelinePage } = await import("./components/builder/TimelinePage");
  const api = fakeApi();
  api.timeline = vi.fn(async () => ({ ...TIMELINE, preview: false }));
  api.startJob = vi.fn(async () => ({ id: "j1" } as never));
  api.job = vi.fn(async () => ({ id: "j1", title: "Prepare video", state: "running", steps: [], cancel_requested: false } as never));
  wrap(<TimelinePage sopId="wheel" run="r1" api={api} />);
  fireEvent.click(await screen.findByRole("button", { name: "Prepare the video" }));
  await waitFor(() => expect(api.startJob).toHaveBeenCalledWith("previews", { sop_file: "sops/drafts/wheel.yaml", runs: ["r1"] }));
});
