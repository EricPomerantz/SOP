import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { BuilderApi, SopView } from "../../builderApi";
import { Providers } from "../../ui/Providers";
import { TimelinesStage } from "./TimelinesStage";

const runs = [{ id: "correct10", split: "train", steps: true, draft: false, torque: true }, { id: "r2", split: "test", steps: false, draft: true, torque: false },
  { id: "r3", split: "val", steps: false, draft: false, torque: false }];
const view = { id: "wheel", file: "sops/drafts/wheel.yaml", draft: true, data: { dataset: { dir: "wheel" } }, warnings: [],
  status: { steps: true, runs, recordings: { train: 1, val: 1, test: 1 }, timelines: { train: 1, val: 0, test: 0 } } } as unknown as SopView;
const api = () => ({ jobs: async () => [], startJob: vi.fn(async () => ({ id: "j1" })), job: vi.fn(async () => ({ id: "j1", title: "Draft", state: "queued", steps: [], created: 0, cancel_requested: false })) } as unknown as BuilderApi);

test("each run says where its timeline is, and one run can be drafted on its own", async () => {
  const a = api();
  render(<Providers><TimelinesStage api={a} view={view} onReload={() => {}} /></Providers>);
  expect(screen.getByText("1 of 3 runs have a saved timeline")).toBeInTheDocument();
  expect(screen.getByText("Draft ready to review")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Review r2" })).toHaveAttribute("href", "#/sops/wheel/runs/r2");
  fireEvent.click(screen.getByRole("button", { name: "Draft correct10" }));
  await waitFor(() => expect(a.startJob).toHaveBeenCalledWith("draft_timelines", { sop_file: "sops/drafts/wheel.yaml", runs: ["correct10"] }));
});

test("drafting only the runs without a timeline sends just those", async () => {
  const a = api();
  render(<Providers><TimelinesStage api={a} view={view} onReload={() => {}} /></Providers>);
  fireEvent.click(screen.getByRole("button", { name: "Draft the 2 runs without a timeline" }));
  await waitFor(() => expect(a.startJob).toHaveBeenCalledWith("draft_timelines", { sop_file: "sops/drafts/wheel.yaml", runs: ["r2", "r3"] }));
});
