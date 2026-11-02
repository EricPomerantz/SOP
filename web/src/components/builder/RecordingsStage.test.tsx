import { render, screen } from "@testing-library/react";
import type { BuilderApi, SopData, SopView } from "../../builderApi";
import { Providers } from "../../ui/Providers";
import { RecordingsStage } from "./RecordingsStage";

const DATA: SopData = {
  sop: "wheel", name: "Wheel", version: 1, description: "", detector: { weights: "", classes: [] },
  layout: { anchor: "", match_radius: 0.15, slots: {} }, steps: [], order: "strict", off_sop: "", idle: "",
  dataset: { format: "timeline", dir: "lug_bolts" },
};
const runs = [{ id: "correct2", split: "train", steps: true, draft: false, torque: true }, { id: "testcorrect1", split: "test", steps: false, draft: true, torque: true }];
const view = (over: Partial<SopView["status"]> = {}, data = DATA): SopView => ({
  id: "wheel", file: "sops/drafts/wheel.yaml", draft: true, data, warnings: [],
  status: { recordings: { train: 16, val: 2, test: 7 }, timelines: { train: 16, val: 2, test: 0 }, runs, ...over } as never,
});
const api = { list: async () => [{ id: "lug_bolts_torque", name: "Wheel lug bolts", draft: false, published: true, loaded: true, runs: 25, dataset_dir: "lug_bolts" }] } as unknown as BuilderApi;

test("counts per split say what each split is for, and recording comes back to this stage", async () => {
  render(<Providers><RecordingsStage api={api} view={view()} /></Providers>);
  expect(screen.getByText("16")).toBeInTheDocument();
  expect(screen.getByText(/teach the model/)).toBeInTheDocument();
  expect(screen.getByText(/never trained on/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Record a run" })).toHaveAttribute("href",
    `#/record?sop=${encodeURIComponent("sops/drafts/wheel.yaml")}&back=${encodeURIComponent("#/sops/wheel/recordings")}`);
  expect(screen.getByText("correct2")).toBeInTheDocument();
  expect(await screen.findByText(/Shared with Wheel lug bolts/)).toBeInTheDocument();
});

test("without recordings the stage says so and points at the Record page", () => {
  render(<Providers><RecordingsStage api={api} view={view({ recordings: { train: 0, val: 0, test: 0 }, runs: [] }, { ...DATA, dataset: { dir: "wheel" } })} /></Providers>);
  expect(screen.getByText("No recordings yet")).toBeInTheDocument();
  expect(screen.queryByText(/Shared with/)).toBeNull();
});
