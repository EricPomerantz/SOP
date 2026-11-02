import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { BuilderApi, SopData, SopView } from "../../builderApi";
import { Providers } from "../../ui/Providers";
import { DetectorStage } from "./DetectorStage";

const DATA: SopData = {
  sop: "wheel", name: "Wheel", version: 1, description: "",
  detector: { weights: "models/wheel/best.pt", classes: [], roles: {} },
  layout: { anchor: "", match_radius: 0.15, tools: [], slots: {} }, steps: [], order: "strict", off_sop: "", idle: "",
};
const view = (data: SopData): SopView => ({ id: "wheel", file: "sops/drafts/wheel.yaml", draft: true, data, warnings: [], status: {} as never });
const api = (over: Partial<BuilderApi> = {}) => ({
  models: async () => [{ path: "models/lug_bolts/best.pt", classes: ["hub", "hole_empty", "lug_bolt_red", "hand"] }, { path: "models/fan/detector-2.pt", classes: ["chassis", "fan"] }],
  useModel: vi.fn(async () => view({ ...DATA, detector: { ...DATA.detector, classes: ["hub", "hole_empty", "lug_bolt_red", "hand"] } })),
  uploadDetector: vi.fn(), ...over,
} as unknown as BuilderApi);

test("models under models/ are offered by name, and the chosen one becomes the SOP's detector", async () => {
  const a = api();
  const onView = vi.fn();
  render(<Providers><DetectorStage api={a} view={view(DATA)} data={DATA} onChange={() => {}} onView={onView} /></Providers>);
  expect(await screen.findByText("No detector yet")).toBeInTheDocument();
  const option = await screen.findByRole("radio", { name: /lug_bolts \/ best\.pt/ });
  expect(screen.getByText("4 classes")).toBeInTheDocument();
  fireEvent.click(option);
  fireEvent.click(screen.getByRole("button", { name: "Use this model" }));
  await waitFor(() => expect(a.useModel).toHaveBeenCalledWith("wheel", "models/lug_bolts/best.pt"));
  expect(onView).toHaveBeenCalled();
});

test("a failure to list the models is shown, not swallowed", async () => {
  const a = api({ models: async () => { throw new Error("models/ is not mounted"); } });
  render(<Providers><DetectorStage api={a} view={view(DATA)} data={DATA} onChange={() => {}} onView={() => {}} /></Providers>);
  expect(await screen.findByRole("alert")).toHaveTextContent("models/ is not mounted");
});

test("with classes, each gets a role and the file path sits under Advanced", async () => {
  const data = { ...DATA, detector: { ...DATA.detector, classes: ["hub", "lug_bolt_red", "hand"] } };
  render(<Providers><DetectorStage api={api()} view={view(data)} data={data} onChange={() => {}} onView={() => {}} /></Providers>);
  expect(await screen.findByLabelText("Role of hub")).toHaveValue("ignore");
  expect(screen.getByLabelText("Role of hand")).toBeDisabled();
  expect(screen.getByText("models/wheel/best.pt")).not.toBeVisible();
  fireEvent.click(screen.getByText("Advanced"));
  expect(screen.getByText("models/wheel/best.pt")).toBeVisible();
});
