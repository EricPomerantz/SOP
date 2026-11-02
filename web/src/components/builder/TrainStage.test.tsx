import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { BuilderApi, Checkpoints, SopView } from "../../builderApi";
import { Providers } from "../../ui/Providers";
import { TrainStage } from "./TrainStage";

const bench = { requests: 100, errors: 0, current_step_accuracy: 0.92, labelled: 1240, echo_rate: 0.5, accuracy_when_acting: 0.9,
  off_sop_false_negative_rate: 0.1, off_sop_false_positive_rate: 0.02, anomaly_false_positive_rate: 0.01, request_ms_p50: 80 };
const CPS: Checkpoints = { active: null, shared: ["laya-sop-v2-e12"], versions: [{ checkpoint: "wheel-v1", report: { sop: "wheel", checkpoint: "wheel-v1", active: null, training_set: null, bench, bench_active: null, replay: [] } }] };
const view = (timelines = { train: 1, val: 1, test: 1 }) => ({ id: "wheel", file: "f", draft: true, data: {}, warnings: [],
  status: { errors: [], recordings: { train: 1, val: 1, test: 1 }, timelines, laya_checkpoint: null } } as unknown as SopView);
const api = () => ({ checkpoints: async () => CPS, jobs: async () => [], startJob: vi.fn(async () => ({ id: "j" })), job: vi.fn(async () => ({ id: "j", title: "Train", state: "queued", steps: [], created: 0 })),
  activate: vi.fn(async () => ({ ...CPS, active: "wheel-v1" })) } as unknown as BuilderApi);

test("a checkpoint's results read in plain words, and activating asks first", async () => {
  const a = api();
  const onReload = vi.fn();
  render(<Providers><TrainStage api={a} view={view()} onReload={onReload} /></Providers>);
  expect(await screen.findByText("92%")).toBeInTheDocument();
  expect(screen.getByText(/right step/i)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Switch on wheel-v1" }));
  fireEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Switch it on" }));
  await waitFor(() => expect(a.activate).toHaveBeenCalledWith("wheel", "wheel-v1"));
  await waitFor(() => expect(onReload).toHaveBeenCalled());
});

test("training says why it can't start yet", async () => {
  render(<Providers><TrainStage api={api()} view={view({ train: 1, val: 0, test: 0 })} onReload={() => {}} /></Providers>);
  expect(screen.getByRole("button", { name: "Train a new model" })).toBeDisabled();
  expect(screen.getByText(/needs at least one val and one test run with a timeline/i)).toBeInTheDocument();
});
