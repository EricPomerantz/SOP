import { render, screen } from "@testing-library/react";
import type { BuilderApi, Job } from "../../builderApi";
import { Providers } from "../../ui/Providers";
import { JobPanel } from "./JobPanel";

const base: Job = {
  id: "j1", kind: "autofit", title: "Suggest slots", sop: "wheel", state: "running", error: null, created: 1000,
  steps: [{ name: "detect", argv: [], state: "done", started: 1000, ended: 1060 }, { name: "cluster", argv: [], state: "running", started: 1060, ended: null },
          { name: "frames", argv: [], state: "queued", started: null, ended: null }],
  result: {}, params: {}, cancel_requested: false, log: "line 1\nline 2",
};
const api = (job: Job) => ({ job: vi.fn(async () => job), cancelJob: vi.fn() } as unknown as BuilderApi);

test("a running job shows its progress, the step under way and the time it has taken", async () => {
  vi.spyOn(Date, "now").mockReturnValue(1140 * 1000);
  render(<Providers><JobPanel api={api(base)} jobId="j1" poll={60_000} /></Providers>);
  expect(await screen.findByText("Suggest slots")).toBeInTheDocument();
  expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "33");
  expect(screen.getByText(/Running: cluster · 2 min 20 s/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();
  vi.restoreAllMocks();
});

test("a queued job explains the wait and a failed one opens its log", async () => {
  const { unmount } = render(<Providers><JobPanel api={api({ ...base, state: "queued" })} jobId="j1" poll={60_000} /></Providers>);
  expect(await screen.findByText(/Waiting for the worker/)).toBeInTheDocument();
  unmount();
  const failed = { ...base, state: "failed", error: "laya-train exited with 1", steps: base.steps.map((s) => ({ ...s, state: s.state === "running" ? "failed" : s.state })) };
  render(<Providers><JobPanel api={api(failed)} jobId="j1" poll={60_000} /></Providers>);
  expect(await screen.findByText("laya-train exited with 1")).toBeInTheDocument();
  expect(screen.getByText(/Failed/)).toBeInTheDocument();
  expect(screen.getByText(/line 2/)).toBeVisible();
  expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
});
