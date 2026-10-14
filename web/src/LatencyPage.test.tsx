import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { Api } from "./api";
import { clockOffset, clockText, LatencyPage } from "./components/LatencyPage";

vi.mock("qrcode", () => ({ default: { toCanvas: vi.fn() } }));

function api(overrides: Partial<Api> = {}): Api {
  return {
    getSops: async () => [],
    getSources: async () => [{ id: "test:Install_12", kind: "file", label: "Install_12" }, { id: "rtsp:0", kind: "rtsp", label: "RTSP camera 1" }],
    getCurrent: async () => null,
    startRun: vi.fn(), stopRun: vi.fn(), control: vi.fn(),
    reportUrl: () => "", videoUrl: "",
    latencyClock: async () => ({ now_ms: 1_000_000 }),
    getLatency: async () => ({}),
    measureLatency: vi.fn(async () => ({ frames: 250, fps: 25, decoded: 240, delay_ms: 312.5, p10_ms: 290, p90_ms: 340, saved: true })),
    ...overrides,
  };
}

test("clock offset is the server time minus the round trip's midpoint", () => {
  expect(clockOffset(1000, 5000, 1040)).toBe(3980);
});

test("the QR payload is short: prefix, hold time, time modulo 1e8 ms", () => {
  expect(clockText(1_791_234_567_890.4)).toBe("SVL:50:34567890");
});

test("only live cameras are offered, and a measurement shows the delay", async () => {
  const a = api();
  render(<LatencyPage api={a} />);
  const select = await screen.findByRole("combobox");
  expect([...(select as HTMLSelectElement).options].map((o) => o.textContent)).toEqual(["RTSP camera 1"]);
  fireEvent.click(screen.getByRole("button", { name: "Measure" }));
  await waitFor(() => expect(screen.getByLabelText("Result")).toHaveTextContent("312.5 ms"));
  expect(a.measureLatency).toHaveBeenCalledWith("rtsp:0", 10, true);
});

test("without a live camera it says how to add one", async () => {
  render(<LatencyPage api={api({ getSources: async () => [] })} />);
  expect(await screen.findByText(/SOPVISION_RTSP_URLS/)).toBeTruthy();
});
