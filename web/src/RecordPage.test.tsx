import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { Api } from "./api";
import { RecordPage } from "./components/RecordPage";
import { TorqueButtons } from "./components/TorqueButtons";
import type { RecordingStatus } from "./types";

const STATUS: RecordingStatus = {
  run_id: "r01", sop: "lug_bolts_torque", split: "train", source: "rtsp:0", folder: "/datasets/lug_bolts/runs/r01",
  seconds: 12.5, frames: 312, camera_delay_ms: 387.1, presses: [], error: null,
};

function api(): Api {
  let recording: RecordingStatus | null = null; // like the server: the status while recording, null otherwise
  return {
    getSops: async () => [], getCurrent: async () => null, startRun: vi.fn(), stopRun: vi.fn(), control: vi.fn(),
    reportUrl: () => "", videoUrl: "/video.mjpg",
    getSources: async () => [{ id: "test:Install_12", kind: "file", label: "Install_12" }, { id: "rtsp:0", kind: "rtsp", label: "RTSP camera 1" }],
    getRecordingSops: async () => [{ file: "sops/drafts/lug_bolts.yaml", id: "lug_bolts_torque", name: "Wheel lug bolts", dataset: "/datasets/lug_bolts", torque: true }],
    getRecording: async () => recording,
    startRecording: vi.fn(async () => (recording = STATUS)),
    recordingTorque: vi.fn(async () => ({ time_s: 4.25 })),
    discardRecording: vi.fn(async () => { recording = null; return { run_id: "r01", discarded: true }; }),
    stopRecording: vi.fn(async () => {
      recording = null;
      return { run_id: "r01", folder: "/datasets/lug_bolts/runs/r01", frames: 312, seconds: 12.5, presses: 1, split: "train" };
    }),
  };
}

test("record: opened from the builder, it links back and says what to do with the run next", async () => {
  const a = api();
  render(<RecordPage api={a} poll={60_000} back="#/sops/lug_bolts_torque/recordings" />);
  await screen.findByText("Wheel lug bolts");
  expect(screen.getByRole("link", { name: "Back to the SOP" })).toHaveAttribute("href", "#/sops/lug_bolts_torque/recordings");
  fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
  await screen.findByLabelText("Recording");
  fireEvent.click(screen.getByRole("button", { name: "Stop and save" }));
  await screen.findByText(/Saved r01 \(train\)/);
  expect(screen.getByRole("link", { name: /label its timeline/ })).toHaveAttribute("href", "#/sops/lug_bolts_torque/timelines");
  expect(screen.queryByText(/steps.csv/)).toBeNull();
});

test("record: start, press torque, stop and save", async () => {
  const a = api();
  render(<RecordPage api={a} poll={60_000} />);
  await screen.findByText("Wheel lug bolts");
  fireEvent.change(screen.getByPlaceholderText(/r07/), { target: { value: "r01" } });
  fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
  await screen.findByLabelText("Recording");
  expect(a.startRecording).toHaveBeenCalledWith("sops/drafts/lug_bolts.yaml", "rtsp:0", "train", "r01");
  expect(screen.getByAltText("Camera preview")).toHaveAttribute("src", "/video.mjpg");
  fireEvent.click(screen.getByRole("button", { name: /Torque reached/ }));
  await waitFor(() => expect(screen.getByText("Torque reached at 4.25 s")).toBeTruthy());
  expect(a.recordingTorque).toHaveBeenCalledWith(true);
  fireEvent.click(screen.getByRole("button", { name: "Stop and save" }));
  await waitFor(() => expect(screen.getByText(/Saved r01 \(train\)/)).toBeTruthy());
});

test("torque buttons answer to T and N, but not while typing", async () => {
  const onPress = vi.fn(async (ok: boolean) => (ok ? "yes" : "no"));
  render(<><input aria-label="field" /><TorqueButtons onPress={onPress} /></>);
  await act(async () => { fireEvent.keyDown(window, { key: "t" }); });
  await act(async () => { fireEvent.keyDown(window, { key: "N" }); });
  await act(async () => { fireEvent.keyDown(screen.getByLabelText("field"), { key: "t" }); });
  expect(onPress.mock.calls).toEqual([[true], [false]]);
});

test("a SOP file that fails to load is named, with the reason", async () => {
  const a = { ...api(), getRecordingSops: async () => [],
    getRecordingSopErrors: async () => [{ file: "sops/drafts/lug_bolts.yaml", error: "layout.anchor_follw: Extra inputs are not permitted" }] };
  render(<RecordPage api={a} poll={60_000} />);
  expect(await screen.findByText(/lug_bolts.yaml could not be loaded: layout.anchor_follw/)).toBeTruthy();
});


test("discard asks first, and does nothing if you cancel", async () => {
  const a = api();
  let answer = false;
  render(<RecordPage api={a} poll={60_000} confirm={() => answer} />);
  await screen.findByText("Wheel lug bolts");
  fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
  await screen.findByLabelText("Recording");
  fireEvent.click(screen.getByRole("button", { name: "Discard (don't save)" }));
  await waitFor(() => expect(a.discardRecording).not.toHaveBeenCalled());
  expect(screen.getByLabelText("Recording")).toBeTruthy();
  answer = true;
  fireEvent.click(screen.getByRole("button", { name: "Discard (don't save)" }));
  await waitFor(() => expect(screen.getByText(/Discarded r01: nothing was saved/)).toBeTruthy());
  expect(a.discardRecording).toHaveBeenCalledTimes(1);
  expect(a.stopRecording).not.toHaveBeenCalled();
});
