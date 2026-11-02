import { act, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { SAVE_TIMEOUT_MS, useAutoSave } from "./useAutoSave";

type Doc = { n: number };

function Harness({ save, onSaved }: { save: (d: Doc, signal: AbortSignal) => Promise<string>; onSaved?: (v: string) => void }) {
  const [data, setData] = useState<Doc>({ n: 0 });
  const auto = useAutoSave({ data, enabled: true, save, onSaved: onSaved ?? (() => {}), delay: 800 });
  return (
    <div>
      <button onClick={() => setData({ n: data.n + 1 })}>edit</button>
      <button onClick={() => void auto.flush()}>flush</button>
      <output>{auto.state}{auto.error ? `:${auto.error}` : ""}</output>
    </div>
  );
}

beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
afterEach(() => vi.useRealTimers());

test("edits are saved once after a pause, with the latest data", async () => {
  const save = vi.fn(async (d: Doc) => `v${d.n}`);
  render(<Harness save={save} />);
  fireEvent.click(screen.getByText("edit"));
  fireEvent.click(screen.getByText("edit"));
  expect(screen.getByRole("status")).toHaveTextContent("dirty");
  await act(async () => { vi.advanceTimersByTime(900); });
  expect(save).toHaveBeenCalledTimes(1);
  expect(save).toHaveBeenCalledWith({ n: 2 });
  expect(screen.getByRole("status")).toHaveTextContent("clean");
});

test("an edit made while a save is in flight is saved afterwards, and the stale reply is marked", async () => {
  let release: (v: string) => void = () => {};
  const save = vi.fn((d: Doc) => (d.n === 1 ? new Promise<string>((r) => { release = r; }) : Promise.resolve(`v${d.n}`)));
  const onSaved = vi.fn();
  render(<Harness save={save} onSaved={onSaved} />);
  fireEvent.click(screen.getByText("edit"));
  await act(async () => { vi.advanceTimersByTime(900); });
  expect(screen.getByRole("status")).toHaveTextContent("saving");
  fireEvent.click(screen.getByText("edit"));  // while n=1 is being saved
  await act(async () => { release("v1"); });
  await act(async () => { vi.advanceTimersByTime(10); });
  expect(save).toHaveBeenLastCalledWith({ n: 2 });
  expect(onSaved).toHaveBeenNthCalledWith(1, "v1", true);   // stale: edits happened meanwhile
  expect(onSaved).toHaveBeenNthCalledWith(2, "v2", false);
  expect(screen.getByRole("status")).toHaveTextContent("clean");
});

test("a save that times out is aborted and retried", async () => {
  let calls = 0;
  const save = vi.fn((d: Doc, signal: AbortSignal) => {
    calls += 1;
    if (calls > 1) return Promise.resolve(`v${d.n}`);
    return new Promise<string>((_, reject) => {
      signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    });
  });
  render(<Harness save={save} />);
  fireEvent.click(screen.getByText("edit"));
  await act(async () => { vi.advanceTimersByTime(900); });
  expect(screen.getByRole("status")).toHaveTextContent("saving");
  await act(async () => { vi.advanceTimersByTime(SAVE_TIMEOUT_MS); });
  expect(save).toHaveBeenCalledTimes(2);
  expect(screen.getByRole("status")).toHaveTextContent("clean");
});

test("a failed save keeps the edits and says why; the next edit tries again", async () => {
  const save = vi.fn().mockRejectedValueOnce(new Error("disk full")).mockResolvedValue("ok");
  render(<Harness save={save} />);
  fireEvent.click(screen.getByText("edit"));
  await act(async () => { vi.advanceTimersByTime(900); });
  expect(screen.getByRole("status")).toHaveTextContent("error:disk full");
  fireEvent.click(screen.getByText("edit"));
  await act(async () => { vi.advanceTimersByTime(900); });
  expect(screen.getByRole("status")).toHaveTextContent("clean");
  expect(save).toHaveBeenCalledTimes(2);
});

test("flush saves at once without waiting for the pause", async () => {
  const save = vi.fn(async (d: Doc) => `v${d.n}`);
  render(<Harness save={save} />);
  fireEvent.click(screen.getByText("edit"));
  await act(async () => { fireEvent.click(screen.getByText("flush")); });
  expect(save).toHaveBeenCalledWith({ n: 1 });
  expect(screen.getByRole("status")).toHaveTextContent("clean");
});
