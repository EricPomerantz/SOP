import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import type { SopData } from "../../builderApi";
const lastData = (fn: ReturnType<typeof vi.fn>) => fn.mock.calls[fn.mock.calls.length - 1]![0] as SopData;
import { Providers } from "../../ui/Providers";
import { SlotTable } from "./SlotTable";

const DATA: SopData = {
  sop: "wheel", name: "Wheel", version: 1, description: "",
  detector: { weights: "", classes: ["hub", "hole_empty", "red", "blue"], roles: { hole_empty: "empty", red: "part", blue: "part" } },
  layout: { anchor: "hub", match_radius: 0.15, slots: { slot_1: { at: [0.5, 0.2], filled_by: "red", empty_as: "hole_empty" }, slot_2: { at: [0.2, 0.5], filled_by: ["blue"], empty_as: null } } },
  steps: [{ id: "place_1", text: "x", evidence: { slot_1: "installed" } }, { id: "place_2", text: "y", evidence: { slot_2: "installed" } }],
  order: "strict", off_sop: "", idle: "",
};
function Harness({ onData }: { onData: (d: SopData) => void }) {
  const [data, setData] = useState(DATA);
  return <Providers><SlotTable data={data} readOnly={false} onChange={(d) => { setData(d); onData(d); }} /></Providers>;
}

test("'Allow every part in every slot' fills each slot with all the part classes", () => {
  const onData = vi.fn();
  render(<Harness onData={onData} />);
  fireEvent.click(screen.getByRole("button", { name: "Allow every part in every slot" }));
  const d = lastData(onData);
  expect(d.layout.slots.slot_1.filled_by).toEqual(["red", "blue"]);
  expect(d.layout.slots.slot_2.filled_by).toEqual(["red", "blue"]);
});

test("deleting a slot names the steps that use it and takes it out of their evidence", async () => {
  const onData = vi.fn();
  render(<Harness onData={onData} />);
  fireEvent.click(screen.getByRole("button", { name: "Delete slot_1" }));
  const dialog = await screen.findByRole("alertdialog");
  expect(within(dialog).getByText(/place_1/)).toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }));
  await waitFor(() => expect(Object.keys((lastData(onData)).layout.slots)).toEqual(["slot_2"]));
  expect((lastData(onData)).steps[0].evidence).toEqual({});
});

test("a rename to a taken name is refused with a message and the field goes back", () => {
  render(<Harness onData={() => {}} />);
  const field = screen.getByLabelText("Name of slot_1");
  fireEvent.change(field, { target: { value: "slot_2" } });
  fireEvent.blur(field);
  expect(screen.getByText(/slot_2 is already a slot/)).toBeInTheDocument();
  expect(screen.getByLabelText("Name of slot_1")).toHaveValue("slot_1");
});
