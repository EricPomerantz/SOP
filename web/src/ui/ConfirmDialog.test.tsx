import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ConfirmDialog } from "./ConfirmDialog";
import { Providers } from "./Providers";

test("a plain confirm asks a question and runs the action on yes", async () => {
  const onConfirm = vi.fn(async () => {});
  const onClose = vi.fn();
  render(<Providers><ConfirmDialog open title="Delete this step?" body="It can't be undone." actionLabel="Delete" onClose={onClose} onConfirm={onConfirm} /></Providers>);
  expect(screen.getByRole("alertdialog", { name: "Delete this step?" })).toBeInTheDocument();
  const button = screen.getByRole("button", { name: "Delete" });
  expect(button).toBeEnabled();
  fireEvent.click(button);
  await waitFor(() => expect(onConfirm).toHaveBeenCalled());
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(onClose).toHaveBeenCalled();
});

test("a typed confirm stays disabled until the text matches", async () => {
  const onConfirm = vi.fn(async () => {});
  render(<Providers><ConfirmDialog open title="Delete the draft?" body="Gone." confirmText="wheel" onClose={() => {}} onConfirm={onConfirm} /></Providers>);
  const button = screen.getByRole("button", { name: "Delete" });
  expect(button).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Type wheel to confirm"), { target: { value: "wheel" } });
  expect(button).toBeEnabled();
  fireEvent.click(button);
  await waitFor(() => expect(onConfirm).toHaveBeenCalled());
});

test("a failing action shows its message and keeps the dialog open", async () => {
  render(<Providers><ConfirmDialog open title="Delete?" body="x" onClose={() => {}} onConfirm={async () => { throw new Error("Nope: a run is live"); }} /></Providers>);
  fireEvent.click(screen.getByRole("button", { name: "Delete" }));
  expect(await screen.findByText("Nope: a run is live")).toBeInTheDocument();
  expect(screen.getByRole("alertdialog")).toBeInTheDocument();
});
