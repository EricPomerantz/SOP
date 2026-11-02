import { render, screen } from "@testing-library/react";
import { AppShell } from "./AppShell";
import { Providers } from "./Providers";
import { applyThemeVars } from "../theme/applyTheme";

test("the theme variables are painted from the suite's tokens", () => {
  applyThemeVars();
  expect(document.documentElement.style.getPropertyValue("--dw-accent")).toBe("#002f70");
  expect(document.documentElement.style.getPropertyValue("--dw-radius")).toBe("0px");
});

test("the shell shows the title, the navigation and the right-hand status", () => {
  render(<Providers><AppShell title="Wheel lug bolts" subtitle="v2" status={<span role="status">No run</span>} current="dashboard">
    <p>body</p></AppShell></Providers>);
  expect(screen.getByRole("heading", { name: "Wheel lug bolts" })).toBeInTheDocument();
  expect(screen.getByText("v2")).toBeInTheDocument();
  const nav = screen.getByRole("navigation", { name: "Pages" });
  expect(nav).toHaveTextContent("Dashboard");
  expect(screen.getByRole("link", { name: "SOPs" })).toHaveAttribute("href", "#/sops");
  expect(screen.getByRole("link", { name: "Record" })).toHaveAttribute("href", "#/record");
  expect(screen.getByRole("link", { name: "Camera latency" })).toHaveAttribute("href", "#/latency");
  expect(screen.getByRole("link", { name: "Dashboard" })).toHaveAttribute("aria-current", "page");
  expect(screen.getByRole("status")).toHaveTextContent("No run");
  expect(screen.getByText("body")).toBeInTheDocument();
});

test("a back link replaces the navigation when a page belongs to another page", () => {
  render(<Providers><AppShell title="Label run r1" back={{ href: "#/sops/x/timelines", label: "Back to the SOP" }} /></Providers>);
  expect(screen.getByRole("link", { name: "Back to the SOP" })).toHaveAttribute("href", "#/sops/x/timelines");
  expect(screen.queryByRole("navigation", { name: "Pages" })).toBeNull();
});
