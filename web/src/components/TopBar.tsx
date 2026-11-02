import type { ReactNode } from "react";
import { clock } from "../format";
import type { RunStatus } from "../state";
import { AppShell } from "../ui/AppShell";
import { StatusPill } from "../ui/kit";

type Props = { title: string; version?: number; source?: string; elapsed: number; status: RunStatus; running: boolean; children?: ReactNode };

/** The dashboard's frame: the SOP's name, the run clock and the run's state at the right. */
export function TopBar({ title, version, source, elapsed, status, running, children }: Props) {
  const live = running || status !== "On track";
  const health = !live ? "idle" : status === "On track" ? "ok" : status === "Deviation" ? "bad" : "warn";
  const subtitle = [version !== undefined ? `v${version}` : "", source ?? ""].filter(Boolean).join(" · ");
  return (
    <AppShell title={title} subtitle={subtitle || undefined} current="dashboard"
      status={<><span className="timer" aria-label="Run time">{clock(elapsed)}</span><StatusPill health={health} label={live ? status : "No run"} /></>}>
      {children}
    </AppShell>
  );
}
