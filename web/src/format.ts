export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function stamp(seconds: number): string {
  return `${clock(seconds)}.${Math.floor((seconds % 1) * 10)}`;
}

export function stepName(id: string, labels: Record<string, string>): string {
  if (id === "idle") return "Idle";
  if (id === "off_sop") return "Off-SOP activity";
  return labels[id] ?? id;
}
