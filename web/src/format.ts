export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function stamp(seconds: number): string {
  return `${clock(seconds)}.${Math.floor((seconds % 1) * 10)}`;
}

const MODEL_NAMES: Record<string, string> = { jev: "Jev", decider: "Decider", laya: "Laya" };

/** Display name of the model making SOP decisions. */
export function modelName(id: string | undefined): string {
  return MODEL_NAMES[id ?? "jev"] ?? id ?? "Jev";
}

export function stepName(id: string, labels: Record<string, string>): string {
  if (id === "idle") return "Idle";
  if (id === "off_sop") return "Off-SOP activity";
  return labels[id] ?? id;
}
