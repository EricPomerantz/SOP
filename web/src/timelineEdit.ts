/** Editing a step timeline: contiguous rows [step, start, end] covering the video, as steps.csv stores them. */
export type Row = [string, number, number];

const r2 = (t: number) => Math.round(t * 100) / 100;

/** Adjacent rows with the same step become one; zero-length rows go. */
export function normalize(rows: Row[]): Row[] {
  const out: Row[] = [];
  for (const [s, a, b] of rows) {
    if (b - a <= 0.005) continue;
    const last = out[out.length - 1];
    if (last && last[0] === s) last[2] = b;
    else out.push([s, a, b]);
  }
  if (out.length) out[0][1] = 0;
  for (let i = 1; i < out.length; i++) out[i][1] = out[i - 1][2];
  return out.map(([s, a, b]) => [s, r2(a), r2(b)]);
}

export function rowAt(rows: Row[], t: number): number {
  const i = rows.findIndex(([, a, b]) => t >= a && t < b);
  return i >= 0 ? i : rows.length - 1;
}

/** From `t` on, until the end of the row under it, the work is `step` (the step change happens at t). */
export function splitAt(rows: Row[], t: number, step: string): Row[] {
  if (!rows.length) return rows;
  const i = rowAt(rows, t);
  const [s, a, b] = rows[i];
  const at = Math.min(Math.max(r2(t), a), b);
  const parts: Row[] = at > a ? [[s, a, at], [step, at, b]] : [[step, a, b]];
  return normalize([...rows.slice(0, i), ...parts, ...rows.slice(i + 1)].map((r) => [...r] as Row));
}

/** Move the boundary between row i and row i+1 to t (kept inside both rows). */
export function setBoundary(rows: Row[], i: number, t: number): Row[] {
  if (i < 0 || i >= rows.length - 1) return rows;
  const lo = rows[i][1] + 0.05, hi = rows[i + 1][2] - 0.05;
  const at = r2(Math.min(Math.max(t, lo), hi));
  const out = rows.map((r) => [...r] as Row);
  out[i][2] = at;
  out[i + 1][1] = at;
  return out;
}

export function setStep(rows: Row[], i: number, step: string): Row[] {
  return normalize(rows.map((r, j) => (j === i ? [step, r[1], r[2]] : [...r]) as Row));
}

/** Row i goes; the row before takes its time (the row after, for the first). */
export function removeRow(rows: Row[], i: number): Row[] {
  if (rows.length <= 1) return rows;
  const out = rows.map((r) => [...r] as Row);
  if (i > 0) out[i - 1][2] = out[i][2];
  else out[1][1] = 0;
  out.splice(i, 1);
  return normalize(out);
}

/** What a save will refuse, or null. */
export function problem(rows: Row[], duration: number): string | null {
  if (!rows.length) return "The timeline is empty";
  if (Math.abs(rows[0][1]) > 0.01) return "The first row must start at 0";
  for (let i = 1; i < rows.length; i++) if (Math.abs(rows[i][1] - rows[i - 1][2]) > 0.01) return `Gap before row ${i + 1}`;
  if (duration && Math.abs(rows[rows.length - 1][2] - duration) > 0.5) return `The timeline ends at ${rows[rows.length - 1][2]} s, the video at ${duration} s`;
  return null;
}
