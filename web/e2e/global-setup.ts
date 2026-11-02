import { env } from "./lib/env";
import { removeClone } from "./lib/cleanup";

async function get(path: string): Promise<{ status: number; body: unknown }> {
  const res = await fetch(env.base + path);
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}

/** Refuse to start when the run would disturb something (a live run, a recording, a job) or can't work. */
export default async function setup() {
  try { await get("/builder/sops"); } catch {
    throw new Error(`The app isn't reachable at ${env.base}: start it with ./start.sh (the worker too).`);
  }
  const run = await get("/runs/current");
  if (run.status === 200 && run.body) throw new Error("A run is in progress: stop it first (publishing refuses while a run is live).");
  const rec = await get("/recording");
  if (rec.body) throw new Error("A recording is in progress: stop it first.");
  const jobs = (await get("/jobs")).body as { state: string; title: string }[];
  const busy = jobs.filter((j) => j.state === "running" || j.state === "queued");
  if (busy.length) throw new Error(`The worker is busy (${busy.map((j) => j.title).join(", ")}): wait for it or cancel the job.`);
  const models = (await get("/builder/models")).body as { path: string }[];
  if (!models.some((m) => m.path === "models/lug_bolts/best.pt")) throw new Error("models/lug_bolts/best.pt is missing.");
  const removed = removeClone();
  if (removed.length) console.log(`Removed a previous clone's files: ${removed.join(", ")}`);
}
