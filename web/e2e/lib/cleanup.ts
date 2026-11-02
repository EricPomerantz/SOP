import { execSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { join } from "node:path";
import { env } from "./env";

/** Remove what a clone run leaves: its draft, its published file and the builder's job results for it. */
export function removeClone(): string[] {
  const paths = [
    join(env.repo, "sops", "drafts", `${env.sopId}.yaml`),
    join(env.repo, "sops", `${env.sopId}.yaml`),
    join(env.repo, "data", "builder", env.sopId),
  ];
  const removed = paths.filter((p) => existsSync(p));
  for (const p of removed) rmSync(p, { recursive: true, force: true });
  return removed;
}

/** The container runs as root (the app's files are root's); give the results back to the host user. */
export function handBack(): void {
  try { execSync(`chown -R ${env.hostUser} test-results`, { stdio: "ignore" }); } catch { /* not root: nothing to do */ }
}
