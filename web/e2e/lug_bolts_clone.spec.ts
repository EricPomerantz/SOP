/**
 * The lug-bolt SOP rebuilt from nothing through the dashboard only, as a user would: a new SOP that reuses the
 * lug-bolt recordings, the existing detector, slots suggested from the recordings and named, generated steps put in
 * the star order, a timeline drafted and checked, then Review and publish. The published file is compared with the
 * hand-written sops/lug_bolts.yaml. Training the decision model (an hour on the GPU) is left out.
 */
import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";
import { env } from "./lib/env";

type Slot = { at: [number, number]; filled_by: string | string[]; empty_as?: string | null; match_radius?: number | null };
type Step = { id: string; label?: string; text: string; evidence?: Record<string, string> };
type Sop = {
  sop: string; name: string; description: string; detector: { weights: string; classes: string[] };
  layout: { anchor: string; match_radius: number; tools?: string[]; anchor_follow?: number; removal_seconds?: number; slots: Record<string, Slot> };
  steps: Step[]; order: string; groups: (string[] | { sequence: string[] })[]; off_sop: string; idle: string;
  decisions: Record<string, number>; dataset: { format: string; dir: string };
};

const ORIGINAL: Sop = parse(readFileSync(join(env.repo, "sops", env.sourceFile), "utf-8"));
const SHOTS = "test-results/walkthrough";
mkdirSync(SHOTS, { recursive: true });
let shot = 0;
const snap = (page: Page, name: string) => page.screenshot({ path: join(SHOTS, `${String(++shot).padStart(2, "0")}-${name}.png`), fullPage: true });
const saved = (page: Page) => expect(page.getByText("All changes saved")).toBeVisible({ timeout: 30_000 });
const fills = (s: Slot) => [...(Array.isArray(s.filled_by) ? s.filled_by : [s.filled_by])].sort();
const ROLE: Record<string, string> = { hub: "anchor", hole_empty: "empty", torque_wrench: "tool", hand: "hand" };

/** The pairing of suggested slots to the original's holes with the smallest total distance (5! = 120 to try). */
function pair(found: Record<string, [number, number]>, wanted: Record<string, [number, number]>): Record<string, string> {
  const f = Object.keys(found);
  const w = Object.keys(wanted);
  const d = (a: string, b: string) => Math.hypot(found[a][0] - wanted[b][0], found[a][1] - wanted[b][1]);
  let best: string[] = [];
  let bestCost = Infinity;
  const permute = (rest: string[], chosen: string[]) => {
    if (!rest.length) {
      const cost = chosen.reduce((sum, hole, i) => sum + d(f[i], hole), 0);
      if (cost < bestCost) { bestCost = cost; best = chosen; }
      return;
    }
    rest.forEach((h, i) => permute([...rest.slice(0, i), ...rest.slice(i + 1)], [...chosen, h]));
  };
  permute(w, []);
  return Object.fromEntries(f.map((name, i) => [name, best[i]]));
}

test("the lug-bolt SOP can be rebuilt through the dashboard alone", async ({ page }) => {
  // 1 A new SOP that reuses the lug-bolt recordings.
  await page.goto("/#/sops");
  await expect(page.getByRole("heading", { name: "SOPs" })).toBeVisible();
  await snap(page, "sop-list");
  await page.getByRole("button", { name: "New SOP" }).click();
  const create = page.getByRole("dialog", { name: "New SOP" });
  await create.getByLabel(/^Name/).fill(ORIGINAL.name);
  await create.getByLabel("Description").fill(ORIGINAL.description);
  await create.getByLabel("Reuse recordings from").selectOption(env.sourceSop);
  await create.getByLabel("Id", { exact: true }).fill(env.sopId);
  await snap(page, "new-sop");
  await create.getByRole("button", { name: "Create" }).click();
  await expect(page).toHaveURL(new RegExp(`#/sops/${env.sopId}/detector$`));

  // 2 Detector: the existing model, and a role per class.
  await page.getByRole("radio", { name: /lug_bolts \/ best\.pt/ }).check();
  await page.getByRole("button", { name: "Use this model" }).click();
  await expect(page.getByLabel("Role of hub")).toBeVisible();
  for (const c of ORIGINAL.detector.classes) {
    if (c === "hand") continue;
    await page.getByLabel(`Role of ${c}`).selectOption(ROLE[c] ?? "part");
  }
  await saved(page);
  await expect(page.getByText("This stage has everything it needs.")).toBeVisible();
  await snap(page, "detector");

  // 3 Recordings: the shared folder's runs.
  await page.getByRole("button", { name: /^Next: Recordings/ }).click();
  await expect(page.getByText(/Shared with Wheel lug bolts/)).toBeVisible();
  const rows = page.getByRole("table", { name: "Recordings" }).getByRole("row");
  await expect(rows).toHaveCount(26);  // header + 25 runs
  await snap(page, "recordings");

  // 4 Slots: suggested from the recordings, named as the procedure numbers them.
  await page.getByRole("button", { name: /^Next: Slots/ }).click();
  await page.getByRole("button", { name: "Suggest slots" }).click();
  await expect(page.getByText(/^Finished in/)).toBeVisible({ timeout: 20 * 60_000 });
  const use = page.getByRole("button", { name: /^Use the \d+ suggested slots$/ });
  await expect(use).toHaveText(`Use the ${Object.keys(ORIGINAL.layout.slots).length} suggested slots`);
  await use.click();
  const found: Record<string, [number, number]> = {};
  const radius: Record<string, string> = {};
  for (let n = 1; n <= Object.keys(ORIGINAL.layout.slots).length; n++) {
    const name = `slot_${n}`;
    found[name] = [Number(await page.getByLabel(`x of ${name}`).inputValue()), Number(await page.getByLabel(`y of ${name}`).inputValue())];
    radius[name] = await page.getByLabel(`Radius of ${name}`).inputValue();
  }
  const wanted = Object.fromEntries(Object.entries(ORIGINAL.layout.slots).map(([k, s]) => [k, s.at]));
  const names = pair(found, wanted);
  await snap(page, "slots-suggested");
  for (const [from, to] of Object.entries(names)) {
    const field = page.getByLabel(`Name of ${from}`);
    await field.fill(to);
    await field.press("Enter");
    await expect(page.getByLabel(`Name of ${to}`)).toBeVisible();
  }
  await page.getByRole("button", { name: "Allow every part in every slot" }).click();
  for (const [hole, s] of Object.entries(ORIGINAL.layout.slots)) {
    await page.getByLabel(`Empty marker of ${hole}`).selectOption(s.empty_as ?? "");
    await page.getByLabel(`Radius of ${hole}`).fill("");
  }
  await page.locator("summary", { hasText: "Advanced" }).last().click();
  await page.getByLabel("Default radius").fill(String(ORIGINAL.layout.match_radius));
  await page.getByLabel("Anchor follow").fill(String(ORIGINAL.layout.anchor_follow));
  await page.getByLabel("Removal wait (s)").fill(String(ORIGINAL.layout.removal_seconds));
  await saved(page);
  await snap(page, "slots");
  const deltas = Object.fromEntries(Object.entries(names).map(([from, hole]) => [hole, {
    suggested_as: from, at: found[from], original: wanted[hole], radius_suggested: radius[from] || "default",
    dx: +(found[from][0] - wanted[hole][0]).toFixed(3), dy: +(found[from][1] - wanted[hole][1]).toFixed(3),
  }]));
  writeFileSync("test-results/slot-deltas.json", JSON.stringify(deltas, null, 2));

  // 5 Steps: generated, worded as the original, torques in the star order.
  await page.getByRole("button", { name: /^Next: Steps/ }).click();
  await page.getByRole("button", { name: "Generate from slots" }).click();
  await expect(page.getByLabel("Label of place_1")).toBeVisible();
  for (const s of ORIGINAL.steps) {
    await page.getByLabel(`Label of ${s.id}`).fill(s.label ?? "");
    await page.getByLabel(`Instruction of ${s.id}`).fill(s.text);
  }
  const star = (ORIGINAL.groups[1] as { sequence: string[] }).sequence;
  for (const [i, id] of star.entries()) await page.getByLabel(`Position of ${id}`).selectOption(String(i + 1));
  await page.locator("summary", { hasText: "Advanced" }).click();
  await page.getByLabel("Off-SOP means").fill(ORIGINAL.off_sop);
  await page.getByLabel("Idle means").fill(ORIGINAL.idle);
  await saved(page);
  await snap(page, "steps");

  // 6 Timelines: the 25 runs are labelled; draft one with the tracker and look at it, without saving over its labels.
  await page.getByRole("button", { name: /^Next: Timelines/ }).click();
  await expect(page.getByText("25 of 25 runs have a saved timeline")).toBeVisible();
  await page.getByRole("button", { name: `Draft ${env.draftRun}` }).click();
  await expect(page.getByText(/^Finished in/)).toBeVisible({ timeout: 15 * 60_000 });
  await snap(page, "timelines");
  await page.getByRole("link", { name: `Open ${env.draftRun}` }).click();
  await expect(page.getByRole("button", { name: "Start from the tracker's draft" })).toBeVisible();
  await expect(page.getByRole("status").filter({ hasText: "Saved" })).toBeVisible();
  await snap(page, "timeline-page");
  await page.getByRole("link", { name: "Back to the timelines" }).click();

  // 7 Review and publish.
  await page.getByRole("button", { name: "Review and publish" }).click();
  const review = page.getByRole("dialog", { name: `Publish ${ORIGINAL.name}?` });
  await expect(review.getByText("Before it can be published")).toHaveCount(0);
  await snap(page, "publish");
  await review.getByRole("button", { name: "Publish" }).click();
  await expect(page.getByText(/is published and in the dashboard's SOP list/)).toBeVisible({ timeout: 30_000 });

  // The published clone matches the hand-written SOP.
  const clone: Sop = parse(readFileSync(join(env.repo, "sops", `${env.sopId}.yaml`), "utf-8"));
  expect(clone.sop).toBe(env.sopId);
  expect([clone.name, clone.description]).toEqual([ORIGINAL.name, ORIGINAL.description]);
  expect(clone.detector.weights).toBe(ORIGINAL.detector.weights);
  expect(clone.detector.classes).toEqual(ORIGINAL.detector.classes);
  const { slots, ...layout } = clone.layout;
  const { slots: originalSlots, ...originalLayout } = ORIGINAL.layout;
  expect(layout).toEqual(originalLayout);
  expect(Object.keys(slots).sort()).toEqual(Object.keys(originalSlots).sort());
  for (const [hole, s] of Object.entries(originalSlots)) {
    expect.soft(Math.abs(slots[hole].at[0] - s.at[0]), `${hole} across`).toBeLessThanOrEqual(env.slotTolerance);
    expect.soft(Math.abs(slots[hole].at[1] - s.at[1]), `${hole} down`).toBeLessThanOrEqual(env.slotTolerance);
    expect(fills(slots[hole])).toEqual(fills(s));
    expect(slots[hole].empty_as ?? null).toBe(s.empty_as ?? null);
    expect(slots[hole].match_radius ?? null).toBeNull();
  }
  const byId = (steps: Step[]) => Object.fromEntries(steps.map((s) => [s.id, { label: s.label, text: s.text, evidence: s.evidence }]));
  expect(byId(clone.steps)).toEqual(byId(ORIGINAL.steps));
  expect(clone.order).toBe(ORIGINAL.order);
  expect([...(clone.groups[0] as string[])].sort()).toEqual([...(ORIGINAL.groups[0] as string[])].sort());
  expect(clone.groups[1]).toEqual(ORIGINAL.groups[1]);
  expect([clone.off_sop, clone.idle]).toEqual([ORIGINAL.off_sop, ORIGINAL.idle]);
  expect(clone.decisions).toEqual(ORIGINAL.decisions);
  expect(clone.dataset).toEqual(ORIGINAL.dataset);
});
