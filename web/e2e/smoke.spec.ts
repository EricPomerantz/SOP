import { expect, test } from "@playwright/test";

/** Every page loads on the suite's theme (quick: `docker compose --profile e2e run --rm e2e smoke.spec.ts`). */
for (const [hash, heading] of [["#/", null], ["#/sops", "SOPs"], ["#/record", "Record a run"], ["#/latency", "Camera latency check"]] as const) {
  test(`${hash} renders on the deviceWISE theme`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`/${hash}`);
    await expect(page.getByRole("navigation", { name: "Pages" })).toBeVisible();
    if (heading) await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    const accent = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--dw-accent").trim());
    expect(accent).toBe("#002f70");
    const font = await page.evaluate(() => getComputedStyle(document.body).fontFamily);
    expect(font).toContain("Inter");
    expect(errors).toEqual([]);
  });
}
