import { defineConfig, devices } from "@playwright/test";

/** Drives the real app (the built dashboard served by sopvision) with the worker doing the GPU jobs. */
export default defineConfig({
  testDir: ".",
  testMatch: /.*\.spec\.ts/,
  timeout: 40 * 60_000,
  expect: { timeout: 15_000 },
  workers: 1,
  retries: 0,
  outputDir: "test-results/artifacts",
  reporter: [["list"], ["html", { outputFolder: "test-results/report", open: "never" }]],
  globalSetup: "./global-setup.ts",
  globalTeardown: "./global-teardown.ts",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://sopvision:8000",
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
});
