/// <reference types="vitest/config" />
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://sopvision:8000";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  build: {
    rollupOptions: {
      output: {
        // MUI and Emotion in their own chunk: the app's own code stays small and re-downloads alone after a change.
        manualChunks: (id) => (/node_modules\/(@mui|@emotion)\//.test(id) ? "mui" : undefined),
      },
    },
  },
  server: {
    proxy: {
      "/builder": api,
      "/jobs": api,
      "/sops": api,
      "/sources": api,
      "/runs": api,
      "/config": api,
      "/recording": api,
      "/latency": api,
      "/video.mjpg": api,
      "/ws": { target: api, ws: true },
    },
  },
  // e2e/ holds the Playwright specs: they run in a browser against the live app (the e2e compose service).
  test: { globals: true, environment: "jsdom", setupFiles: ["./src/setupTests.ts"], include: ["src/**/*.test.{ts,tsx}"] },
});
