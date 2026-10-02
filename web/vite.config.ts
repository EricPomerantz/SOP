/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/sops": api,
      "/sources": api,
      "/runs": api,
      "/video.mjpg": api,
      "/ws": { target: api, ws: true },
    },
  },
  test: { globals: true, environment: "jsdom", setupFiles: ["./src/setupTests.ts"] },
});
