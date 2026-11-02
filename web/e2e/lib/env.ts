/** What the specs work on; docker-compose.yml's e2e service sets these. */
export const env = {
  base: process.env.E2E_BASE_URL ?? "http://sopvision:8000",
  sopId: process.env.E2E_SOP_ID ?? "lug_bolts_ui",
  sourceSop: process.env.E2E_SOURCE_SOP ?? "lug_bolts_torque",
  sourceFile: process.env.E2E_SOURCE_FILE ?? "lug_bolts.yaml",
  slotTolerance: Number(process.env.E2E_SLOT_TOL ?? "0.03"),
  draftRun: process.env.E2E_DRAFT_RUN ?? "correct10",
  repo: process.env.E2E_REPO ?? "/repo",
  hostUser: process.env.E2E_HOST_USER ?? "1000:1000",
};
