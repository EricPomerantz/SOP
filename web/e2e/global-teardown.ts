import { handBack, removeClone } from "./lib/cleanup";

/** Leave the repository as it was. The app keeps the published clone loaded until its next restart or publish. */
export default async function teardown() {
  if (process.env.E2E_KEEP === "1") { console.log("E2E_KEEP=1: the clone's files are kept."); handBack(); return; }
  removeClone();
  handBack();
}
