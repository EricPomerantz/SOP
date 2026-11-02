import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { LatencyPage } from "./components/LatencyPage";
import { RecordPage } from "./components/RecordPage";
import { SopBuilder } from "./components/builder/SopBuilder";
import { SopsPage } from "./components/builder/SopsPage";
import { TimelinePage } from "./components/builder/TimelinePage";
import type { StageId } from "./builderApi";
import { applyThemeVars } from "./theme/applyTheme";
import { Providers } from "./ui/Providers";
import "@fontsource-variable/inter";
import "./base.css";
import "./theme/dw-alert.css";
import "./styles.css";

applyThemeVars();

/** The dashboard, the camera latency check (#/latency), recording (#/record?sop=<file>) or the SOP builder (#/sops). */
function Root() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  if (hash === "#/latency") return <LatencyPage />;
  const [route, query] = hash.split("?");
  if (route === "#/record") {
    const q = new URLSearchParams(query);
    return <RecordPage initialSop={q.get("sop") ?? undefined} back={q.get("back") ?? undefined} />;
  }
  if (route === "#/sops") return <SopsPage />;
  const timeline = route.match(/^#\/sops\/([a-z0-9_]+)\/runs\/([A-Za-z0-9_.%-]+)$/);
  if (timeline) return <TimelinePage key={route} sopId={timeline[1]} run={decodeURIComponent(timeline[2])} />;
  const sop = route.match(/^#\/sops\/([a-z0-9_]+)(?:\/(detector|recordings|layout|steps|timelines|train))?$/);
  if (sop) return <SopBuilder key={sop[1]} sopId={sop[1]} stage={sop[2] as StageId | undefined} />;
  return <App />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Providers>
      <Root />
    </Providers>
  </StrictMode>,
);
