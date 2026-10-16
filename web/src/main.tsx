import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { LatencyPage } from "./components/LatencyPage";
import { RecordPage } from "./components/RecordPage";
import { SopBuilder } from "./components/builder/SopBuilder";
import { SopsPage } from "./components/builder/SopsPage";
import { TimelinePage } from "./components/builder/TimelinePage";
import "./styles.css";

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
  if (route === "#/record") return <RecordPage initialSop={new URLSearchParams(query).get("sop") ?? undefined} />;
  if (route === "#/sops") return <SopsPage />;
  const timeline = route.match(/^#\/sops\/([a-z0-9_]+)\/runs\/([A-Za-z0-9_.%-]+)$/);
  if (timeline) return <TimelinePage key={route} sopId={timeline[1]} run={decodeURIComponent(timeline[2])} />;
  const sop = route.match(/^#\/sops\/([a-z0-9_]+)$/);
  if (sop) return <SopBuilder key={sop[1]} sopId={sop[1]} />;
  return <App />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
