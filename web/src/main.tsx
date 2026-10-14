import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { LatencyPage } from "./components/LatencyPage";
import { RecordPage } from "./components/RecordPage";
import "./styles.css";

/** The dashboard, the camera latency check (#/latency) or recording (#/record). */
function Root() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  if (hash === "#/latency") return <LatencyPage />;
  if (hash === "#/record") return <RecordPage />;
  return <App />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
