import { modelName, stepName } from "../format";
import type { DecisionView } from "../state";

type Props = { decision?: DecisionView; labels: Record<string, string>; jevOffline: boolean; model: string };

/** What the deciding model (Jev, or a local Decider/Laya) currently says. */
export function JevStrip({ decision, labels, jevOffline, model }: Props) {
  const name = modelName(model);
  return (
    <section className="jev card" aria-label={`${name} now`}>
      <span className="muted">{name} now</span>
      <span className="jev-step">{decision ? stepName(decision.currentStep, labels) : "Waiting for the first decision"}</span>
      <div className="bar" aria-hidden="true">
        <div className="bar-fill" style={{ width: `${Math.round((decision?.confidence ?? 0) * 100)}%` }} />
      </div>
      <span className="jev-num">{decision ? decision.confidence.toFixed(2) : "–"}</span>
      <span className="muted">
        {decision ? `off-SOP ${decision.offSop.toFixed(2)} · ${Math.round(decision.latencyMs)} ms` : ""}
      </span>
      {jevOffline && <span className="pill pill-warn">{name} offline</span>}
    </section>
  );
}
