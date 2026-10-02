import type { DecisionView, StepView } from "../state";

type Props = { steps: StepView[]; decision?: DecisionView };

const ICONS = { done: "✓", active: "▶", pending: "○" } as const;

export function Checklist({ steps, decision }: Props) {
  return (
    <section className="checklist card">
      <h2>Checklist</h2>
      <ol aria-label="Checklist">
        {steps.map((s) => {
          const confidence = s.status === "active" && decision?.currentStep === s.id ? decision.confidence.toFixed(2) : "";
          return (
            <li key={s.id} data-status={s.status} className={`step step-${s.status}`}>
              <span className="step-icon" aria-hidden="true">{ICONS[s.status]}</span>
              <span className="step-label">{s.label}</span>
              <span className="step-meta">
                {s.status === "done" ? (s.duration != null ? `${s.duration.toFixed(1)} s` : "done") : confidence}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
