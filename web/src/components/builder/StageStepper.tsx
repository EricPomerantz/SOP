import { Box, Step, StepButton, Stepper, Tooltip } from "@mui/material";
import type { StageId, StageStatus } from "../../builderApi";
import { HealthDot, plantColor } from "../../ui/kit";
import { STAGE_IDS, STAGE_INFO } from "./stages";

type Props = { stages: StageStatus[]; current: StageId; onGo: (stage: StageId) => void };

/** The six stages in a row: a green dot when a stage has everything, the current one in the accent; click to jump. */
export function StageStepper({ stages, current, onGo }: Props) {
  const byId = Object.fromEntries(stages.map((s) => [s.id, s]));
  return (
    <Box component="nav" aria-label="Stages" sx={{ bgcolor: plantColor.surface, border: `1px solid ${plantColor.line}`, px: 1, py: 1.25 }}>
      <Stepper nonLinear activeStep={STAGE_IDS.indexOf(current)} alternativeLabel>
        {STAGE_IDS.map((id) => {
          const st = byId[id];
          const done = !!st?.done;
          return (
            <Step key={id} completed={done} active={id === current}>
              <Tooltip title={st && !done ? st.blockers.join(" · ") : done ? "Done" : ""} placement="bottom" describeChild>
                <StepButton onClick={() => onGo(id)} aria-current={id === current ? "step" : undefined}
                  icon={<HealthDot health={done ? "ok" : id === current ? "pending" : "idle"} size={12} pulse={false} />}
                  sx={{ py: 0.5, "& .MuiStepLabel-iconContainer": { pr: 0 }, "& .MuiStepLabel-label": { mt: 0.75, fontSize: 12.5, fontWeight: id === current ? 700 : 600 } }}>
                  {STAGE_INFO[id].title}
                </StepButton>
              </Tooltip>
            </Step>
          );
        })}
      </Stepper>
    </Box>
  );
}
