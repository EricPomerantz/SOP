import type { ReactNode } from "react";
import { Box, Button, Typography } from "@mui/material";
import { ArrowBack as BackIcon, ArrowForward as NextIcon, CheckCircleOutline as DoneIcon } from "@mui/icons-material";
import type { StageId, StageStatus } from "../../builderApi";
import { FormBanner } from "../../ui/dialogKit";
import { plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";
import { nextStage, prevStage, STAGE_INFO } from "./stages";

type Props = { stage: StageId; status?: StageStatus; onGo: (stage: StageId) => void; children: ReactNode };

/** One stage's page: what it is for, what it still needs, the stage itself, and Back / Next. */
export function StageFrame({ stage, status, onGo, children }: Props) {
  const info = STAGE_INFO[stage];
  const prev = prevStage(stage);
  const next = nextStage(stage);
  return (
    <Box component="section" aria-labelledby={`stage-${stage}`} sx={{ bgcolor: plantColor.surface, border: `1px solid ${plantColor.line}`, display: "flex", flexDirection: "column" }}>
      <Box sx={{ px: 2.5, pt: 2, pb: 1.5, borderBottom: `1px solid ${plantColor.line}` }}>
        <Typography id={`stage-${stage}`} component="h2" sx={{ fontSize: 18, fontWeight: 700, color: plantColor.ink, letterSpacing: "-0.01em" }}>{info.title}</Typography>
        <Typography sx={{ fontSize: 13, color: plantColor.muted, mt: 0.25 }}>{info.purpose}</Typography>
      </Box>
      <Box sx={{ px: 2.5, pt: 2, pb: 2.5, display: "flex", flexDirection: "column", gap: 2 }}>
        {status && (status.done ? (
          <FormBanner tone="ok"><Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}><DoneIcon sx={{ fontSize: 18, color: "success.main" }} /><span>This stage has everything it needs.{next && <> Next: <strong>{STAGE_INFO[next].title}</strong>.</>}</span></Box></FormBanner>
        ) : (
          <FormBanner tone="warn">
            <strong>What to do next</strong>
            <Box component="ul" sx={{ m: 0, mt: 0.5, pl: 2.25 }}>{status.blockers.map((b) => <li key={b}>{b}</li>)}</Box>
          </FormBanner>
        ))}
        {children}
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 2.5, py: 1.5, borderTop: `1px solid ${plantColor.line}`, bgcolor: plantColor.header }}>
        {prev && <Button variant="outlined" sx={plantBtn} startIcon={<BackIcon />} onClick={() => onGo(prev)}>Back: {STAGE_INFO[prev].title}</Button>}
        <Box sx={{ flex: 1, fontSize: 12.5, color: plantColor.muted, textAlign: "right" }}>
          {status && !status.done && next && "You can go on and come back to this stage later."}
        </Box>
        {next && <Button variant="contained" disableElevation sx={plantBtn} endIcon={<NextIcon />} onClick={() => onGo(next)}>Next: {STAGE_INFO[next].title}</Button>}
      </Box>
    </Box>
  );
}
