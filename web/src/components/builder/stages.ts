import type { StageId } from "../../builderApi";

export const STAGE_IDS: StageId[] = ["detector", "recordings", "layout", "steps", "timelines", "train"];

/** Each stage's name and, in a sentence, what it is for (the server says what each still needs). */
export const STAGE_INFO: Record<StageId, { title: string; purpose: string }> = {
  detector: { title: "Detector", purpose: "Pick the trained model that sees the parts, and say what each of its classes is." },
  recordings: { title: "Recordings", purpose: "Film the procedure being done, correctly and with mistakes: the examples everything else learns from." },
  layout: { title: "Slots", purpose: "Mark where each part goes on the work piece, so the app knows which slot is being filled." },
  steps: { title: "Steps", purpose: "Write the steps, what completes each one, and the order they must happen in." },
  timelines: { title: "Timelines", purpose: "Mark which step was under way at each moment of every recording: the decision model's answer key." },
  train: { title: "Decision model", purpose: "Train the model that judges live runs, check it on recordings it has never seen, and switch it on." },
};

export const stageHash = (sopId: string, stage: StageId) => `#/sops/${sopId}/${stage}`;
export const nextStage = (stage: StageId): StageId | undefined => STAGE_IDS[STAGE_IDS.indexOf(stage) + 1];
export const prevStage = (stage: StageId): StageId | undefined => STAGE_IDS[STAGE_IDS.indexOf(stage) - 1];
