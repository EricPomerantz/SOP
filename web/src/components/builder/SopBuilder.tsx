import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Box, Button, IconButton, Tooltip, Typography } from "@mui/material";
import { Edit as EditIcon, Publish as PublishIcon, Settings as SettingsIcon } from "@mui/icons-material";
import { builderApi, type BuilderApi, type SopData, type SopView, type StageId } from "../../builderApi";
import { AppShell } from "../../ui/AppShell";
import { ConfirmDialog } from "../../ui/ConfirmDialog";
import { useGlobalAlert } from "../../ui/GlobalAlertProvider";
import { EmptyNote, HealthLabel, StatusPill, plantColor } from "../../ui/kit";
import { plantBtn } from "../../ui/plantTheme";
import { DetectorStage } from "./DetectorStage";
import { LayoutStage } from "./LayoutStage";
import { PublishDialog } from "./PublishDialog";
import { RecordingsStage } from "./RecordingsStage";
import { SopSettingsDialog } from "./SopSettingsDialog";
import { StageFrame } from "./StageFrame";
import { StageStepper } from "./StageStepper";
import { StepsStage } from "./StepsStage";
import { TimelinesStage } from "./TimelinesStage";
import { TrainStage } from "./TrainStage";
import { stageHash } from "./stages";
import { useAutoSave } from "./useAutoSave";

type Props = { sopId: string; stage?: StageId; api?: BuilderApi; go?: (hash: string) => void };

/**
 * One SOP from detector to decision model (#/sops/<id>/<stage>): a draft edited stage by stage, saved as it is
 * edited, and published into the app's SOPs after a review.
 */
export function SopBuilder({ sopId, stage: stageProp, api = builderApi, go = (h) => { window.location.hash = h; } }: Props) {
  const alert = useGlobalAlert();
  const [view, setView] = useState<SopView | null>(null);
  const [data, setData] = useState<SopData | null>(null);
  const [stage, setStage] = useState<StageId>(stageProp ?? "detector");
  const [error, setError] = useState("");
  const [dialog, setDialog] = useState<"publish" | "settings" | "discard" | null>(null);

  useEffect(() => { if (stageProp) setStage(stageProp); }, [stageProp]);

  const save = useCallback((d: SopData, signal: AbortSignal) => api.save(sopId, d, signal), [api, sopId]);
  const onSaved = useCallback((v: SopView) => setView(v), []);
  const auto = useAutoSave({ data: data as SopData, enabled: !!data && !!view?.draft, save, onSaved });
  const { markSaved, flush } = auto;

  /** The server's view after an action that replaced the draft: what's shown and what counts as saved. */
  const show = useCallback((v: SopView) => { setView(v); setData(v.data); markSaved(v.data); }, [markSaved]);
  const reload = useCallback(() => api.get(sopId).then((v) => { setView(v); setData((d) => d ?? v.data); if (!v.draft) markSaved(v.data); }).catch((e) => setError(e.message)), [api, sopId, markSaved]);
  useEffect(() => { api.get(sopId).then(show).catch((e) => setError(e.message)); }, [api, sopId, show]);

  const act = async (action: () => Promise<void>) => {
    setError("");
    try { await action(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const goStage = (s: StageId) => { setStage(s); go(stageHash(sopId, s)); };

  if (!view || !data) {
    return (
      <AppShell title="SOP builder" current="sops">
        {error ? <Typography role="alert" color="error">{error}</Typography> : <EmptyNote title="Loading…" />}
      </AppShell>
    );
  }
  const s = view.status;
  const change = (d: SopData) => setData(d);
  const publish = async () => {
    await flush();
    const v = await api.publish(sopId);
    show(v);
    setDialog(null);
    alert.success(`${v.data.name || v.id} is published and in the dashboard's SOP list.`, { actionLabel: "Open the dashboard", onAction: () => go("#/") });
  };
  const discard = async () => {
    const r = await api.discard(sopId);
    if ("deleted" in r) { go("#/sops"); } else { show(r); setDialog(null); }
  };
  const edit = () => act(async () => show(await api.edit(sopId)));

  const stageStatus = s.stages?.find((st) => st.id === stage);
  const stages: Record<StageId, ReactNode> = {
    detector: <DetectorStage api={api} view={view} data={data} onChange={change} onView={show} flush={flush} />,
    recordings: <RecordingsStage api={api} view={view} />,
    layout: <LayoutStage api={api} view={view} data={data} onChange={change} flush={flush} />,
    steps: <StepsStage api={api} view={view} data={data} onChange={change} flush={flush} />,
    timelines: <TimelinesStage api={api} view={view} onReload={reload} flush={flush} />,
    train: <TrainStage api={api} view={view} onReload={reload} flush={flush} />,
  };
  const saveLabel = auto.state === "clean" ? "All changes saved" : auto.state === "error" ? `Couldn't save: ${auto.error}` : "Saving…";

  return (
    <AppShell title={data.name || view.id} subtitle={`${view.id} · v${data.version}`} current="sops"
      status={
        <>
          {view.draft && <HealthLabel health={auto.state === "error" ? "bad" : auto.state === "clean" ? "ok" : "pending"} label={saveLabel} strong={false} />}
          <StatusPill health={view.draft ? "idle" : s.loaded ? "ok" : "warn"} label={view.draft ? (s.published ? "Draft of a published SOP" : "Draft") : s.loaded ? "Published · live" : "Published"} />
          {view.draft ? (
            <>
              <Tooltip title="Name, description, recordings folder"><IconButton aria-label="Settings" size="small" onClick={() => setDialog("settings")}><SettingsIcon fontSize="small" /></IconButton></Tooltip>
              <Button variant="contained" disableElevation sx={plantBtn} startIcon={<PublishIcon />} onClick={() => setDialog("publish")}>Review and publish</Button>
              <Button variant="text" color="error" sx={plantBtn} onClick={() => setDialog("discard")}>{s.published ? "Discard changes" : "Delete draft"}</Button>
            </>
          ) : (
            <Button variant="contained" disableElevation sx={plantBtn} startIcon={<EditIcon />} onClick={edit}>Edit</Button>
          )}
        </>
      }>
      {error && <Typography role="alert" sx={{ fontSize: 13, color: plantColor.ink, bgcolor: plantColor.surface, border: `1px solid ${plantColor.line}`, borderLeft: "3px solid", borderLeftColor: "error.main", px: 1.5, py: 1 }}>{error}</Typography>}
      {!view.draft && (
        <Typography sx={{ fontSize: 13, color: plantColor.muted }}>
          This SOP is published. Its stages are shown as they are; press Edit to change them in a draft copy (runs keep using the published version until the draft is published).
        </Typography>
      )}
      <StageStepper stages={s.stages ?? []} current={stage} onGo={goStage} />
      <StageFrame stage={stage} status={stageStatus} onGo={goStage}>
        <Box className="stage-body">{stages[stage]}</Box>
      </StageFrame>
      <PublishDialog open={dialog === "publish"} view={view} data={data} onClose={() => setDialog(null)} onPublish={publish} />
      <SopSettingsDialog open={dialog === "settings"} api={api} data={data} onClose={() => setDialog(null)} onChange={change} />
      <ConfirmDialog open={dialog === "discard"} onClose={() => setDialog(null)} onConfirm={discard}
        title={s.published ? "Throw away the changes in this draft?" : "Delete this draft SOP?"}
        body={s.published ? "The published SOP stays as it is; the edits made since are lost." : "The draft, its roles, slots and steps are deleted. Recordings and timelines on disk are kept."}
        confirmText={view.id} actionLabel={s.published ? "Discard changes" : "Delete"} />
    </AppShell>
  );
}
