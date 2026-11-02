import { useEffect, useState, type ReactNode } from "react";
import { Box, Button, CircularProgress, TextField, Typography } from "@mui/material";
import { Warning as WarningIcon } from "@mui/icons-material";
import { dialogBtn, fieldSx } from "./dialogStyles";
import { FooterNote, FormDialog, FormField } from "./dialogKit";
import { plantColor } from "./plantTheme";

/**
 * A question before something that can't be undone (the suite's ConfirmDialog, with the typed confirmation made
 * optional: a slot or a step is deleted with one click, a whole draft only after typing its id).
 */
export function ConfirmDialog({
  open, title, body, confirmText, actionLabel = "Delete", tone = "danger", onClose, onConfirm,
}: {
  open: boolean;
  title: string;
  body: ReactNode;
  /** What to type to enable the button; omitted, the button is enabled at once. */
  confirmText?: string;
  actionLabel?: string;
  tone?: "danger" | "accent";
  onClose: () => void;
  onConfirm: () => Promise<void> | void;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    if (open) { setTyped(""); setProblem(null); setBusy(false); }
  }, [open]);
  const ready = !confirmText || typed.trim() === confirmText.trim();
  const run = async () => {
    if (!ready || busy) return;
    setBusy(true);
    setProblem(null);
    try {
      await onConfirm();
    } catch (err) {
      setProblem((err as Error).message || "Request failed");
    } finally {
      setBusy(false);
    }
  };
  return (
    <FormDialog
      open={open}
      onClose={onClose}
      busy={busy}
      tone={tone}
      role="alertdialog"
      icon={<WarningIcon />}
      title={title}
      maxWidth="xs"
      status={problem ? <FooterNote tone="bad">{problem}</FooterNote> : null}
      actions={
        <>
          <Button sx={{ ...dialogBtn, color: plantColor.muted }} onClick={onClose} disabled={busy}>Cancel</Button>
          <Button
            color={tone === "danger" ? "error" : "primary"}
            variant="contained"
            disableElevation
            sx={dialogBtn}
            onClick={() => void run()}
            disabled={!ready || busy}
            startIcon={busy ? <CircularProgress size={14} color="inherit" /> : undefined}
          >
            {actionLabel}
          </Button>
        </>
      }
    >
      <Typography component="div" sx={{ fontSize: 13.5, color: plantColor.ink, lineHeight: 1.55 }}>{body}</Typography>
      {confirmText && (
        <Box sx={{ mt: 2 }}>
          <FormField label={<>Type <Box component="span" sx={{ fontFamily: "ui-monospace, Menlo, monospace", color: plantColor.ink }}>{confirmText}</Box> to confirm</>} htmlFor="confirm-typed">
            <TextField
              id="confirm-typed"
              autoFocus
              fullWidth
              size="small"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void run()}
              slotProps={{ htmlInput: { "aria-label": `Type ${confirmText} to confirm`, spellCheck: false } }}
              sx={fieldSx}
            />
          </FormField>
        </Box>
      )}
    </FormDialog>
  );
}
