/**
 * The look of the Factory's form dialogs: a header that says what is being
 * edited (icon, name, a few facts as chips), sections of fields laid out on a
 * two-column grid with their labels above them, and a footer with the form's
 * state on the left and its buttons on the right. Square, as the suite's
 * theme is.
 */
import { useId, useRef, type ReactNode } from 'react';
import { Box, Dialog, IconButton, Tooltip, Typography, type DialogProps } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { Close as CloseIcon } from '@mui/icons-material';

import tokens from '@/theme/tokens.json';
import { HEALTH_COLOR, HEALTH_TEXT, plantColor } from './plantTheme';

/** A labelled field of the grid. `wide` spans both columns. */
export function FormField({
  label,
  required,
  htmlFor,
  labelId,
  wide,
  aside,
  children,
}: {
  label: ReactNode;
  required?: boolean;
  /** The input the label names. */
  htmlFor?: string;
  labelId?: string;
  wide?: boolean;
  /** Beside the label, at its right (a small action, a count). */
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Box sx={{ gridColumn: wide ? '1 / -1' : undefined, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 0.625 }}>
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, minHeight: 18 }}>
        <Typography component="label" id={labelId} htmlFor={htmlFor} sx={{ fontSize: 12.5, fontWeight: 600, color: tokens.gray[700], lineHeight: 1.4 }}>
          {label}
          {required && (
            <Box component="span" aria-hidden sx={{ color: tokens.color.danger, ml: 0.375 }}>
              *
            </Box>
          )}
        </Typography>
        {aside && <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center' }}>{aside}</Box>}
      </Box>
      {children}
    </Box>
  );
}

/** Fields two to a row (one on narrow screens). */
export function FormGrid({ children }: { children: ReactNode }) {
  return <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' }, columnGap: 2, rowGap: 2 }}>{children}</Box>;
}

/** A titled part of the form; the next one sits below a hairline. */
export function FormSection({ title, action, children }: { title: ReactNode; action?: ReactNode; children: ReactNode }) {
  return (
    <Box component="section" sx={{ '& + &': { borderTop: `1px solid ${tokens.gray[100]}`, pt: 2.5, mt: 2.5 } }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.75, minHeight: 28 }}>
        <Typography component="h3" sx={{ fontSize: 14, fontWeight: 700, color: plantColor.ink, flex: 1 }}>
          {title}
        </Typography>
        {action}
      </Box>
      {children}
    </Box>
  );
}

/** A fact in the header: the type, the core, the state. */
export function MetaChip({ icon, dot, children, title }: { icon?: ReactNode; dot?: keyof typeof HEALTH_COLOR; children: ReactNode; title?: string }) {
  return (
    <Box
      component="span"
      title={title}
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 0.625,
        height: 24,
        px: 1,
        fontSize: 12,
        fontWeight: 500,
        color: dot ? HEALTH_TEXT[dot] : tokens.gray[600],
        bgcolor: dot ? alpha(HEALTH_COLOR[dot], 0.1) : tokens.gray[50],
        border: `1px solid ${dot ? alpha(HEALTH_COLOR[dot], 0.25) : tokens.gray[100]}`,
        whiteSpace: 'nowrap',
        '& svg': { fontSize: 15, color: tokens.gray[500] },
      }}
    >
      {dot && <Box component="span" aria-hidden sx={{ width: 7, height: 7, borderRadius: '50%', bgcolor: HEALTH_COLOR[dot] }} />}
      {icon}
      {children}
    </Box>
  );
}

/** The form's state at the footer's left: valid, what saving does, what to fix. */
export function FooterNote({ tone = 'idle', icon, children }: { tone?: 'ok' | 'warn' | 'bad' | 'idle'; icon?: ReactNode; children: ReactNode }) {
  return (
    <Box
      role={tone === 'bad' ? 'alert' : 'status'}
      sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.75, fontSize: 12.5, fontWeight: 500, color: tone === 'idle' ? plantColor.muted : HEALTH_TEXT[tone], minWidth: 0, '& svg': { fontSize: 17 } }}
    >
      {icon}
      <Box component="span" sx={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {children}
      </Box>
    </Box>
  );
}

/** A problem across the top of the body, with a bar in its color. */
export function FormBanner({ tone, children, onClose }: { tone: 'bad' | 'ok' | 'warn'; children: ReactNode; onClose?: () => void }) {
  return (
    <Box
      role={tone === 'bad' ? 'alert' : 'status'}
      sx={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: 1,
        mb: 2.5,
        px: 1.5,
        py: 1,
        fontSize: 13,
        color: plantColor.ink,
        bgcolor: alpha(HEALTH_COLOR[tone], 0.07),
        borderLeft: `3px solid ${HEALTH_COLOR[tone]}`,
        whiteSpace: 'pre-wrap',
      }}
    >
      <Box sx={{ flex: 1, minWidth: 0, pt: 0.25 }}>{children}</Box>
      {onClose && (
        <IconButton size="small" aria-label="Dismiss" onClick={onClose} sx={{ p: 0.25, color: plantColor.muted }}>
          <CloseIcon sx={{ fontSize: 16 }} />
        </IconButton>
      )}
    </Box>
  );
}

/** The dialog as drawn (see `FormDialog`). */
function FormDialogView({
  open,
  onClose,
  icon,
  title,
  meta,
  status,
  actions,
  children,
  maxWidth = 'md',
  busy,
  minBodyHeight,
  tone = 'accent',
  role,
  drop,
}: {
  open: boolean;
  onClose: () => void;
  icon?: ReactNode;
  title: ReactNode;
  meta?: ReactNode;
  status?: ReactNode;
  actions: ReactNode;
  children: ReactNode;
  maxWidth?: DialogProps['maxWidth'];
  /** Working: the dialog cannot be closed. */
  busy?: boolean;
  minBodyHeight?: number;
  /** `danger`: the icon in red, for what cannot be undone. */
  tone?: 'accent' | 'danger';
  /** `alertdialog` for a question. */
  role?: string;
  /** Takes files dropped anywhere on it, backdrop too. */
  drop?: Pick<DialogProps, 'onDragOver' | 'onDragLeave' | 'onDrop'>;
}) {
  const titleId = `${useId()}-title`;
  return (
    <Dialog
      open={open}
      onClose={() => !busy && onClose()}
      maxWidth={maxWidth}
      fullWidth
      aria-labelledby={titleId}
      {...drop}
      slotProps={{
        paper: {
          ...(role ? { role } : {}),
          sx: {
            borderRadius: 0,
            border: `1px solid ${tokens.gray[100]}`,
            boxShadow: '0 24px 64px rgba(13, 27, 46, 0.22), 0 2px 8px rgba(13, 27, 46, 0.08)',
            maxHeight: 'calc(100% - 48px)',
          },
        },
        backdrop: { sx: { bgcolor: alpha(tokens.gray[900], 0.42) } },
      }}
    >
      <Box component="header" sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.75, px: 3, pt: 2.5, pb: 2, borderBottom: `1px solid ${tokens.gray[100]}` }}>
        {icon && (
          <Box
            aria-hidden
            sx={{
              width: 40,
              height: 40,
              flexShrink: 0,
              display: 'grid',
              placeItems: 'center',
              bgcolor: tone === 'danger' ? alpha(tokens.color.danger, 0.09) : tokens.primary[50],
              color: tone === 'danger' ? tokens.color.danger : tokens.color.accent,
              '& svg': { fontSize: 22 },
            }}
          >
            {icon}
          </Box>
        )}
        <Box sx={{ flex: 1, minWidth: 0, alignSelf: 'center' }}>
          <Typography id={titleId} component="h2" noWrap sx={{ fontSize: 18, fontWeight: 700, color: plantColor.ink, lineHeight: 1.3 }}>
            {title}
          </Typography>
          {meta && <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 0.75 }}>{meta}</Box>}
        </Box>
        <Tooltip title="Close">
          <span>
            <IconButton aria-label="Close" onClick={onClose} disabled={busy} size="small" sx={{ mt: -0.5, mr: -1, color: plantColor.muted }}>
              <CloseIcon sx={{ fontSize: 20 }} />
            </IconButton>
          </span>
        </Tooltip>
      </Box>
      <Box sx={{ flex: '1 1 auto', overflowY: 'auto', px: 3, py: 2.5, minHeight: minBodyHeight }}>{children}</Box>
      <Box
        component="footer"
        sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 3, py: 1.5, borderTop: `1px solid ${tokens.gray[100]}`, bgcolor: plantColor.header }}
      >
        <Box sx={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center' }}>{status}</Box>
        {actions}
      </Box>
    </Dialog>
  );
}

/**
 * A form dialog: header (icon, title, chips, close), a scrolling body, and a
 * footer with `status` at the left and `actions` at the right. While it fades
 * out it keeps showing what it showed open: its caller has often cleared what
 * it was showing (an edit turning into an empty "add" form for a moment).
 */
export function FormDialog(props: Parameters<typeof FormDialogView>[0]) {
  const shown = useRef(props);
  if (props.open) shown.current = props;
  return <FormDialogView {...(props.open ? props : { ...shown.current, open: false, onClose: props.onClose })} />;
}
