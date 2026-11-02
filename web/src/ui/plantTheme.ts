import { alpha, darken } from '@mui/material/styles';

import tokens from '@/theme/tokens.json';
import type { Health } from './plantModel';

export const plantColor = {
  ink: tokens.color.ink,
  muted: tokens.color.muted,
  faint: tokens.color.disabled,
  line: tokens.gray[200],
  lineSoft: tokens.gray[100],
  canvas: tokens.color.canvas,
  surface: tokens.color.surface,
  header: tokens.color.header,
  accent: tokens.color.accent,
  selected: tokens.primary[50],
  /** Neutral tag and chip text; darker than `muted` so it stays readable on gray fills. */
  subtle: tokens.gray[600],
};

/**
 * Square, compact buttons used across the Factory inspector and tables. The app
 * theme pads every button 6px 16px with wide tracking; that is too loud here.
 */
export const plantBtn = {
  borderRadius: 0,
  textTransform: 'none' as const,
  fontWeight: 600,
  fontSize: 12.5,
  letterSpacing: 0,
  px: 1.25,
  py: 0.5,
};

/** Filter toggles in the table toolbars: quiet until selected. */
export const plantToggleBtn = {
  borderRadius: 0,
  textTransform: 'none' as const,
  fontWeight: 600,
  fontSize: 12,
  lineHeight: 1.4,
  height: 32,
  px: 1.25,
  py: 0,
  color: plantColor.muted,
  bgcolor: plantColor.surface,
  borderColor: plantColor.line,
  '&:hover': { bgcolor: plantColor.lineSoft },
  '&.Mui-selected, &.Mui-selected:hover': { color: plantColor.accent, bgcolor: plantColor.selected },
};

/** Text fields and selects in the table toolbars. */
export const plantInputSx = {
  '& .MuiOutlinedInput-root': { borderRadius: 0, fontSize: 12.5, bgcolor: plantColor.surface },
  '& .MuiOutlinedInput-input': { py: '7px' },
};

/** Dots, bars, and borders. */
export const HEALTH_COLOR: Record<Health, string> = {
  ok: tokens.color.success,
  pending: tokens.primary[300],
  idle: tokens.gray[400],
  unknown: tokens.gray[300],
  warn: tokens.color.warning,
  bad: tokens.color.danger,
};

/** Text in a health color. Raw warning orange is 3.5:1 on white, so text uses darker shades. */
export const HEALTH_TEXT: Record<Health, string> = {
  ok: darken(tokens.color.success, 0.15),
  pending: tokens.primary[500],
  idle: tokens.gray[600],
  unknown: tokens.gray[600],
  warn: darken(tokens.color.warning, 0.3),
  bad: darken(tokens.color.danger, 0.15),
};

/** Background washes. Healthy things stay white; only problems get a tint. */
export const HEALTH_TINT: Record<Health, string> = {
  ok: alpha(tokens.color.success, 0.08),
  pending: alpha(tokens.primary[300], 0.12),
  idle: tokens.gray[100],
  unknown: tokens.gray[100],
  warn: alpha(tokens.color.warning, 0.08),
  bad: alpha(tokens.color.danger, 0.06),
};

export const HEALTH_LABEL: Record<Health, string> = {
  ok: 'Healthy',
  pending: 'In progress',
  idle: 'Stopped',
  unknown: 'Not reported',
  warn: 'Degraded',
  bad: 'Fault',
};

/** Health that needs someone to look at it. Stopped and healthy stay quiet. */
export function isAlarming(health: Health): boolean {
  return health === 'bad' || health === 'warn';
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return 'never';
  const seconds = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return new Date(iso).toLocaleString();
}
