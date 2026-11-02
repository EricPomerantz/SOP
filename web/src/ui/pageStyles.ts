/**
 * Shared industrial design tokens and components.
 * All pages (except Chat) import from here for visual consistency.
 */


import type { SxProps, Theme } from '@mui/material';

// ─── Style tokens ────────────────────────────────────────────────────────────

/** Card / Paper — flat, sharp, with subtle border */
export const panelSx: SxProps<Theme> = {
  borderRadius: 0,
  border: '1px solid',
  borderColor: 'grey.300',
  boxShadow: 'none',
  bgcolor: 'background.paper',
};

/** Panel section header row */
export const panelHeaderSx: SxProps<Theme> = {
  px: 1.5,
  py: 1,
  borderBottom: '1px solid',
  borderColor: 'grey.200',
  bgcolor: 'grey.100',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
};

/** Buttons — sharp, no text transform */
export const btnSx = {
  borderRadius: 0,
  textTransform: 'none' as const,
  fontWeight: 600,
};

/** Chips — sharp, bold */
export const chipSx = {
  borderRadius: 0,
  fontWeight: 600,
  letterSpacing: '0.01em',
};

/** Text inputs — force sharp corners regardless of theme */
export const inputSx = {
  '& .MuiOutlinedInput-root': { borderRadius: 0 },
  '& .MuiInputBase-root': { borderRadius: 0 },
};

/** Standard padded page body below the full-bleed PageHeader / PageHeroBar */
export const pageBodySx: SxProps<Theme> = {
  px: { xs: 2, md: 3 },
  pt: 2.5,
  pb: { xs: 2, md: 3 },
  width: '100%',
  boxSizing: 'border-box',
};
