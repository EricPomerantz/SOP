/** Styles of the Factory's form dialogs (see `dialogKit`). */
import { alpha } from '@mui/material/styles';

import tokens from '@/theme/tokens.json';
import { plantBtn, plantColor } from './plantTheme';

const ring = alpha(tokens.color.accent, 0.16);

/** Text fields, selects and autocompletes in these forms: the label sits above (see `FormField`). */
export const fieldSx = {
  '& .MuiOutlinedInput-root': {
    borderRadius: 0,
    fontSize: 13.5,
    bgcolor: plantColor.surface,
    transition: 'box-shadow 120ms ease, border-color 120ms ease',
    '& fieldset': { borderColor: tokens.gray[200] },
    '&:hover:not(.Mui-disabled) fieldset': { borderColor: tokens.gray[400] },
    '&.Mui-focused fieldset': { borderColor: tokens.color.accent, borderWidth: 1 },
    '&.Mui-focused': { boxShadow: `0 0 0 3px ${ring}` },
    '&.Mui-error fieldset': { borderColor: tokens.color.danger },
    '&.Mui-error.Mui-focused': { boxShadow: `0 0 0 3px ${alpha(tokens.color.danger, 0.14)}` },
    '&.Mui-disabled': { bgcolor: tokens.gray[50] },
    '&.Mui-disabled fieldset': { borderColor: tokens.gray[100] },
  },
  '& .MuiOutlinedInput-input': { py: '9px', px: '12px' },
  '& .MuiInputBase-inputMultiline': { py: 0, px: 0 },
  '& .MuiOutlinedInput-root.MuiInputBase-multiline': { py: '9px', px: '12px' },
  '& .MuiInputBase-input.Mui-disabled': { WebkitTextFillColor: plantColor.muted },
  '& .MuiInputBase-input::placeholder': { color: plantColor.faint, opacity: 1 },
  '& .MuiFormHelperText-root': { mx: 0, mt: 0.5, fontSize: 12 },
  // An autocomplete as tall as a text field.
  '& .MuiAutocomplete-inputRoot': { py: '0 !important', pl: '3px !important' },
  '& .MuiAutocomplete-inputRoot .MuiAutocomplete-input': { py: '9px !important' },
} as const;

/** The dropdown of a select or autocomplete in these forms. */
export const menuPaperSx = { borderRadius: 0, mt: 0.5, boxShadow: '0 8px 24px rgba(13, 27, 46, 0.14)', border: `1px solid ${tokens.gray[100]}` } as const;

/** Buttons in the footer, a little taller than the table toolbars'. */
export const dialogBtn = { ...plantBtn, height: 34, px: 2, fontSize: 13 } as const;

