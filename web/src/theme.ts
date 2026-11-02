import { createTheme, alpha } from '@mui/material/styles';
import tokens from './theme/tokens.json';

// Colors, type, and radius come from theme/tokens.json.
// That file is the source for the suite and for Visual Inspection.

const ACCENT = tokens.color.accent;

const alertTone = (color: string) => ({
  backgroundColor: tokens.color.surface,
  color: tokens.color.ink,
  borderLeftColor: color,
  '& .MuiAlert-icon': { color },
});

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: ACCENT,
      light: tokens.color.accentLight,
      dark: tokens.color.accentDark,
      contrastText: '#ffffff',
    },
    secondary: {
      main: tokens.color.secondary,
      light: tokens.color.secondaryLight,
      dark: tokens.color.secondaryDark,
    },
    background: {
      default: tokens.color.canvas,
      paper:   tokens.color.surface,
    },
    text: {
      primary:   tokens.color.ink,
      secondary: tokens.color.muted,
      disabled:  tokens.color.disabled,
    },
    divider: tokens.color.line,
    success: { main: tokens.color.success },
    warning: { main: tokens.color.warning },
    error: { main: tokens.color.danger },
    info: { main: ACCENT },
    action: {
      hover:    tokens.color.hover,
      selected: tokens.color.selected,
      focus:    tokens.color.focus,
    },
  },
  typography: {
    fontFamily: tokens.font.sans,
    h1: { fontWeight: 700, letterSpacing: '-0.02em' },
    h2: { fontWeight: 700, letterSpacing: '-0.02em' },
    h3: { fontWeight: 600, letterSpacing: '-0.015em' },
    h4: { fontWeight: 600, letterSpacing: '-0.015em' },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
    subtitle1: { fontWeight: 500 },
    subtitle2: { fontWeight: 600, fontSize: '0.875rem' },
    body1:  { fontSize: '0.9375rem', lineHeight: 1.55 },
    body2:  { fontSize: '0.875rem',  lineHeight: 1.55 },
    button: { textTransform: 'none', fontWeight: 600, letterSpacing: '0.02em' },
    overline: { letterSpacing: '0.1em', fontWeight: 700, fontSize: '0.65rem' },
  },
  shape: {
    borderRadius: 0, // Sharp, industrial — no rounded corners
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: tokens.color.canvas,
          color: tokens.color.ink,
          scrollbarWidth: 'thin',
          '&::-webkit-scrollbar': { width: '5px', height: '5px' },
          '&::-webkit-scrollbar-track': { background: 'transparent' },
          '&::-webkit-scrollbar-thumb': { background: tokens.color.scrollbar, borderRadius: 0 },
          '&::-webkit-scrollbar-thumb:hover': { background: tokens.color.scrollbarHover },
        },
      },
    },
    MuiPaper: {
      defaultProps: { elevation: 0 },
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 0,
          border: `1px solid ${tokens.color.line}`,
        },
        elevation0: { boxShadow: 'none' },
        elevation1: { boxShadow: `0 1px 4px ${tokens.color.line}`, border: 'none' },
        elevation2: { boxShadow: '0 2px 8px rgba(0,0,0,0.1)',  border: 'none' },
      },
    },
    MuiCard: {
      defaultProps: { elevation: 0 },
      styleOverrides: {
        root: {
          borderRadius: 0,
          border: `1px solid ${tokens.color.line}`,
          boxShadow: 'none',
          backgroundColor: tokens.color.surface,
          transition: 'box-shadow 0.15s ease',
          '&:hover': { boxShadow: '0 2px 8px rgba(0,0,0,0.1)' },
        },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: { '&:last-child': { paddingBottom: 16 } },
      },
    },
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: {
          borderRadius: 0,
          padding: '6px 16px',
          fontSize: '0.8125rem',
          fontWeight: 600,
          letterSpacing: '0.03em',
          transition: 'background 0.12s ease, box-shadow 0.12s ease',
        },
        contained: {
          '&:hover': { boxShadow: '0 2px 6px rgba(0,0,0,0.2)' },
        },
        outlined: {
          borderColor: 'rgba(0,0,0,0.18)',
          color: tokens.color.secondaryDark,
          '&:hover': {
            backgroundColor: tokens.color.canvas,
            borderColor: 'rgba(0,0,0,0.28)',
            color: tokens.color.ink,
          },
        },
        text: {
          color: tokens.color.secondaryDark,
          '&:hover': { backgroundColor: 'rgba(0,0,0,0.04)' },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          color: tokens.color.secondary,
          transition: 'background 0.12s ease',
          '&:hover': { backgroundColor: 'rgba(0,0,0,0.05)', color: tokens.color.ink },
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          '&.Mui-selected': {
            backgroundColor: tokens.color.selected,
            '&:hover': { backgroundColor: 'rgba(0,47,112,0.09)' },
          },
          '&:hover': { backgroundColor: 'rgba(0,0,0,0.04)' },
        },
      },
    },
    MuiListItemIcon: {
      styleOverrides: {
        root: { minWidth: 36, color: tokens.color.secondary },
      },
    },
    MuiInputBase: {
      styleOverrides: { root: { borderRadius: 0 } },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(0,0,0,0.18)' },
          '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(0,0,0,0.32)' },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: ACCENT,
            borderWidth: '1px',
            boxShadow: `0 0 0 2px ${alpha(ACCENT, 0.1)}`,
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: { icon: { color: tokens.color.secondary } },
    },
    MuiChip: {
      styleOverrides: {
        root:     { borderRadius: 2, fontWeight: 500, fontSize: '0.75rem' },
        outlined: { borderColor: 'rgba(0,0,0,0.15)' },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid rgba(0,0,0,0.07)',
          padding: '10px 16px',
          fontSize: '0.8125rem',
          color: tokens.color.ink,
        },
        head: {
          fontWeight: 700,
          fontSize: '0.75rem',
          letterSpacing: '0.04em',
          color: tokens.color.muted,
          backgroundColor: tokens.color.canvas,
          textTransform: 'uppercase',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:hover': { backgroundColor: 'rgba(0,0,0,0.02)' },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        indicator: { backgroundColor: ACCENT, height: 2 },
        root: { borderBottom: `1px solid ${tokens.color.line}` },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          fontWeight: 600,
          fontSize: '0.8125rem',
          textTransform: 'none',
          minHeight: 40,
          padding: '8px 16px',
          color: tokens.color.muted,
          '&.Mui-selected': { color: ACCENT },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: tokens.color.surface,
          borderBottom: `1px solid ${tokens.color.line}`,
          color: tokens.color.ink,
          boxShadow: 'none',
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: { borderRadius: 0, fontSize: '0.75rem', backgroundColor: tokens.color.ink },
        arrow:   { color: tokens.color.ink },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          backgroundColor: tokens.color.surface,
          color: tokens.color.ink,
          border: `1px solid ${tokens.color.line}`,
          borderLeftWidth: 3,
          borderLeftStyle: 'solid',
          alignItems: 'flex-start',
          padding: '10px 12px 10px 14px',
          fontSize: '0.8125rem',
          boxShadow: 'none',
          '& .MuiAlert-icon': {
            padding: '1px 0 0',
            marginRight: 10,
            opacity: 1,
            fontSize: 20,
          },
          '& .MuiAlert-message': {
            padding: '1px 0 0',
            fontSize: '0.8125rem',
            lineHeight: 1.45,
            color: tokens.color.ink,
            overflowWrap: 'anywhere',
          },
          '& .MuiAlert-action': { padding: '0 0 0 8px', marginRight: 0, alignItems: 'flex-start' },
        },
        standardSuccess: alertTone(tokens.color.success),
        standardInfo: alertTone(ACCENT),
        standardWarning: alertTone(tokens.color.warning),
        standardError: alertTone(tokens.color.danger),
        filledSuccess: alertTone(tokens.color.success),
        filledInfo: alertTone(ACCENT),
        filledWarning: alertTone(tokens.color.warning),
        filledError: alertTone(tokens.color.danger),
        outlinedSuccess: alertTone(tokens.color.success),
        outlinedInfo: alertTone(ACCENT),
        outlinedWarning: alertTone(tokens.color.warning),
        outlinedError: alertTone(tokens.color.danger),
      },
    },
    MuiAlertTitle: {
      styleOverrides: {
        root: {
          margin: '0 0 2px',
          fontSize: '0.8125rem',
          fontWeight: 600,
          lineHeight: 1.35,
          color: tokens.color.ink,
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: { borderRadius: 0 },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: { borderRadius: 0 },
      },
    },
    MuiPopover: {
      styleOverrides: {
        paper: { borderRadius: 0 },
      },
    },
    MuiSnackbar: {
      styleOverrides: {
        root: { '& .MuiPaper-root': { borderRadius: 0 } },
      },
    },
  },
});

export default theme;
