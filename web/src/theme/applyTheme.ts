import tokens from './tokens.json';

export type ThemeTokens = typeof tokens;

function hexChannels(hex: string): string | null {
  const match = /^#([0-9a-fA-F]{6})$/.exec(hex);
  if (!match) return null;
  const value = parseInt(match[1], 16);
  return `${(value >> 16) & 255} ${(value >> 8) & 255} ${value & 255}`;
}

/** Paint CSS variables from tokens.json. Call once before render. */
export function applyThemeVars(target: HTMLElement = document.documentElement) {
  const color = tokens.color;
  const pairs: Record<string, string> = {
    '--dw-accent': color.accent,
    '--dw-accent-light': color.accentLight,
    '--dw-accent-dark': color.accentDark,
    '--dw-bg': color.canvas,
    '--dw-surface': color.surface,
    '--dw-header': color.header,
    '--dw-ink': color.ink,
    '--dw-muted': color.muted,
    '--dw-disabled': color.disabled,
    '--dw-line': color.line,
    '--dw-sidebar': color.sidebar,
    '--dw-sidebar-text': color.sidebarText,
    '--dw-sidebar-text-dim': color.sidebarTextDim,
    '--dw-sidebar-icon': color.sidebarIcon,
    '--dw-sidebar-icon-active': color.sidebarIconActive,
    '--dw-sidebar-hover': color.sidebarHover,
    '--dw-sidebar-active': color.sidebarActive,
    '--dw-sidebar-border': color.sidebarBorder,
    '--dw-selected': color.selected,
    '--dw-focus': color.focus,
    '--dw-hover': color.hover,
    '--dw-secondary': color.secondary,
    '--dw-scrollbar': color.scrollbar,
    '--dw-scrollbar-hover': color.scrollbarHover,
    '--dw-success': color.success,
    '--dw-warning': color.warning,
    '--dw-danger': color.danger,
    '--dw-font': tokens.font.sans,
    '--dw-radius': tokens.radius,
  };

  const scales: Record<string, Record<string, string>> = {
    primary: tokens.primary,
    secondary: tokens.secondary,
    gray: tokens.gray,
  };

  Object.entries(scales).forEach(([name, scale]) => {
    Object.entries(scale).forEach(([step, hex]) => {
      pairs[`--dw-${name}-${step}`] = hex;
      const channels = hexChannels(hex);
      if (channels) pairs[`--dw-${name}-${step}-rgb`] = channels;
    });
  });

  Object.entries(pairs).forEach(([name, value]) => {
    target.style.setProperty(name, value);
  });
}

export default tokens;
