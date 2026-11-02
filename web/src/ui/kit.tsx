/**
 * Small building blocks in the deviceWISE suite's look (copied from its Workstation Sentinel kit,
 * frontend/src/pages/wokstationSentinel/sentinelUi.tsx, without the Sentinel-specific pieces): flat white
 * surfaces with a 1px line, square corners, status as a colored dot with text, and color only where
 * something needs attention.
 */
import { Box, ButtonBase, Typography } from '@mui/material';
import type { ReactNode } from 'react';

import { HEALTH_COLOR, HEALTH_TEXT, HEALTH_TINT, plantColor } from './plantTheme';
import type { Health } from './plantModel';

export { plantColor, HEALTH_COLOR, HEALTH_TEXT, HEALTH_TINT };
export type { Health };

const PULSE = {
  animation: 'sentinelPulse 1.4s ease-in-out infinite',
  '@keyframes sentinelPulse': { '50%': { opacity: 0.35 } },
} as const;

const alarming = (h: Health | undefined) => h === 'bad' || h === 'warn';

/** A status dot. Pending states pulse. */
export function HealthDot({ health, size = 8, pulse }: { health: Health; size?: number; pulse?: boolean }) {
  return (
    <Box
      component="span"
      aria-hidden
      sx={{
        width: size,
        height: size,
        borderRadius: '50%',
        flexShrink: 0,
        display: 'inline-block',
        bgcolor: HEALTH_COLOR[health],
        ...((pulse ?? health === 'pending') ? PULSE : {}),
      }}
    />
  );
}

/** ● Label, in the health's text color. */
export function HealthLabel({ health, label, title, strong = true }: { health: Health; label: ReactNode; title?: string; strong?: boolean }) {
  return (
    <Box
      component="span"
      title={title}
      sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.75, color: HEALTH_TEXT[health], fontWeight: strong ? 600 : 400, minWidth: 0 }}
    >
      <HealthDot health={health} />
      <Box component="span" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {label}
      </Box>
    </Box>
  );
}

/** A status pill for page headers: bordered, quiet, dot + text. */
export function StatusPill({ health, label, title }: { health: Health; label: string; title?: string }) {
  return (
    <Box
      title={title}
      role="status"
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 0.75,
        height: 30,
        px: 1.25,
        border: `1px solid ${plantColor.line}`,
        bgcolor: alarming(health) ? HEALTH_TINT[health] : plantColor.surface,
        fontSize: 12,
        fontWeight: 600,
        color: HEALTH_TEXT[health],
        whiteSpace: 'nowrap',
      }}
    >
      <HealthDot health={health} />
      {label}
    </Box>
  );
}

/** A bordered surface with an optional title row and footer. */
export function Panel({
  title,
  subtitle,
  icon,
  actions,
  footer,
  flush = false,
  health,
  children,
  sx,
  bodySx,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  footer?: ReactNode;
  /** No padding around the body (lists, media). */
  flush?: boolean;
  /** A problem colors the panel's top edge. */
  health?: Health;
  children?: ReactNode;
  sx?: object;
  bodySx?: object;
}) {
  return (
    <Box
      component="section"
      sx={{
        border: `1px solid ${plantColor.line}`,
        borderTop: alarming(health) ? `3px solid ${HEALTH_COLOR[health!]}` : `1px solid ${plantColor.line}`,
        bgcolor: plantColor.surface,
        display: 'flex',
        flexDirection: 'column',
        minWidth: 0,
        ...sx,
      }}
    >
      {(title || actions) && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1.5, py: 0.5, minHeight: 40, borderBottom: `1px solid ${plantColor.line}`, flexShrink: 0 }}>
          {icon && <Box sx={{ color: plantColor.accent, display: 'flex', '& svg': { fontSize: 18 } }}>{icon}</Box>}
          <Box sx={{ flex: 1, minWidth: 0 }}>
            {title && (
              <Typography component="h2" noWrap sx={{ fontSize: 13.5, fontWeight: 700, color: plantColor.ink, lineHeight: 1.3 }}>
                {title}
              </Typography>
            )}
            {subtitle && (
              <Typography noWrap sx={{ fontSize: 11.5, color: plantColor.muted, lineHeight: 1.3 }}>
                {subtitle}
              </Typography>
            )}
          </Box>
          {actions && <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>{actions}</Box>}
        </Box>
      )}
      <Box sx={{ p: flush ? 0 : 1.5, flex: 1, minHeight: 0, minWidth: 0, ...bodySx }}>{children}</Box>
      {footer && (
        <Box sx={{ borderTop: `1px solid ${plantColor.line}`, bgcolor: plantColor.header, px: 1.5, py: 0.75, fontSize: 12, color: plantColor.muted, flexShrink: 0 }}>
          {footer}
        </Box>
      )}
    </Box>
  );
}

/** Uppercase section label inside a panel. */
export function SectionLabel({ children, actions }: { children: ReactNode; actions?: ReactNode }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.75, minHeight: 22 }}>
      <Typography component="h3" sx={{ fontSize: 11.5, fontWeight: 700, color: plantColor.muted, textTransform: 'uppercase', letterSpacing: 0.4, flex: 1 }}>
        {children}
      </Typography>
      {actions}
    </Box>
  );
}

/** Compact label / value rows. Empty values show a faint dash. */
export function InfoRows({ rows }: { rows: Array<[string, ReactNode]> }) {
  return (
    <Box
      component="dl"
      sx={{ m: 0, display: 'grid', gridTemplateColumns: 'max-content minmax(0, 1fr)', alignItems: 'baseline', columnGap: 1.5, rowGap: 0.5, fontSize: 12.5 }}
    >
      {rows.map(([label, value]) => (
        <Box key={label} sx={{ display: 'contents' }}>
          <Box component="dt" sx={{ color: plantColor.muted, whiteSpace: 'nowrap' }}>
            {label}
          </Box>
          <Box component="dd" sx={{ m: 0, color: plantColor.ink, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {value === '' || value == null ? <Box component="span" sx={{ color: plantColor.faint }}>—</Box> : value}
          </Box>
        </Box>
      ))}
    </Box>
  );
}

/** One headline number. A problem gets a colored edge and a wash. */
export function KpiTile({
  icon,
  label,
  value,
  detail,
  health,
  onClick,
  selected,
}: {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  health?: Health;
  onClick?: () => void;
  selected?: boolean;
}) {
  const hot = alarming(health);
  const body = (
    <>
      <Box
        sx={{
          width: 34,
          height: 34,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: hot ? 'transparent' : plantColor.selected,
          color: hot ? HEALTH_TEXT[health!] : plantColor.accent,
          '& svg': { fontSize: 20 },
        }}
      >
        {icon}
      </Box>
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography sx={{ fontSize: 11.5, fontWeight: 700, color: plantColor.muted, textTransform: 'uppercase', letterSpacing: 0.4, lineHeight: 1.3 }} noWrap>
          {label}
        </Typography>
        <Typography sx={{ fontSize: 22, fontWeight: 700, color: hot ? HEALTH_TEXT[health!] : plantColor.ink, fontVariantNumeric: 'tabular-nums', lineHeight: 1.25 }}>
          {value}
        </Typography>
        {detail && (
          <Typography component="div" sx={{ fontSize: 12, color: plantColor.muted, lineHeight: 1.35 }} noWrap>
            {detail}
          </Typography>
        )}
      </Box>
    </>
  );
  const sx = {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 1.25,
    p: 1.5,
    width: '100%',
    textAlign: 'left' as const,
    border: `1px solid ${selected ? plantColor.accent : plantColor.line}`,
    borderLeft: hot ? `3px solid ${HEALTH_COLOR[health!]}` : `1px solid ${selected ? plantColor.accent : plantColor.line}`,
    bgcolor: hot ? HEALTH_TINT[health!] : selected ? plantColor.selected : plantColor.surface,
    minWidth: 0,
  };
  if (onClick) {
    return (
      <ButtonBase
        component="div"
        onClick={onClick}
        aria-pressed={selected}
        sx={{ ...sx, justifyContent: 'flex-start', '&:hover': { borderColor: plantColor.accent }, '&:focus-visible': { outline: `2px solid ${plantColor.accent}`, outlineOffset: 1 } }}
      >
        {body}
      </ButtonBase>
    );
  }
  return <Box sx={sx}>{body}</Box>;
}

/** A quiet centered message for empty lists and missing setup. */
export function EmptyNote({ icon, title, children, action, compact = false }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode; compact?: boolean }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', gap: 0.75, py: compact ? 1.5 : 4, px: 2 }}>
      {icon && <Box sx={{ color: plantColor.faint, display: 'flex', '& svg': { fontSize: compact ? 24 : 36 } }}>{icon}</Box>}
      <Typography sx={{ fontSize: compact ? 12.5 : 14, fontWeight: 600, color: plantColor.ink }}>{title}</Typography>
      {children && <Typography component="div" sx={{ fontSize: 12, color: plantColor.muted, maxWidth: 420 }}>{children}</Typography>}
      {action && <Box sx={{ mt: 0.75 }}>{action}</Box>}
    </Box>
  );
}

/** A small tag over media (dark) or on a surface. */
export function Tag({ children, tone = 'dark' }: { children: ReactNode; tone?: 'dark' | 'bad' | 'accent' }) {
  const bg = tone === 'bad' ? HEALTH_COLOR.bad : tone === 'accent' ? plantColor.accent : 'rgba(0,0,0,0.55)';
  return (
    <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', height: 20, px: 0.75, bgcolor: bg, color: '#fff', fontSize: 10.5, fontWeight: 700, letterSpacing: 0.4 }}>
      {children}
    </Box>
  );
}
