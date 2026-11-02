import { Box, Stack, Typography } from '@mui/material';
import type { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  /** Optional icon shown left of title */
  icon?: ReactNode;
  actions?: ReactNode;
}

/**
 * Full-bleed page top bar — edge-to-edge in the main content area,
 * flush to the top (no layout top padding), primary bottom accent.
 */
export const PageHeader = ({ title, subtitle, icon, actions }: PageHeaderProps) => (
  <Box
    sx={{
      width: '100%',
      flexShrink: 0,
      bgcolor: 'grey.50',
      borderBottom: '2px solid',
      borderBottomColor: 'primary.main',
      borderRadius: 0,
    }}
  >
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        px: { xs: 2, md: 3 },
        py: 1.5,
        flexWrap: 'wrap',
        gap: 1,
      }}
    >
      <Stack direction="row" alignItems="center" spacing={1.5}>
        {icon && (
          <Box sx={{ display: 'flex', alignItems: 'center', color: 'primary.main' }}>
            {icon}
          </Box>
        )}
        <Box>
          <Typography
            variant="subtitle1"
            component="h1"
            fontWeight={700}
            color="text.primary"
            sx={{ letterSpacing: '-0.01em', lineHeight: 1.2 }}
          >
            {title}
          </Typography>
          {subtitle && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ fontFamily: 'monospace', fontSize: '0.7rem', lineHeight: 1 }}
            >
              {subtitle}
            </Typography>
          )}
        </Box>
      </Stack>

      {actions && (
        <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap">
          {actions}
        </Stack>
      )}
    </Box>
  </Box>
);
