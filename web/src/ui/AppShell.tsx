import type { ReactNode } from "react";
import { Box, Stack, Typography } from "@mui/material";
import { ArrowBack as BackIcon } from "@mui/icons-material";
import tokens from "@/theme/tokens.json";
import { pageBodySx } from "./pageStyles";
import { plantColor } from "./plantTheme";

export type PageId = "dashboard" | "sops" | "record" | "latency";

const PAGES: { id: PageId; label: string; href: string }[] = [
  { id: "dashboard", label: "Dashboard", href: "#/" },
  { id: "sops", label: "SOPs", href: "#/sops" },
  { id: "record", label: "Record", href: "#/record" },
  { id: "latency", label: "Camera latency", href: "#/latency" },
];

type Props = {
  title: string;
  /** Small monospace context under the title (a version, an id). */
  subtitle?: ReactNode;
  /** Pills and buttons at the right. */
  status?: ReactNode;
  /** Which page link is the current one. */
  current?: PageId;
  /** A sub-page of another page: a back link replaces the page navigation. */
  back?: { href: string; label: string };
  /** Page body width; the dashboard and the builder pages fit 1400px. */
  maxWidth?: number;
  children?: ReactNode;
};

/**
 * The page frame: a top bar in the suite's PageHeader look (its logo mark, the page title, the page links,
 * status at the right) and the body below it. Kept as a bar rather than the suite's sidebar: when this app moves
 * into the suite, its pages become entries of that sidebar and this frame goes.
 */
export function AppShell({ title, subtitle, status, current, back, maxWidth = 1400, children }: Props) {
  return (
    <Box sx={{ minHeight: "100vh", display: "flex", flexDirection: "column", bgcolor: plantColor.canvas }}>
      <Box component="header" sx={{ bgcolor: tokens.gray[50], borderBottom: "2px solid", borderBottomColor: "primary.main", flexShrink: 0 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, px: { xs: 2, md: 3 }, py: 1.25, flexWrap: "wrap" }}>
          <Stack direction="row" alignItems="center" spacing={1.5} sx={{ minWidth: 0 }}>
            <Box aria-hidden sx={{ width: 26, height: 26, flexShrink: 0, bgcolor: "primary.main", color: "#fff", display: "grid", placeItems: "center" }}>
              <svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8">
                <rect x="2" y="2" width="12" height="12" />
                <path d="M2 8h12M8 2v12" />
              </svg>
            </Box>
            <Box sx={{ minWidth: 0 }}>
              <Typography component="h1" variant="subtitle1" fontWeight={700} noWrap sx={{ letterSpacing: "-0.01em", lineHeight: 1.2 }}>
                {title}
              </Typography>
              {subtitle && (
                <Typography variant="caption" color="text.secondary" sx={{ fontFamily: "monospace", fontSize: "0.7rem", lineHeight: 1 }}>
                  {subtitle}
                </Typography>
              )}
            </Box>
          </Stack>
          {back ? (
            <Box component="a" href={back.href} sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, fontSize: 13, fontWeight: 600, color: "primary.main", textDecoration: "none", "&:hover": { textDecoration: "underline" } }}>
              <BackIcon sx={{ fontSize: 16 }} />
              {back.label}
            </Box>
          ) : (
            <Box component="nav" aria-label="Pages" sx={{ display: "flex", gap: 0.5, flexWrap: "wrap" }}>
              {PAGES.map((p) => (
                <Box
                  key={p.id}
                  component="a"
                  href={p.href}
                  aria-current={current === p.id ? "page" : undefined}
                  sx={{
                    px: 1.25, py: 0.5, fontSize: 13, fontWeight: 600, textDecoration: "none",
                    color: current === p.id ? "primary.main" : plantColor.muted,
                    bgcolor: current === p.id ? plantColor.selected : "transparent",
                    "&:hover": { color: plantColor.ink, bgcolor: plantColor.lineSoft },
                  }}
                >
                  {p.label}
                </Box>
              ))}
            </Box>
          )}
          {status && <Stack direction="row" alignItems="center" spacing={1.5} sx={{ ml: "auto", flexWrap: "wrap" }}>{status}</Stack>}
        </Box>
      </Box>
      <Box sx={{ ...pageBodySx, maxWidth, mx: "auto", flex: 1, display: "flex", flexDirection: "column", gap: 1.5 }}>{children}</Box>
    </Box>
  );
}
