import type { ReactNode } from "react";
import { CssBaseline, ThemeProvider } from "@mui/material";
import theme from "../theme";
import { GlobalAlertProvider } from "./GlobalAlertProvider";

/** The suite's theme, CSS baseline and toasts around a page (the app root and tests both use it). */
export function Providers({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <GlobalAlertProvider>{children}</GlobalAlertProvider>
    </ThemeProvider>
  );
}
