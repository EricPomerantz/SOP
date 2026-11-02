import React, { createContext, useCallback, useContext, useEffect, useId, useMemo, useState, useRef, type RefObject } from 'react';
import { Box, Stack } from '@mui/material';
import type { AlertColor } from '@mui/material';

export type AlertSeverity = AlertColor; // 'success' | 'info' | 'warning' | 'error'

export interface GlobalAlertOptions {
  title?: string;
  actionLabel?: string;
  onAction?: () => void;
  /** Omit or pass null to keep the toast until the user closes it. */
  autoHideDurationMs?: number | null;
}

export interface GlobalAlert {
  id: number;
  message: string;
  severity: AlertSeverity;
  options?: GlobalAlertOptions;
  exiting?: boolean;
}

interface GlobalAlertContextType {
  notify: (message: string, severity?: AlertSeverity, options?: GlobalAlertOptions) => number;
  success: (message: string, options?: GlobalAlertOptions) => number;
  info: (message: string, options?: GlobalAlertOptions) => number;
  warning: (message: string, options?: GlobalAlertOptions) => number;
  error: (message: string, options?: GlobalAlertOptions) => number;
  close: (id: number) => void;
  clearAll: () => void;
  /**
   * Keep toasts at least `px` above the bottom of the window (null releases it),
   * so they do not cover a page's bottom bar. See `useAlertClearance`.
   */
  setClearance: (key: string, px: number | null) => void;
}

const GlobalAlertContext = createContext<GlobalAlertContextType | undefined>(undefined);

function AlertGlyph({ severity }: { severity: AlertSeverity }) {
  if (severity === 'success') {
    return (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6">
        <circle cx="10" cy="10" r="7.25" />
        <path d="M6.5 10.2 8.8 12.5 13.5 7.8" />
      </svg>
    );
  }
  if (severity === 'warning') {
    return (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M10 3.2 17.2 16H2.8L10 3.2Z" />
        <path d="M10 8.2v3.4" />
        <path d="M10 14.2h.01" />
      </svg>
    );
  }
  if (severity === 'error') {
    return (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6">
        <circle cx="10" cy="10" r="7.25" />
        <path d="M7.4 7.4 12.6 12.6M12.6 7.4 7.4 12.6" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6">
      <circle cx="10" cy="10" r="7.25" />
      <path d="M10 9.1V14" />
      <path d="M10 6.4h.01" />
    </svg>
  );
}

function CloseGlyph() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M4 4 12 12M12 4 4 12" />
    </svg>
  );
}

export const useGlobalAlert = (): GlobalAlertContextType => {
  const ctx = useContext(GlobalAlertContext);
  if (!ctx) throw new Error('useGlobalAlert must be used within GlobalAlertProvider');
  return ctx;
};

/** Fired when something along the bottom of the page grows or shrinks, moving the bars above it. */
const LAYOUT_EVENT = 'app:bottom-layout';

/**
 * Tell the bars that keep toasts clear (`useAlertClearance`) to measure again:
 * something below them appeared, went or changed height, so they moved
 * without changing size.
 */
export function announceBottomLayout(): void {
  window.dispatchEvent(new Event(LAYOUT_EVENT));
}

/**
 * Keep the toasts above an element pinned to the bottom of the page (a button
 * bar, a status bar) while it is shown: they stack from its top edge instead
 * of covering it. Measured again when it resizes, the window does, or the
 * page's bottom changes (`announceBottomLayout`). Works without a provider
 * (does nothing).
 */
export function useAlertClearance(ref: RefObject<HTMLElement | null>, active = true): void {
  const ctx = useContext(GlobalAlertContext);
  const key = useId();
  const setClearance = ctx?.setClearance;
  useEffect(() => {
    const el = ref.current;
    if (!setClearance || !active || !el) return;
    const update = () => {
      const box = el.getBoundingClientRect();
      // Hidden or scrolled away: nothing to keep clear.
      const px = box.height > 0 && box.top < window.innerHeight ? Math.max(0, Math.round(window.innerHeight - box.top)) : null;
      setClearance(key, px);
    };
    update();
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(update) : null;
    observer?.observe(el);
    window.addEventListener('resize', update);
    window.addEventListener(LAYOUT_EVENT, update);
    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', update);
      window.removeEventListener(LAYOUT_EVENT, update);
      setClearance(key, null);
    };
  }, [active, key, ref, setClearance]);
}

const MAX_TOASTS = 4;
/** Space between the toasts and the window edge, or the bar they keep clear of. */
const EDGE_PX = 16;
const FADE_OUT_MS = 280;
const DEFAULT_HIDE_MS: Record<AlertSeverity, number> = {
  success: 4000,
  info: 4000,
  warning: 5000,
  error: 5000,
};

const defaultAutoHide = (severity: AlertSeverity, options?: GlobalAlertOptions): number | null => {
  if (options && 'autoHideDurationMs' in options) {
    return options.autoHideDurationMs ?? null;
  }
  return DEFAULT_HIDE_MS[severity];
};

export const GlobalAlertProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [alerts, setAlerts] = useState<GlobalAlert[]>([]);
  const [clearances, setClearances] = useState<Record<string, number>>({});
  const setClearance = useCallback((key: string, px: number | null) => {
    setClearances((prev) => {
      if (px == null) {
        if (!(key in prev)) return prev;
        const next = { ...prev };
        delete next[key];
        return next;
      }
      return prev[key] === px ? prev : { ...prev, [key]: px };
    });
  }, []);
  const clearance = Math.max(0, ...Object.values(clearances));
  const hideTimers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());
  const nextAlertId = useRef<number>(Date.now());

  const remove = useCallback((id: number) => {
    const timer = hideTimers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      hideTimers.current.delete(id);
    }
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const close = useCallback((id: number) => {
    const existing = hideTimers.current.get(id);
    if (existing) {
      clearTimeout(existing);
      hideTimers.current.delete(id);
    }
    setAlerts((prev) => {
      const current = prev.find((a) => a.id === id);
      if (!current || current.exiting) return prev;
      return prev.map((a) => (a.id === id ? { ...a, exiting: true } : a));
    });
    hideTimers.current.set(
      id,
      setTimeout(() => remove(id), FADE_OUT_MS),
    );
  }, [remove]);

  const clearAll = useCallback(() => {
    hideTimers.current.forEach((timer) => clearTimeout(timer));
    hideTimers.current.clear();
    setAlerts([]);
  }, []);

  const scheduleHide = useCallback((id: number, ms: number | null) => {
    const existing = hideTimers.current.get(id);
    if (existing) clearTimeout(existing);
    if (ms == null || ms <= 0) return;
    hideTimers.current.set(id, setTimeout(() => close(id), ms));
  }, [close]);

  useEffect(() => () => {
    hideTimers.current.forEach((timer) => clearTimeout(timer));
    hideTimers.current.clear();
  }, []);

  const add = useCallback((message: string, severity: AlertSeverity = 'info', options?: GlobalAlertOptions) => {
    const text = String(message || '').trim();
    if (!text) return -1;

    const candidateId = nextAlertId.current++;
    let id = candidateId;
    setAlerts((prev) => {
      const duplicate = prev.find((a) => a.message === text && a.severity === severity);
      if (duplicate) {
        id = duplicate.id;
        return [...prev.filter((a) => a.id !== duplicate.id), { ...duplicate, exiting: false, options }].slice(-MAX_TOASTS);
      }
      return [...prev, { id: candidateId, message: text, severity, options }].slice(-MAX_TOASTS);
    });
    scheduleHide(id, defaultAutoHide(severity, options));
    return id;
  }, [scheduleHide]);

  const value = useMemo<GlobalAlertContextType>(() => ({
    notify: add,
    success: (m, o) => add(m, 'success', o),
    info: (m, o) => add(m, 'info', o),
    warning: (m, o) => add(m, 'warning', o),
    error: (m, o) => add(m, 'error', o),
    close,
    clearAll,
    setClearance,
  }), [add, close, clearAll, setClearance]);

  return (
    <GlobalAlertContext.Provider value={value}>
      {children}
      <Box
        role="region"
        aria-label="Notifications"
        aria-live="polite"
        sx={{
          position: 'fixed',
          zIndex: (theme) => theme.zIndex.snackbar,
          // Sit in the main column so the sidebar nav stays clickable and visible.
          left: {
            xs: 12,
            sm: 'calc(var(--dw-nav-width, 0px) + 16px)',
          },
          // Left of the assistant dock when it is open (AssistantProvider sets the width).
          right: 'calc(var(--dw-assistant-width, 0px) + 16px)',
          bottom: clearance + EDGE_PX,
          transition: 'bottom 160ms ease-out',
          display: 'flex',
          justifyContent: 'flex-end',
          pointerEvents: 'none',
        }}
      >
        <Stack
          spacing={1}
          sx={{
            width: '100%',
            maxWidth: 480,
            pointerEvents: 'none',
          }}
        >
          {alerts.map(({ id, message, severity, options, exiting }) => (
            <div
              key={id}
              className={`dw-alert dw-alert-${severity}`}
              role={severity === 'error' || severity === 'warning' ? 'alert' : 'status'}
              style={{
                pointerEvents: exiting ? 'none' : 'auto',
                animation: exiting
                  ? `dwToastOut ${FADE_OUT_MS}ms ease-in forwards`
                  : 'dwToastIn 180ms ease-out',
              }}
            >
              <span className="dw-alert-icon" aria-hidden="true">
                <AlertGlyph severity={severity} />
              </span>
              <div className="dw-alert-body">
                {options?.title && <p className="dw-alert-title">{options.title}</p>}
                <div className="dw-alert-message">{message}</div>
                {options?.actionLabel && options?.onAction && (
                  <button
                    type="button"
                    className="dw-alert-link"
                    onClick={() => {
                      options.onAction?.();
                      close(id);
                    }}
                  >
                    {options.actionLabel}
                  </button>
                )}
              </div>
              <button
                type="button"
                className="dw-alert-close"
                aria-label="Close notification"
                onClick={() => close(id)}
              >
                <CloseGlyph />
              </button>
            </div>
          ))}
        </Stack>
      </Box>
    </GlobalAlertContext.Provider>
  );
};
