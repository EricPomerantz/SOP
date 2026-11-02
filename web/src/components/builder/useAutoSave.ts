import { useCallback, useEffect, useRef, useState } from "react";

export type SaveState = "clean" | "dirty" | "saving" | "error";

export const SAVE_TIMEOUT_MS = 10_000;
const SAVE_ATTEMPTS = 3;

class SaveTimeoutError extends Error {}

type Options<T, V> = {
  data: T;
  /** Off for a published (read-only) SOP. */
  enabled: boolean;
  save: (data: T, signal: AbortSignal) => Promise<V>;
  /** The server's answer. `stale` when edits were made while it was saving: those are saved next, so don't take its data. */
  onSaved: (view: V, stale: boolean) => void;
  /** How long after the last edit the save goes out, in ms. */
  delay?: number;
};

async function saveWithTimeout<T, V>(save: (data: T, signal: AbortSignal) => Promise<V>, data: T): Promise<V> {
  let lastError: unknown;
  for (let attempt = 0; attempt < SAVE_ATTEMPTS; attempt++) {
    const controller = new AbortController();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        save(data, controller.signal),
        new Promise<never>((_, reject) => {
          timeout = setTimeout(() => {
            reject(new SaveTimeoutError(`Save timed out after ${SAVE_TIMEOUT_MS / 1000}s`));
            controller.abort();
          }, SAVE_TIMEOUT_MS);
        }),
      ]);
    } catch (e) {
      lastError = e;
      if (!(e instanceof SaveTimeoutError) || attempt === SAVE_ATTEMPTS - 1) throw e;
    } finally {
      if (timeout !== undefined) clearTimeout(timeout);
    }
  }
  throw lastError;
}

/**
 * Saves `data` on its own: a little after the last edit, once, with whatever is current by then. An edit made while a
 * save is in flight is saved right after it. A failed save keeps the edits and says why; the next edit (or `flush`)
 * tries again. Leaving the page with unsaved edits asks first.
 */
export function useAutoSave<T, V>({ data, enabled, save, onSaved, delay = 800 }: Options<T, V>) {
  const savedRef = useRef(data);
  const dataRef = useRef(data);
  dataRef.current = data;
  const onSavedRef = useRef(onSaved);
  onSavedRef.current = onSaved;
  const saveRef = useRef(save);
  saveRef.current = save;
  const inFlight = useRef<Promise<void> | null>(null);
  const [state, setState] = useState<SaveState>("clean");
  const [error, setError] = useState("");

  const run = useCallback((): Promise<void> => {
    if (inFlight.current) return inFlight.current;
    const p = (async () => {
      try {
        while (dataRef.current !== savedRef.current) {
          const d = dataRef.current;
          setState("saving");
          try {
            const v = await saveWithTimeout(saveRef.current, d);
            savedRef.current = d;
            onSavedRef.current(v, dataRef.current !== d);
          } catch (e) {
            setState("error");
            setError(e instanceof Error ? e.message : String(e));
            return;
          }
        }
        setState("clean");
        setError("");
      } finally {
        inFlight.current = null;
      }
    })();
    inFlight.current = p;
    return p;
  }, []);

  useEffect(() => {
    if (!enabled || data === savedRef.current) return;
    setState((s) => (s === "saving" ? s : "dirty"));
    const timer = setTimeout(() => { void run(); }, delay);
    return () => clearTimeout(timer);
  }, [data, enabled, delay, run]);

  useEffect(() => {
    const guard = (e: BeforeUnloadEvent) => {
      if (dataRef.current !== savedRef.current) { e.preventDefault(); e.returnValue = ""; }
    };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, []);

  /** Save now (before an action that reads the saved file). */
  const flush = useCallback(async () => {
    if (dataRef.current !== savedRef.current) await run();
  }, [run]);
  /** The server replaced the draft (a detector import, publish, edit): this is what's saved now. */
  const markSaved = useCallback((d: T) => {
    savedRef.current = d;
    setState("clean");
    setError("");
  }, []);

  return { state, error, flush, markSaved };
}
