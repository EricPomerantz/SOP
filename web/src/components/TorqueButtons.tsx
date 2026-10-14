import { useEffect, useState } from "react";

type Props = { onPress: (ok: boolean) => Promise<string>; disabled?: boolean };

/** Stand-in for the torque wrench until it reports by itself: press when the tightening reached torque (or didn't). */
export function TorqueButtons({ onPress, disabled = false }: Props) {
  const [last, setLast] = useState("");

  const press = (ok: boolean) => {
    if (disabled) return;
    onPress(ok).then(setLast).catch((e) => setLast(e instanceof Error ? e.message : String(e)));
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || e.repeat) return;
      if (e.key === "t" || e.key === "T") press(true);
      if (e.key === "n" || e.key === "N") press(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <section className="card torque" aria-label="Torque">
      <button className="torque-ok" onClick={() => press(true)} disabled={disabled}>Torque reached <kbd>T</kbd></button>
      <button className="torque-not" onClick={() => press(false)} disabled={disabled}>Not reached <kbd>N</kbd></button>
      {last && <span className="muted" role="status">{last}</span>}
    </section>
  );
}
