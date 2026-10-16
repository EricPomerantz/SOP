import type { SlotData, SopData } from "../../builderApi";

type Props = { data: SopData; readOnly: boolean; onChange: (d: SopData) => void; selected?: string; onSelect?: (s: string) => void };

export const partClasses = (d: SopData) => Object.entries(d.detector.roles ?? {}).filter(([, r]) => r === "part").map(([c]) => c);
export const emptyClasses = (d: SopData) => Object.entries(d.detector.roles ?? {}).filter(([, r]) => r === "empty").map(([c]) => c);
const fills = (s: SlotData) => (Array.isArray(s.filled_by) ? s.filled_by : [s.filled_by]);

/** The slots as a table: name, position on the anchor (0-1), what fills each one, and its match radius. */
export function SlotTable({ data, readOnly, onChange, selected, onSelect }: Props) {
  const slots = data.layout.slots ?? {};
  const parts = partClasses(data);
  const empties = emptyClasses(data);
  const setSlots = (next: Record<string, SlotData>) => onChange({ ...data, layout: { ...data.layout, slots: next } });
  const patch = (name: string, p: Partial<SlotData>) => setSlots({ ...slots, [name]: { ...slots[name], ...p } });
  const rename = (from: string, to: string) => {
    if (!to || to === from || slots[to]) return;
    const steps = data.steps.map((s) => (s.evidence && from in s.evidence
      ? { ...s, evidence: Object.fromEntries(Object.entries(s.evidence).map(([k, v]) => [k === from ? to : k, v])) } : s));
    onChange({ ...data, steps, layout: { ...data.layout, slots: Object.fromEntries(Object.entries(slots).map(([k, v]) => [k === from ? to : k, v])) } });
  };
  const toggleFill = (name: string, cls: string) => {
    const cur = fills(slots[name]).filter(Boolean);
    const next = cur.includes(cls) ? cur.filter((c) => c !== cls) : [...cur, cls];
    patch(name, { filled_by: next.length === 1 ? next[0] : next });
  };
  const num = (v: string) => Math.min(1, Math.max(0, Number(v)));

  return (
    <table className="grid-table" aria-label="Slots">
      <thead><tr><th>Slot</th><th>x</th><th>y</th><th>Filled by</th><th>Empty marker</th><th>Radius</th><th></th></tr></thead>
      <tbody>
        {Object.entries(slots).map(([name, s]) => (
          <tr key={name} className={selected === name ? "selected" : undefined} onClick={() => onSelect?.(name)}>
            <td><input aria-label={`Name of ${name}`} className="mono" defaultValue={name} disabled={readOnly}
              onBlur={(e) => rename(name, e.target.value.trim())} /></td>
            <td><input aria-label={`x of ${name}`} type="number" step="0.01" min="0" max="1" value={s.at[0]} disabled={readOnly}
              onChange={(e) => patch(name, { at: [num(e.target.value), s.at[1]] })} /></td>
            <td><input aria-label={`y of ${name}`} type="number" step="0.01" min="0" max="1" value={s.at[1]} disabled={readOnly}
              onChange={(e) => patch(name, { at: [s.at[0], num(e.target.value)] })} /></td>
            <td className="nowrap">
              {parts.map((c) => (
                <label key={c} className="check"><input type="checkbox" checked={fills(s).includes(c)} disabled={readOnly}
                  onChange={() => toggleFill(name, c)} /> {c}</label>
              ))}
            </td>
            <td><select aria-label={`Empty marker of ${name}`} value={s.empty_as ?? ""} disabled={readOnly}
              onChange={(e) => patch(name, { empty_as: e.target.value || null })}>
              <option value="">none (unseen = unknown)</option>
              {empties.map((c) => <option key={c} value={c}>{c}</option>)}
            </select></td>
            <td><input aria-label={`Radius of ${name}`} type="number" step="0.01" min="0" max="1" placeholder={String(data.layout.match_radius)}
              value={s.match_radius ?? ""} disabled={readOnly}
              onChange={(e) => patch(name, { match_radius: e.target.value === "" ? null : num(e.target.value) })} /></td>
            <td><button className="icon" aria-label={`Delete ${name}`} disabled={readOnly}
              onClick={() => { const { [name]: _, ...rest } = slots; setSlots(rest); }}>✕</button></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** A new slot at (x, y) with the draft's parts and first empty marker. */
export function addSlot(data: SopData, x = 0.5, y = 0.5): { data: SopData; name: string } {
  const slots = data.layout.slots ?? {};
  let n = Object.keys(slots).length + 1;
  while (slots[`slot_${n}`]) n += 1;
  const parts = partClasses(data);
  const slot: SlotData = { at: [Number(x.toFixed(3)), Number(y.toFixed(3))], filled_by: parts.length === 1 ? parts[0] : parts,
    empty_as: emptyClasses(data)[0] ?? null };
  return { data: { ...data, layout: { ...data.layout, slots: { ...slots, [`slot_${n}`]: slot } } }, name: `slot_${n}` };
}
