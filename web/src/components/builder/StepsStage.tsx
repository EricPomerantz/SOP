import type { BuilderApi, Evidence, SopData, StepData, SopView } from "../../builderApi";

type Props = { api: BuilderApi; view: SopView; data: SopData; onChange: (d: SopData) => void };

type GroupKind = "any" | "sequence";

/** Which group each step is in (index) and each group's kind, from the draft's `groups`. */
function groupsOf(data: SopData): { of: Record<string, number>; kinds: GroupKind[] } {
  const of: Record<string, number> = {};
  const kinds: GroupKind[] = [];
  (data.groups ?? []).forEach((g, i) => {
    const ids = Array.isArray(g) ? g : g.sequence;
    kinds.push(Array.isArray(g) ? "any" : "sequence");
    ids.forEach((id) => { of[id] = i; });
  });
  return { of, kinds };
}

/** Groups rebuilt from per-step group numbers; a sequence group runs in the steps' list order. */
function buildGroups(steps: StepData[], of: Record<string, number>, kinds: GroupKind[]): SopData["groups"] {
  return kinds.map((kind, i) => {
    const ids = steps.filter((s) => (of[s.id] ?? 0) === i).map((s) => s.id);
    return kind === "sequence" ? { sequence: ids } : ids;
  });
}

/** The steps, what completes each one, and the order they must happen in. */
export function StepsStage({ api, view, data, onChange }: Props) {
  const slots = Object.keys(data.layout.slots ?? {});
  const steps = data.steps ?? [];
  const { of, kinds } = groupsOf(data);
  const ro = !view.draft;

  const setSteps = (next: StepData[], renamed?: [string, string]) => {
    const map = { ...of };
    if (renamed) { map[renamed[1]] = map[renamed[0]] ?? 0; delete map[renamed[0]]; }
    onChange({ ...data, steps: next, groups: data.order === "groups" ? buildGroups(next, map, kinds) : data.groups });
  };
  const update = (i: number, patch: Partial<StepData>) => {
    const next = steps.map((s, j) => (j === i ? { ...s, ...patch } : s));
    setSteps(next, patch.id !== undefined ? [steps[i].id, patch.id] : undefined);
  };
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= steps.length) return;
    const next = [...steps];
    [next[i], next[j]] = [next[j], next[i]];
    setSteps(next);
  };
  const setOrder = (order: SopData["order"]) => {
    onChange({ ...data, order, groups: order === "groups" ? (data.groups?.length ? data.groups : [steps.map((s) => s.id)]) : null });
  };
  const setGroupOf = (id: string, g: number) => onChange({ ...data, groups: buildGroups(steps, { ...of, [id]: g }, kinds) });
  const setKind = (g: number, kind: GroupKind) => onChange({ ...data, groups: buildGroups(steps, of, kinds.map((k, i) => (i === g ? kind : k))) });
  const addGroup = () => onChange({ ...data, groups: [...buildGroups(steps, of, kinds) ?? [], []] });
  const generate = async () => {
    const g = await api.generateSteps(view.id);
    onChange({ ...data, ...g });
  };
  const evidenceSlot = (s: StepData) => Object.keys(s.evidence ?? {})[0] ?? "";
  const evidenceState = (s: StepData): Evidence => Object.values(s.evidence ?? {})[0] ?? "installed";

  return (
    <div className="stage">
      <p className="muted">
        Each step completes when its slot shows the evidence (a part installed, a torque result on it); the decision
        model judges the step under way and whether it's done. Generate a starting point from the slots, then edit.
      </p>
      <div className="row">
        <button disabled={ro || slots.length === 0} onClick={generate}>Generate from slots</button>
        <button disabled={ro} onClick={() => setSteps([...steps, { id: `step_${steps.length + 1}`, text: "", evidence: {} }])}>Add step</button>
        <label>Order <select aria-label="Order" value={data.order} disabled={ro} onChange={(e) => setOrder(e.target.value as SopData["order"])}>
          <option value="strict">strict: in list order</option><option value="any">any order</option><option value="groups">groups</option>
        </select></label>
        {data.order === "groups" && <button disabled={ro} onClick={addGroup}>Add group</button>}
      </div>
      {data.order === "groups" && (
        <div className="row">
          {kinds.map((k, g) => (
            <label key={g}>Group {g + 1} <select aria-label={`Group ${g + 1} kind`} value={k} disabled={ro}
              onChange={(e) => setKind(g, e.target.value as GroupKind)}>
              <option value="any">any order</option><option value="sequence">in sequence</option>
            </select></label>
          ))}
          <span className="muted">Groups happen one after another; within a sequence group, in list order.</span>
        </div>
      )}
      <table className="grid-table steps-table" aria-label="Steps">
        <thead><tr><th></th><th>Id</th><th>Label</th><th>Instruction</th><th>Evidence</th><th>Hands off</th>
          {data.order === "groups" && <th>Group</th>}<th></th></tr></thead>
        <tbody>
          {steps.map((s, i) => (
            <tr key={i}>
              <td className="nowrap">
                <button className="icon" aria-label={`Move ${s.id} up`} disabled={ro} onClick={() => move(i, -1)}>↑</button>
                <button className="icon" aria-label={`Move ${s.id} down`} disabled={ro} onClick={() => move(i, 1)}>↓</button>
              </td>
              <td><input aria-label={`Id of step ${i + 1}`} className="mono" value={s.id} disabled={ro} onChange={(e) => update(i, { id: e.target.value })} /></td>
              <td><input aria-label={`Label of ${s.id}`} value={s.label ?? ""} disabled={ro} onChange={(e) => update(i, { label: e.target.value || undefined })} /></td>
              <td><input aria-label={`Instruction of ${s.id}`} className="wide" value={s.text} disabled={ro} onChange={(e) => update(i, { text: e.target.value })} /></td>
              <td className="nowrap">
                <select aria-label={`Evidence slot of ${s.id}`} value={evidenceSlot(s)} disabled={ro}
                  onChange={(e) => update(i, { evidence: e.target.value ? { [e.target.value]: evidenceState(s) } : {} })}>
                  <option value="">none</option>
                  {slots.map((sl) => <option key={sl} value={sl}>{sl}</option>)}
                </select>
                <select aria-label={`Evidence state of ${s.id}`} value={evidenceState(s)} disabled={ro || !evidenceSlot(s)}
                  onChange={(e) => update(i, { evidence: { [evidenceSlot(s)]: e.target.value as Evidence } })}>
                  <option value="installed">installed</option><option value="empty">empty</option><option value="torqued">torqued</option>
                </select>
              </td>
              <td><input type="checkbox" aria-label={`Hands off ${s.id}`} checked={!!s.hands_off} disabled={ro}
                onChange={(e) => update(i, { hands_off: e.target.checked || undefined })} /></td>
              {data.order === "groups" && (
                <td><select aria-label={`Group of ${s.id}`} value={of[s.id] ?? 0} disabled={ro} onChange={(e) => setGroupOf(s.id, Number(e.target.value))}>
                  {kinds.map((_, g) => <option key={g} value={g}>{g + 1}</option>)}
                </select></td>
              )}
              <td><button className="icon" aria-label={`Delete ${s.id}`} disabled={ro}
                onClick={() => setSteps(steps.filter((_, j) => j !== i))}>✕</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <label>Off-SOP means <input className="wide" value={data.off_sop} disabled={ro} onChange={(e) => onChange({ ...data, off_sop: e.target.value })} /></label>
      <label>Idle means <input className="wide" value={data.idle} disabled={ro} onChange={(e) => onChange({ ...data, idle: e.target.value })} /></label>
    </div>
  );
}
