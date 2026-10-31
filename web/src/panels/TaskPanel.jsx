import { getRun, getTrial } from "../api.js";
import { Card, Chip, Code, KV, Loading, Ok, RunPicker, useAsync } from "../ui.jsx";

export function Trace({ subtasks }) {
  return (
    <div className="trace">
      {subtasks.map((st, i) => (
        <div key={i} className={`step ${st.ok ? "" : "fail"}`}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
            <b>
              {st.obj} → {st.target || "–"} {st.attempt ? <span className="pill">retry {st.attempt}</span> : null}
            </b>
            <span>
              <Ok ok={st.ok} /> {st.label && <span className="bad">{st.label}</span>}
            </span>
          </div>
          {st.routing && (
            <div className="t">
              grasp choice: {st.routing.card || st.routing.reason || ""}
              {st.routing.group ? ` · shape group ${st.routing.group}` : ""}
              {st.routing.patches ? ` · patches ${JSON.stringify(st.routing.patches)}` : ""}
            </div>
          )}
          {(st.trace || []).map((s, j) => (
            <div key={j} className="t">
              <Ok ok={s.ok} /> {s.skill} <span className="muted">{s.t0}–{s.t1}s · {s.check}</span>
            </div>
          ))}
          {st.constraint_violations && <div className="bad">constraint: {st.constraint_violations.join(", ")}</div>}
        </div>
      ))}
    </div>
  );
}

export default function TaskPanel({ runs, sel, open }) {
  const run = useAsync(() => (sel.run ? getRun(sel.run) : Promise.resolve(null)), [sel.run]);
  const trialName = sel.trial;
  const trial = useAsync(
    () => (sel.run && trialName ? getTrial(sel.run, trialName) : Promise.resolve(null)),
    [sel.run, trialName]
  );
  return (
    <div className="grid">
      <div className="toolbar">
        <RunPicker runs={runs} kinds={["task"]} value={sel.run} onChange={(id) => open({ run: id, trial: null })} />
        <Loading state={run}>
          {(r) =>
            r && (
              <label className="picker">
                <span>Trial</span>
                <select value={trialName || ""} onChange={(e) => open({ trial: e.target.value })}>
                  <option value="" disabled>choose ({r.trials.length})</option>
                  {r.trials.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
            )
          }
        </Loading>
      </div>
      <Loading state={trial}>
        {(t) =>
          t ? (
            <div className="grid grid-2">
              <div className="grid">
                <Card title="Request and task spec" status="measured">
                  <KV rows={[
                    ["request", t.prompt || "–"],
                    ["outcome", <span key="o"><Ok ok={t.success} /> goals {t.goals_met}/{t.goals_total}</span>],
                    ["seed / split", `${t.seed} / ${t.split || "–"}`],
                    ["recovery", t.recovery ? `on (${t.recovery_attempts} retries used)` : "off"],
                  ]} />
                  <h3 style={{ marginTop: 10 }}>Objects</h3>
                  <table>
                    <tbody>
                      {(t.task_spec?.objects || []).map((o) => (
                        <tr key={o.id}><td>{o.id}</td><td>{o.template}</td><td className="muted">{o.role}</td>
                          <td className="muted">{Object.entries(o.params || {}).map(([k, v]) => `${k} ${typeof v === "number" ? v.toFixed(3) : v}`).join(", ")}</td></tr>
                      ))}
                    </tbody>
                  </table>
                  <h3 style={{ marginTop: 10 }}>Goals (checked by code)</h3>
                  <ul>
                    {(t.task_spec?.goal || []).map((g, i) => (
                      <li key={i}>{g.obj} {g.pred} {g.target}{g.upright ? " (upright)" : ""}</li>
                    ))}
                    {(t.task_spec?.constraints || []).map((c, i) => <li key={`c${i}`}>constraint: {c.pred}({c.obj})</li>)}
                  </ul>
                </Card>
                {t.program && (
                  <Card title="Plan program" actions={<Chip status={t.program_error ? "rejected" : "measured"}>
                    {t.program_error ? "program error" : "ran"}</Chip>}>
                    <div className="sub">
                      source: {typeof t.program_source === "object" ? JSON.stringify(t.program_source) : t.program_source || "model"}
                    </div>
                    <Code>{t.program}</Code>
                    {t.program_error && <p className="bad">{t.program_error}</p>}
                  </Card>
                )}
              </div>
              <Card title="What happened, step by step" status="measured"
                    actions={<button className="btn primary" onClick={() => open({ tab: "replay" })}>3D replay</button>}>
                <Trace subtasks={t.subtasks || []} />
              </Card>
            </div>
          ) : (
            <p className="muted">Choose a run and a trial.</p>
          )
        }
      </Loading>
    </div>
  );
}
