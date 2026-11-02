import { useEffect, useState } from "react";
import { getRun } from "../api.js";
import { HBars } from "../charts.jsx";
import { Card, Chip, Code, Details, Guard, Loading, RunPicker, useAsync } from "../ui.jsx";

const nice = (s) => String(s ?? "").replace(/_/g, " ");

function testText(t) {
  if (!t) return "";
  if (t.type === "skill_param") return `${t.skill}.${t.param} = ${t.value}`;
  if (t.type === "object_param") return `object ${t.param} ${"value" in t ? `= ${t.value}` : `× ${t.factor}`}`;
  return `use ${t.card} instead`;
}

// Derived facts as short readable lines: metres -> cm, booleans -> yes/no.
function fact(k, v) {
  const label = nice(k.replace(/_m$/, ""));
  if (typeof v === "boolean") return v ? label : null;            // show only facts that are true
  if (typeof v === "number" && k.endsWith("_m")) return `${label}: ${(100 * v).toFixed(1)} cm`;
  return `${label}: ${v}`;
}

const VERDICT = { confirmed: "confirmed", rejected: "rejected" };

export default function Investigator({ runs }) {
  const invRuns = runs.filter((r) => r.kind === "investigation");
  // default: the newest standalone investigation (nested ones inside repairs usually hold 1-2 cases)
  const [id, setId] = useState((invRuns.find((r) => !r.id.includes("/")) || invRuns[0])?.id || "");
  const [ci, setCi] = useState(0);
  const run = useAsync(() => (id ? getRun(id) : Promise.resolve(null)), [id]);
  useEffect(() => setCi(0), [id]);
  return (
    <div className="grid">
      <div className="toolbar">
        <RunPicker runs={runs} kinds={["investigation"]} value={id} onChange={setId} label="Investigation" />
      </div>
      <Loading state={run}>
        {(r) => {
          const rep = r?.report;
          if (!rep) return <p className="muted">Choose an investigation.</p>;
          const m = rep.metrics;
          const cases = rep.cases.filter((c) => c.hypotheses?.length);
          const c = cases[ci];
          return (
            <Guard resetKey={id} data={rep}>
              <div className="grid">
                <div className="kpis">
                  <div className="kpi" title="The model never sees the simulator's label"><div className="kpi-value">{Math.round(100 * m.top1_label_accuracy)}%</div>
                    <div className="kpi-label">Top-cause accuracy</div></div>
                  <div className="kpi"><div className="kpi-value">{m.cases_with_a_confirmed_cause}/{m.diagnosed}</div>
                    <div className="kpi-label">Explained failures</div></div>
                  <div className="kpi"><div className="kpi-value"><span className="ok">{m.confirmed}</span> / <span className="bad">{m.rejected}</span></div>
                    <div className="kpi-label">Hypotheses confirmed / rejected</div></div>
                  <div className="kpi"><div className="kpi-value small">{rep.backend}</div><div className="kpi-label">Diagnoser · {rep.model} · {rep.library} library</div></div>
                </div>
                <div className="split">
                  <Card title="Cases">
                    <div className="list">
                      {cases.map((x, i) => (
                        <button key={i} className={i === ci ? "on" : ""} onClick={() => setCi(i)}>
                          <span>{x.cluster}<div className="sub">{nice(x.simulator_label)}</div></span>
                          {(() => { const ok = x.hypotheses.some((h) => h.verdict === "confirmed"); return (
                            <span className={`step-icon ${ok ? "ok" : "fail"}`} title={ok ? "cause confirmed" : "no cause confirmed"}>
                              <svg viewBox="0 0 16 16">{ok ? <path d="M3.5 8.5l3 3 6-7" /> : <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />}</svg>
                            </span>); })()}
                        </button>
                      ))}
                    </div>
                    {m.by_simulator_label && (
                      <Details label="Accuracy by failure type">
                        <HBars unit="%" max={100} rows={Object.entries(m.by_simulator_label).map(([l, v]) => {
                          const [k, n] = String(v).split("/").map(Number);                       // "4/4"
                          return { label: nice(l), sub: `${k}/${n}`, value: n ? Math.round((100 * k) / n) : 0 };
                        })} />
                      </Details>
                    )}
                  </Card>
                  {c && (
                    <Guard resetKey={`${id}/${ci}`} data={c}>
                      <div className="grid">
                        <div className="verdict-line">
                          <span><span className="sub">Simulator label</span> <b>{nice(c.simulator_label)}</b></span>
                          <span><span className="sub">Diagnosis</span> <b className={c.top_label_correct ? "ok" : "bad"}>{nice(c.top_label)} {c.top_label_correct ? "✓" : "✗"}</b></span>
                          <span className="sub">{nice(c.task)} · seed {c.seed}</span>
                        </div>
                        <div className="flow">
                          <section className="card flow-col">
                            <header className="card-head"><h3>Evidence</h3></header>
                            <ul className="facts">
                              {c.ledger.filter((e) => e.status === "measured").slice(0, 4).map((e, i) => <li key={`l${i}`}>{e.text}</li>)}
                              {Object.entries(c.evidence?.derived || {}).map(([k, v]) => fact(k, v)).filter(Boolean).map((f, i) => <li key={`d${i}`}>{f}</li>)}
                              {(c.evidence?.nearby_objects || []).map((o, i) => <li key={`n${i}`}>nearby: {o.id}, {(100 * o.height).toFixed(0)} cm tall, {(100 * o.gap_m).toFixed(1)} cm away</li>)}
                            </ul>
                          </section>
                          <section className="card flow-col wide">
                            <header className="card-head"><h3>Hypotheses</h3><span className="sub">each tested by a one-change replay</span></header>
                            <div className="hyps">
                              {c.hypotheses.map((h, i) => (
                                <div key={i} className={`hyp ${h.verdict || "untested"}`}>
                                  <div className="hyp-top">
                                    <span className="hyp-cause">{h.round === 2 && <span className="pill" title="Proposed after looking at the camera images">second look</span>} {h.cause}</span>
                                    <Chip status={VERDICT[h.verdict] || "hypothesis"}>{h.verdict || "untested"}</Chip>
                                  </div>
                                  <div className="hyp-test">
                                    <code>{testText(h.test)}</code>
                                    <span className="sub">{h.replay ? (h.replay.step_ok ? "step succeeded" : `still failed · ${nice(h.replay.label)}`) : "not replayed"}</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </section>
                        </div>
                        {c.camera && (
                          <section className="card">
                            <header className="card-head"><h3>Camera evidence</h3><span className="sub">second look, only because every first hypothesis was rejected</span></header>
                            <a href={`/api/image?path=${encodeURIComponent(c.camera.image)}`} target="_blank" rel="noreferrer">
                              <img className="evidence-sheet" src={`/api/image?path=${encodeURIComponent(c.camera.image)}`} alt="Key moments of the failing step from the robot's cameras" />
                            </a>
                            <ol className="facts">{(c.camera.moments || []).map((m, i) => <li key={i}>t = {m.t} s · {m.why}</li>)}</ol>
                          </section>
                        )}
                        {c.summary && <p className="caption">{c.summary}</p>}
                        <Details label="Raw evidence">
                          <Code>{JSON.stringify({ derived: c.evidence?.derived, perceived: c.evidence?.perceived, observed: c.evidence?.observed,
                                                  nearby_objects: c.evidence?.nearby_objects, arm_contacts: c.evidence?.arm_contacts }, null, 1)}</Code>
                        </Details>
                      </div>
                    </Guard>
                  )}
                </div>
              </div>
            </Guard>
          );
        }}
      </Loading>
    </div>
  );
}
