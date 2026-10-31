import { useEffect, useState } from "react";
import { getRun, pct } from "../api.js";
import { Card, Chip, Code, KV, Loading, RunPicker, useAsync } from "../ui.jsx";

function testText(t) {
  if (!t) return "";
  if (t.type === "skill_param") return `${t.skill}.${t.param} = ${t.value}`;
  if (t.type === "object_param") return `object ${t.param} ${"value" in t ? `= ${t.value}` : `× ${t.factor}`}`;
  return `use ${t.card} instead`;
}

export default function Investigator({ runs }) {
  const invRuns = runs.filter((r) => r.kind === "investigation");
  const [id, setId] = useState(invRuns[0]?.id || "");
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
          if (!rep) return <p className="muted">Choose an investigation run.</p>;
          const m = rep.metrics;
          const cases = rep.cases.filter((c) => c.hypotheses?.length);
          const c = cases[ci];
          return (
            <div className="grid">
              <div className="grid grid-4">
                <Card title="Top cause matches the simulator" status="measured">
                  <div className="big">{pct(Math.round(m.top1_label_accuracy * m.diagnosed), m.diagnosed)}</div>
                  <div className="sub">{m.diagnosed} failures; the model never sees the simulator's label</div>
                </Card>
                <Card title="Hypotheses tested" status="measured">
                  <div className="big">{m.hypotheses_tested}</div>
                  <div className="sub">
                    <span className="ok">{m.confirmed} confirmed</span> · <span className="bad">{m.rejected} rejected</span>
                    {m.inconclusive_range_limited ? ` · ${m.inconclusive_range_limited} inconclusive` : ""}
                  </div>
                </Card>
                <Card title="Failures explained" status="measured">
                  <div className="big">{m.cases_with_a_confirmed_cause}/{m.diagnosed}</div>
                  <div className="sub">with at least one cause confirmed by a one-change replay</div>
                </Card>
                <Card title="Diagnoser">
                  <div className="big" style={{ fontSize: 20 }}>{rep.backend}</div>
                  <div className="sub">{rep.model} · library {rep.library}</div>
                </Card>
              </div>
              <div className="split">
                <Card title="Failures">
                  <div className="list">
                    {cases.map((x, i) => (
                      <button key={i} className={i === ci ? "on" : ""} onClick={() => setCi(i)}>
                        <span>{x.cluster}<div className="sub">sim: {x.simulator_label}</div></span>
                        <span>
                          {x.hypotheses.some((h) => h.verdict === "confirmed") ? <Chip status="confirmed" /> : <Chip status="rejected">none confirmed</Chip>}
                        </span>
                      </button>
                    ))}
                  </div>
                </Card>
                {c && (
                  <div className="grid">
                    <Card title={`Case ${c.case}: ${c.cluster}`} status="measured">
                      <KV rows={[
                        ["simulator label", <b key="s">{c.simulator_label}</b>],
                        ["model's top label", <span key="t" className={c.top_label_correct ? "ok" : "bad"}>{c.top_label}</span>],
                        ["model summary", c.summary || "–"],
                        ["task", c.task],
                      ]} />
                    </Card>
                    <Card title="Evidence ledger">
                      <ul className="ledger">
                        {c.ledger.map((e, i) => (
                          <li key={i}><Chip status={e.status} /><span>{e.text}{e.note ? <span className="muted"> ({e.note})</span> : null}</span></li>
                        ))}
                      </ul>
                    </Card>
                    <Card title="Hypotheses and one-change replays">
                      <table>
                        <thead><tr><th>cause (model)</th><th>predicted</th><th>one change</th><th>replay</th><th>verdict</th></tr></thead>
                        <tbody>
                          {c.hypotheses.map((h, i) => (
                            <tr key={i}>
                              <td>{h.cause}<div className="sub">confidence {h.confidence ?? "–"}</div></td>
                              <td>{h.predicted_label}</td>
                              <td><span className="pill">{testText(h.test)}</span></td>
                              <td>{h.replay ? (h.replay.step_ok ? "step succeeded" : `still failed (${h.replay.label})`) : "–"}</td>
                              <td><Chip status={h.verdict === "confirmed" ? "confirmed" : h.verdict === "rejected" ? "rejected" : "hypothesis"}>{h.verdict || "untested"}</Chip></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </Card>
                    <Card title="What the model saw (observable evidence only)" status="measured">
                      <Code>{JSON.stringify({ derived: c.evidence?.derived, perceived: c.evidence?.perceived,
                                              observed: c.evidence?.observed, nearby_objects: c.evidence?.nearby_objects,
                                              arm_contacts: c.evidence?.arm_contacts }, null, 1)}</Code>
                    </Card>
                  </div>
                )}
              </div>
            </div>
          );
        }}
      </Loading>
    </div>
  );
}
