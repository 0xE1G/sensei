import { fmtP, getHeadlines, pct } from "../api.js";
import { Card, Chip, Loading, useAsync } from "../ui.jsx";

const DEMOS = [
  ["demo/1_bowl_before_repair", "Bowl, seed skills only", "top grasp misses the wide bowl"],
  ["demo/2_bowl_after_repair", "Bowl, after repair", "discovered rim pinch"],
  ["demo/3_stack_program", "Stack by size", "verified plan program"],
  ["demo/4_plate_created_push", "Plate onto zone", "push skill written by the model"],
  ["demo/5_mug_transfer", "Unseen mug", "bowl skill transferred"],
  ["demo/6_bar_axis_grasp", "Long bar", "axis grasp found by search"],
];

export default function Overview({ open }) {
  const h = useAsync(getHeadlines, []);
  return (
    <Loading state={h}>
      {(d) => (
        <div className="grid">
          <div className="grid grid-4">
            {d.diagnosis && (
              <Card title="Failure diagnosis" status="measured">
                <div className="big">{pct(d.diagnosis.top1_correct, d.diagnosis.diagnosed)}</div>
                <div className="sub">
                  top cause matches the simulator's own label ({d.diagnosis.top1_correct}/{d.diagnosis.diagnosed} failures,
                  model never sees the label). {d.diagnosis.cases_with_confirmed} failures with a cause confirmed by a
                  one-change replay; {d.diagnosis.rejected} hypotheses rejected.
                </div>
              </Card>
            )}
            {d.transfer && (
              <Card title="Transfer to an unseen object" status="measured">
                <div className="big">
                  {d.transfer.verify_trials} <span className="sub">vs</span> {d.transfer.scratch_trials}
                </div>
                <div className="sub">
                  trials to a verified grasp on a {d.transfer.template}: library skill <b>{d.transfer.card}</b> (
                  {d.transfer.verify_success}/{d.transfer.verify_trials} verification) vs from scratch (
                  {d.transfer.scratch_search} search trials)
                </div>
              </Card>
            )}
            {d.created_skill && (
              <Card title="A skill it did not have" status="measured">
                <div className="big">
                  {d.created_skill.held_out[1]}/{d.created_skill.held_out[2]}
                </div>
                <div className="sub">
                  held-out success of <b>{d.created_skill.card}</b>, a push primitive written by the model, tuned and
                  gated in simulation (before: {d.created_skill.held_out[0]}/{d.created_skill.held_out[2]}).{" "}
                  {d.created_skill.rejected_proposals} earlier model-written proposals were rejected by the gates.
                </div>
              </Card>
            )}
            <Card title="Repairs promoted" status="measured">
              <div className="big">{d.repairs.length}</div>
              <div className="sub">
                new cards, routing changes and parameter patches, each through paired tests on held-out objects and
                regression checks
              </div>
            </Card>
          </div>

          <Card title="Demo replays" actions={<span className="sub">saved runs, physics replayed in the browser</span>}>
            <div className="grid grid-3">
              {DEMOS.map(([run, title, sub]) => (
                <button key={run} className="btn" style={{ textAlign: "left", padding: 10 }}
                        onClick={() => open({ run, trial: "00001", tab: "replay" })}>
                  <b>{title}</b>
                  <div className="sub">{sub}</div>
                </button>
              ))}
            </div>
          </Card>

          <div className="grid grid-2">
            <Card title="Repairs (gates passed)" status="measured">
              <table>
                <thead>
                  <tr><th>object</th><th>change</th><th className="num">failed cases fixed</th><th className="num">held-out before → after</th><th>p</th></tr>
                </thead>
                <tbody>
                  {d.repairs.map((r) => (
                    <tr key={r.run} className="click" onClick={() => open({ run: r.run, tab: "experiments" })}>
                      <td>{r.obj}</td>
                      <td>{typeof r.card === "object" ? Object.entries(r.card || {}).map(([k, v]) => `${k} = ${v}`).join(", ") : r.card}
                        <div className="sub">{r.outcome}</div></td>
                      <td className="num">{r.cluster[0]}/{r.cluster[1]}</td>
                      <td className="num">{r.held_out[0]} → {r.held_out[1]} / {r.held_out[2]}</td>
                      <td>{fmtP(r.p)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
            <Card title="Failure search: valid failures per 100 trials" status="measured">
              <p className="sub">Each row is a search over all object types with the library at that time. Fewer is better.</p>
              <table>
                <thead><tr><th>run</th><th>library</th><th className="num">random</th><th className="num">Optuna</th><th className="num">model</th></tr></thead>
                <tbody>
                  {(d.failure_search || []).map((r) => (
                    <tr key={r.run} className="click" onClick={() => open({ run: r.run, tab: "experiments" })}>
                      <td>{r.run.slice(0, 15)}</td><td>{r.library}</td>
                      <td className="num">{r.per100.random}</td><td className="num">{r.per100.optuna}</td>
                      <td className="num">{r.per100.model}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="legend" style={{ marginTop: 8 }}>
                Labels: <Chip status="measured" /> <Chip status="hypothesis" /> <Chip status="confirmed" />
                <Chip status="rejected" />
              </div>
            </Card>
          </div>
        </div>
      )}
    </Loading>
  );
}
