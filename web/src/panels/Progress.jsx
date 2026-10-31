import { fmtP, getProgress } from "../api.js";
import { Dumbbell, HBars, LineChart } from "../charts.jsx";
import { Card, Chip, Guard, Loading, useAsync } from "../ui.jsx";

const DEMOS = [
  ["demo/1_bowl_before_repair", "Bowl, before repair", "the top grasp misses the wide bowl"],
  ["demo/2_bowl_after_repair", "Bowl, after repair", "rim pinch found by search"],
  ["demo/3_stack_program", "Stack by size", "a verified plan program"],
  ["demo/4_plate_created_push", "Plate onto a zone", "a push skill the model wrote"],
  ["demo/5_mug_transfer", "Unseen mug", "the bowl skill, reused"],
  ["demo/6_bar_axis_grasp", "Long bar", "an axis grasp found by search"],
];

function Kpi({ value, label, help, status = "measured" }) {
  return (
    <div className="kpi" title={help}>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label} <Chip status={status} /></div>
    </div>
  );
}

export default function Progress({ open }) {
  const p = useAsync(getProgress, []);
  return (
    <Loading state={p}>
      {(d) => {
        const k = d.kpis, dg = d.diagnosis, tr = d.transfer, cs = d.created;
        const rounds = d.search.map((s, i) => ({
          label: `${i + 1}`, tipTitle: `Round ${i + 1} · ${s.when}`,
          note: s.after_repairs.length > 0,
          tip: s.after_repairs.length ? `after: ${s.after_repairs.join("; ")}` : "no repair since the previous round",
        }));
        return (
          <div className="grid">
            <div className="kpis">
              {k.search_first != null && <Kpi value={<>{k.search_first}<span className="kpi-arrow">→</span>{k.search_last}</>}
                label="real failures per 100 random scenes" help="Failure search with random scenes over all object types, first round vs latest. Only failures an expert could solve count." />}
              <Kpi value={k.repairs} label="repairs passed every gate" help="New skills, routing changes and parameter patches; each fixed its failures, held on held-out objects and caused no regressions." />
              {dg && <Kpi value={`${Math.round((100 * dg.top1_correct) / dg.diagnosed)}%`} label="diagnoses name the right cause"
                          help={`${dg.top1_correct}/${dg.diagnosed} failures: the model's top cause matches the simulator's own label, which the model never sees.`} />}
              {tr && <Kpi value={<>{tr.verify_trials}<span className="kpi-arrow">vs</span>{tr.scratch_trials}</>} label={`trials for an unseen ${tr.template}`}
                          help={`Reusing ${tr.card} from the library (${tr.verify_success}/${tr.verify_trials} verified) vs searching from scratch.`} />}
              {cs && <Kpi value={`${cs.held_out[1]}/${cs.held_out[2]}`} label={`model-written skill (${cs.card})`}
                          help={`Held-out success of a push primitive the model wrote, tuned and gated in simulation (before: ${cs.held_out[0]}/${cs.held_out[2]}). ${cs.rejected_proposals} earlier proposals were rejected by the gates.`} />}
            </div>

            <div className="grid grid-2">
              <Card title="It gets better on its own" status="measured">
                <p className="sub">Real failures found per 100 scenes, each round searching all object types with the library of that time. Lower is better. Hover a round for the repairs before it.</p>
                <LineChart xs={rounds} annotate yLabel="failures per 100 scenes"
                           series={[{ name: "random scenes", cls: "s1", values: d.search.map((s) => s.per100.random) },
                                    { name: "adversarial", cls: "s2", values: d.search.map((s) => s.per100.optuna) }]} />
                <p className="sub">"adversarial" = Optuna steering towards failures. One search per round; vertical lines mark rounds that follow a repair.</p>
              </Card>
              <Card title="Each repair, on objects it never saw" status="measured">
                <p className="sub">Success on held-out objects before and after each change. Hover for the failed cases it fixed and the p-value.</p>
                <Dumbbell rows={d.repairs.map((r) => ({
                  label: r.name.replace(/^new skill /, "").replace(/^patch /, ""), sub: `${r.obj} · ${r.outcome}`,
                  before: r.held_out[0], after: r.held_out[1], n: r.held_out[2],
                  tip: `failed cases fixed ${r.cluster[0]}/${r.cluster[1]} · ${fmtP(r.p)} · ${r.when}`,
                }))} />
              </Card>
            </div>

            <div className="grid grid-2">
              {tr && (
                <Card title={`Reuse vs search from scratch: unseen ${tr.template}`} status="measured">
                  <HBars unit=" trials" rows={[
                    { label: "reuse from the library", value: tr.verify_trials, cls: "s1", tip: `${tr.card} matched by shape, then verified ${tr.verify_success}/${tr.verify_trials}` },
                    { label: "search from scratch", value: tr.scratch_trials, cls: "s0", tip: `${tr.scratch_search} search trials + tuning and gates` },
                  ]} />
                </Card>
              )}
              {dg && (
                <Card title="Diagnosis accuracy per investigation" status="measured">
                  <HBars unit="%" max={100} rows={dg.per_run.filter((r) => r.diagnosed >= 6).map((r) => ({
                    label: `prompt v${r.prompt_version} · ${r.when?.slice(5) || ""}`, sub: `${r.diagnosed} failures`,
                    value: Math.round(100 * r.top1_label_accuracy), cls: r.prompt_version >= 3 ? "s1" : "s0",
                    tip: `${r.confirmed} causes confirmed, ${r.rejected} rejected by one-change replays`,
                  }))} />
                  <p className="sub">Prompts v1-v2 lacked parameter meanings and derived facts (grey).</p>
                </Card>
              )}
            </div>

            <Card title="Watch the demos">
              <Guard>
                <div className="demos">
                  {DEMOS.map(([run, title, sub]) => (
                    <button key={run} className="demo" onClick={() => open({ run, trial: "00001", tab: "episodes" })}>
                      <b>{title}</b><span>{sub}</span>
                    </button>
                  ))}
                </div>
              </Guard>
            </Card>
          </div>
        );
      }}
    </Loading>
  );
}
