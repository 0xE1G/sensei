import { useState } from "react";
import { fmtP, getRun } from "../api.js";
import { Funnel, HBars, Heatmap, RateBar } from "../charts.jsx";
import { Card, Chip, Code, Details, Guard, Loading, useAsync } from "../ui.jsx";

const GROUPS = [
  ["failure_search", "Failure searches", "where does the library fail?"],
  ["explore", "Repair searches", "find a fix for one failure cluster"],
  ["create_skill", "Skill creation", "a capability the library lacked"],
  ["auto_repair", "Auto-repairs", "failures from ordinary runs, repaired"],
  ["transfer", "Transfer", "reuse a skill on an unseen object"],
  ["verify_program", "Program checks", "a plan program verified on fresh scenes"],
  ["recovery", "Recovery tests", "retries inside a run"],
];
const KINDS = GROUPS.map(([k]) => k);
const nice = (s) => String(s ?? "").replace(/_/g, " ");
const GOOD = ["promoted", "patched", "routed", "passed"];
const outcomeChip = (o) => o && (
  <Chip status={GOOD.includes(o) ? "confirmed" : o === "rare_failure" || o === "no_failures" ? "measured" : "rejected"}>
    {{ promoted: "learned", patched: "patched", routed: "rerouted", rare_failure: "too rare", no_failures: "no failures",
       no_candidate_passed_gates: "rejected by gates", no_patch_passed: "no patch passed", failed_gates: "rejected by gates",
       no_working_proposal: "nothing worked", no_feasible_candidates: "no feasible grasp" }[o] || nice(o)}
  </Chip>
);

function Spark({ values, height = 36 }) {
  if (!values.length) return null;
  const w = 220, max = Math.max(...values, 1);
  const pts = values.map((v, i) => `${(i / Math.max(values.length - 1, 1)) * w},${height - (v / max) * (height - 4) - 2}`);
  return <svg className="spark" width={w} height={height}><polyline points={pts.join(" ")} /></svg>;
}

// ------------------------------------------------------------------------------------------- failure search
function FailureSearch({ rep }) {
  const cells = {};
  for (const c of rep.capability_map || []) {
    const row = (cells[c.template] = cells[c.template] || {});
    const cell = (row[c.size] = row[c.size] || { k: 0, n: 0, u: 0 });
    cell.k += c.success; cell.n += c.success + c.valid_failures; cell.u += c.unsolvable;
  }
  const st = Object.entries(rep.strategies || {});
  return (
    <div className="grid">
      <div className="grid grid-2">
        <Card title="Real failures found per 100 scenes" status="measured">
          <HBars rows={st.map(([k, s]) => ({ label: { random: "random scenes", optuna: "Optuna (adversarial)", model: "model-guided" }[k] || k,
            value: s.metrics.valid_failures_per_100, cls: k === "random" ? "s0" : k === "optuna" ? "s2" : "s1",
            sub: `${s.metrics.distinct_clusters} clusters`,
            tip: `${s.metrics.valid_failures} valid failures, ${s.metrics.unsolvable} unsolvable scenes; first found: ${Object.entries(s.metrics.trials_to_find || {}).map(([c, n]) => `${c} @${n}`).join(", ")}` }))} />
          <p className="sub">{rep.budget_per_strategy} scenes per strategy · library: {rep.library}. Only failures an expert could solve count.</p>
        </Card>
        <Card title="Failure clusters handed to the repair search" status="measured">
          <HBars rows={(rep.clusters || []).map((c) => ({ label: c.key, value: c.count, cls: "s2",
            tip: Object.entries(c.labels || {}).map(([l, n]) => `${nice(l)} ×${n}`).join(", ") }))} />
          {rep.strategies?.model?.model_calls && (
            <Details label="the model's reasoning per batch"><ol className="sub">{rep.strategies.model.model_calls.map((c, i) => <li key={i}>{c.rationale}</li>)}</ol></Details>
          )}
        </Card>
      </div>
      <Card title="Capability map: success on solvable scenes, by object and size" status="measured">
        <Heatmap rows={Object.entries(cells).map(([t, row]) => ({ label: t,
          cells: Object.entries(row).sort().map(([size, c]) => ({ label: size, k: c.k, n: c.n, tip: c.u ? `${c.u} unsolvable scenes excluded` : "" })) }))} />
      </Card>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- gates
function GateCard({ g, winner }) {
  const fc = g.fixes_cluster, h = g.held_out, r = g.no_regression;
  return (
    <div className={`gate ${g.pass ? "pass" : "fail"} ${winner ? "winner" : ""}`}>
      <div className="gate-head"><b>{nice(g.card)}</b>{winner && <span className="pill">promoted</span>}<Chip status={g.pass ? "confirmed" : "rejected"}>{g.pass ? "passed" : "failed"}</Chip></div>
      <div className="gate-checks">
        <div className={fc.pass === false ? "bad" : ""} title={fmtP(fc.p_value)}>
          <span>fixes the failures</span><RateBar k={fc.success} n={fc.trials} />
        </div>
        <div className={h.pass === false ? "bad" : ""} title={`${fmtP(h.p_value)}${h.mode ? ` · must ${h.mode}` : ""}`}>
          <span>held-out objects</span><span className="beforeafter"><RateBar k={h.before?.success ?? h.before} n={h.trials} /> → <RateBar k={h.success} n={h.trials} /></span>
        </div>
        <div className={r.pass ? "" : "bad"} title={Object.keys(r.tasks || {}).join(", ")}>
          <span>no regression</span><span>{r.pass ? "✓ none" : "✗ regressed"}</span>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- repair search (explorer)
function Explore({ rep }) {
  const ph = rep.phases || {};
  const winner = rep.promoted?.card?.name;
  const fam = ph.families?.families || [];
  return (
    <div className="grid">
      <Card title="What failed" status="measured">
        <p className="lead">{ph.before?.cluster?.statement || "no failure cluster"}</p>
        {ph.before?.rare && <p className="sub">Measured {ph.before.rare.valid_failures} real failures in {ph.before.rare.trials} scenes: too rare to repair at this budget.</p>}
        {ph.existing && <div className="sub">Existing skills tried first: {ph.existing.candidates.length ? ph.existing.candidates.map((c) => `${c.card} ${c.success}/${c.trials}`).join(", ") : "none fits"}</div>}
        {rep.patched && <div>Patch: <b>{Object.entries(rep.patched.values || {}).map(([k, v]) => `${k} = ${v}`).join(", ")}</b> for {rep.patched.groups?.join(", ")}</div>}
        {rep.routed && <div>Routing: <b>{rep.routed.card}</b> instead of {rep.routed.instead_of}</div>}
      </Card>
      {ph.sample && (
        <div className="grid grid-2">
          <Card title="Grasp candidates" status="measured">
            <Funnel steps={[
              { value: ph.sample.n_sampled, label: "sampled around the object" },
              { value: ph.sample.n_geometric, label: "fit the gripper", tip: Object.entries(ph.sample.reject_reasons || {}).map(([k, v]) => `${nice(k)} ${v}`).join(", ") },
              { value: ph.sample.n_survivors, label: "reachable without collision" },
              ...(fam.length ? [{ value: fam.length, label: "grasp families tried" }] : []),
            ]} />
            <Details label="why candidates were rejected">
              <div className="sub">{Object.entries(ph.sample.reject_reasons || {}).map(([k, v]) => `${nice(k)}: ${v}`).join(" · ")}</div>
            </Details>
          </Card>
          {fam.length > 0 && (
            <Card title="Success by grasp family" status="measured">
              <HBars unit="%" max={100} rows={fam.map((f) => ({ label: nice(f.family), sub: f.primitive || "feature_pinch",
                value: Math.round(100 * f.rate), cls: f.family === winner || nice(f.family) === nice(winner) ? "s1" : "s0",
                tip: `${f.success}/${f.trials} over ${f.candidates} candidates` }))} />
            </Card>
          )}
        </div>
      )}
      {(ph.tune || []).length > 0 && (
        <Card title="Tuning under perturbed physics (Optuna)" status="measured">
          <div className="tunes">
            {ph.tune.map((t, i) => (
              <div key={i} className="tune"><b>{nice(t.family)}</b><Spark values={t.history.map((x) => x.value)} />
                <span className="sub">{t.history.length} trials{t.stopped ? " · stopped early (saturated)" : ""} · confirmation {t.best.confirm}</span></div>
            ))}
          </div>
        </Card>
      )}
      {ph.patch && (
        <Card title="Parameter patch candidates" status="measured">
          <HBars rows={ph.patch.candidates.map((c) => ({ label: c.name, sub: c.source, value: c.confirm.success,
            cls: c.name === ph.patch.chosen ? "s1" : "s0", tip: `${c.confirm.success}/${c.confirm.trials} on fresh objects` }))} />
        </Card>
      )}
      {ph.gates && (
        <Card title="Gates: what a change must pass to be kept" status="measured">
          <div className="gates">{ph.gates.map((g, i) => <GateCard key={i} g={g} winner={g.card === winner} />)}</div>
        </Card>
      )}
      <Details label="raw report"><Code>{JSON.stringify(rep, null, 1).slice(0, 30000)}</Code></Details>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- skill creation
function CreateSkill({ rep }) {
  const ph = rep.phases || {};
  const winner = rep.promoted?.card;
  const searchOf = (p) => (ph.search || []).find((x) => String(x.attempt) === `${p.attempt}${(ph.proposals || []).filter((q) => q.attempt === p.attempt).length > 1 ? "abc"[p.option] : ""}`);
  return (
    <div className="grid">
      <Card title={`Missing capability: ${nice(rep.capability)}`} status="measured">
        <p className="lead">{rep.description || rep.capability}</p>
        <div className="sub">task: {nice(rep.task)} · before: {ph.before?.success}/{ph.before?.trials} with the current library · {rep.trials_simulated} simulated trials</div>
      </Card>
      <div className="proposals">
        {(ph.proposals || []).map((p, i) => {
          const s = searchOf(p);
          const best = s?.candidates?.[0];
          const won = p.card.name === winner;
          return (
            <section key={i} className={`card proposal ${won ? "winner" : ""}`}>
              <header className="card-head"><h3>{nice(p.card.name)}</h3>
                <div className="card-actions">{won ? <Chip status="confirmed">kept</Chip> : <Chip status="hypothesis">attempt {p.attempt}</Chip>}</div></header>
              <Guard data={p}>
                <p className="sub">{p.card.description}</p>
                {s && <>
                  <div className="kv-line"><span>best setting</span><RateBar k={best.success} n={best.trials} /></div>
                  <div className="kv-line"><span>settings that work at all</span><RateBar k={s.candidates.filter((c) => c.success > 0).length} n={s.candidates.length} /></div>
                </>}
                <div className="sub">written by {p.meta?.backend}{p.feedback ? " · after feedback" : ""}</div>
                <Details label="code"><Code>{p.card.code}</Code></Details>
                {p.feedback && <Details label="feedback it received"><Code>{p.feedback}</Code></Details>}
              </Guard>
            </section>
          );
        })}
      </div>
      {ph.gate && (
        <Card title="Gate on held-out scenes" status={ph.gate.pass ? "confirmed" : "rejected"}>
          <div className="gate-checks wide">
            <div><span>held-out scenes</span><span className="beforeafter"><RateBar k={ph.gate.held_out.before} n={ph.gate.held_out.trials} /> → <RateBar k={ph.gate.held_out.success} n={ph.gate.held_out.trials} /></span><span className="sub">{fmtP(ph.gate.held_out.p_value)}</span></div>
            <div><span>no regression</span><span>{ph.gate.no_regression.pass ? "✓ none" : "✗ found"}</span></div>
          </div>
        </Card>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- transfer
function Transfer({ rep }) {
  return (
    <div className="grid grid-2">
      <Card title="Which library skill fits?" actions={<Chip status="hypothesis">model's choice</Chip>}>
        <p className="lead"><b>{rep.retrieval.output.choice}</b>: {rep.retrieval.output.explanation}</p>
        <div className="sub">candidates: {rep.candidates.map((c) => c.card).join(", ")}</div>
      </Card>
      <Card title="Verified, then compared with searching from scratch" status="measured">
        <p>{rep.verification.card}: <RateBar k={rep.verification.success} n={rep.verification.trials} /> on new {rep.template}s{" "}
          <Chip status={rep.verification.trusted ? "confirmed" : "rejected"}>{rep.verification.trusted ? "trusted" : "not trusted"}</Chip></p>
        {rep.scratch && <HBars unit=" trials" rows={[{ label: "reuse", value: rep.verification.trials, cls: "s1" },
                                                     { label: "from scratch", value: rep.scratch.trials_total, cls: "s0" }]} />}
      </Card>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- auto-repair
function AutoRepair({ rep, open }) {
  return (
    <Card title="Failure clusters from ordinary runs" status="measured">
      <table>
        <thead><tr><th>cluster</th><th className="num">real failures</th><th>diagnosis</th><th>outcome</th></tr></thead>
        <tbody>{(rep.clusters || []).map((c, i) => (
          <tr key={i} className={c.repair?.run_id ? "click" : ""} onClick={() => c.repair?.run_id && open({ run: c.repair.run_id, tab: "experiments" })}>
            <td>{nice(c.kind)} / {c.skill}<div className="sub">{Object.entries(c.labels || {}).map(([l, n]) => `${nice(l)} ×${n}`).join(", ")}</div></td>
            <td className="num">{c.valid_failures}</td>
            <td className="sub">{c.investigation ? `${c.investigation.confirmed} confirmed, ${c.investigation.rejected} rejected` : "–"}</td>
            <td>{c.repair ? outcomeChip(c.repair.outcome) : <span className="sub">{c.action}</span>}</td>
          </tr>
        ))}</tbody>
      </table>
    </Card>
  );
}

function Generic({ rep }) {
  return <Card title="Report"><Details label="raw report" ><Code>{JSON.stringify(rep, null, 1).slice(0, 30000)}</Code></Details></Card>;
}

const VIEWS = { failure_search: FailureSearch, explore: Explore, create_skill: CreateSkill, transfer: Transfer, auto_repair: AutoRepair };

export default function Experiments({ runs, open, sel }) {
  const list = runs.filter((r) => KINDS.includes(r.kind));
  const fromSel = list.find((r) => r.id === sel?.run);
  const firstGood = list.find((r) => r.kind === "explore" && GOOD.includes(r.outcome));
  const [id, setId] = useState((fromSel || firstGood || list.find((r) => r.kind === "explore") || list[0])?.id || "");
  const [showAll, setShowAll] = useState(false);
  const run = useAsync(() => (id ? getRun(id) : Promise.resolve(null)), [id]);
  return (
    <div className="split exp-split">
      <Card title="Experiments">
        <div className="exp-list">
          {GROUPS.map(([k, title, sub]) => {
            const items = list.filter((r) => r.kind === k);
            if (!items.length) return null;
            const shown = showAll ? items : items.slice(0, 6);
            return (
              <div key={k}>
                <div className="exp-group" title={sub}>{title} <span className="sub">{items.length}</span></div>
                {shown.map((r) => (
                  <button key={r.id} className={r.id === id ? "on" : ""} onClick={() => setId(r.id)}>
                    <span className="exp-title" title={r.title}>{(r.title || r.id).replace(/^[^·]+· /, "")}</span>
                    {outcomeChip(r.outcome)}
                  </button>
                ))}
              </div>
            );
          })}
          <button className="link" onClick={() => setShowAll((s) => !s)}>{showAll ? "show fewer" : "show all"}</button>
        </div>
      </Card>
      <Loading state={run}>
        {(r) => {
          if (!r?.report) return <p className="muted">Choose an experiment.</p>;
          const View = VIEWS[r.kind] || Generic;
          return (
            <div className="grid">
              <div className="exp-header"><h2>{r.title}</h2>{outcomeChip(r.report.outcome)}</div>
              <Guard resetKey={id} data={r.report}><View rep={r.report} open={open} /></Guard>
            </div>
          );
        }}
      </Loading>
    </div>
  );
}
